"""Run Modephere's conformer stage on the GPU (nvMolKit) next to the CPU one.

Usage (from the project root)::

    uv run python scripts/nvmolkit_conformers.py ibuprofen warfarin imatinib
    uv run python scripts/nvmolkit_conformers.py "CC(=O)Oc1ccccc1C(=O)O" --repeat
    uv run python scripts/nvmolkit_conformers.py --library 100 --seed 7
    uv run python scripts/nvmolkit_conformers.py --library 200 --precision single \
        --minimizer fire --csv /tmp/gpu_vs_cpu.csv

Both backends run the same steps of ``modephere.library.conformers`` (PLAN
C2.2) with the same :class:`~modephere.types.ConformerParams`:

1. stereoisomers: ``_stereoisomers`` (CPU, shared);
2. embedding: RDKit ``EmbedMultipleConfs`` per stereoisomer (CPU) or one
   nvMolKit ``EmbedMolecules`` call per conformer count over every
   stereoisomer of every compound (GPU); same ETKDG parameters, seed and
   embed-time pruning, and the same random-coordinates retry;
3. force field: ``_resolve_force_field`` (shared), then RDKit
   ``MMFFOptimizeMoleculeConfs`` / ``UFFOptimizeMoleculeConfs`` per
   stereoisomer (CPU) or nvMolKit ``MMFFBatchedForcefield`` /
   ``UFFBatchedForcefield`` over the whole batch (GPU), with the same variant
   (MMFF94s), iteration cap, gradient tolerance and UFF selenium retyping;
4. energy window and 5. pruning: the shared ``_prune`` (RDKit, symmetry-aware
   ``GetBestRMS``), so differences come only from steps 2-3.

Per compound it reports how many conformers each backend embedded and kept,
the lowest energy each found, convergence, and how well the two ensembles
cover each other (a conformer is "covered" when the other backend kept one
of the same stereoisomer within ``prune_rms_threshold``, symmetry-aware
heavy-atom RMSD). Checks on the GPU side:

* ``E check``: largest gap between the energy nvMolKit reports and RDKit's
  own force field evaluated on the GPU coordinates (same variant);
* ``nv-RMSD kept``: how many conformers pruning would keep with nvMolKit's
  ``GetConformerRMSMatrix`` (fixed atom order, no symmetry) instead of
  ``GetBestRMS``; more kept = symmetric duplicates it cannot see;
* ``--repeat``: runs the GPU pipeline a second time with the same seed:
  ``rep same`` (identical ensemble, coordinates included), ``rep kept`` and
  ``rep dEmin`` (second run minus first).

Baseline: two CPU runs with different seeds also disagree (ETKDG is random
sampling), so the CPU pipeline is run again with ``random_seed + 1`` and the
same numbers are shown for it (``base kept``, ``base dEmin``, ``cpu in base``,
``base in cpu``). A GPU difference only matters when it is larger than that
CPU-vs-CPU noise. ``--no-baseline`` skips it (halves the CPU time).

Timings: the CPU runs compound by compound, as the library builder does; the
GPU runs the whole set as one batch (that is how it is meant to be used), after
a warm-up that pays the CUDA start-up. Nothing is written to the library.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from _common import (
    chembl_lookup,
    kv,
    looks_like_smiles,
    make_query,
    note,
    pubchem_client,
    section,
    show,
    subsection,
)
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolAlign, rdMolDescriptors

from modephere.config import get_settings
from modephere.core import CompoundError
from modephere.library.conformers import (
    _MAX_RMS_MATCHES,
    _Candidate,
    _embed,
    _optimise,
    _prune,
    _resolve_force_field,
    _stereoisomers,
    _uff_view,
)
from modephere.library.ingest.resolve import Resolver
from modephere.library.standardize import standardize
from modephere.types import ConformerParams, ForceField, SmilesQuery

if TYPE_CHECKING:
    from nvmolkit.types import HardwareOptions

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass
class Item:
    """One compound, standardized; ``isomers`` are shared by both backends."""

    label: str
    inchikey: str
    heavy: int
    isomers: list[Chem.Mol]
    count: int  # embeddings attempted per stereoisomer


@dataclass
class Outcome:
    """One backend's ensemble for one compound."""

    embedded: int = 0
    force_field: ForceField | None = None
    windowed: list[_Candidate] = field(default_factory=list)  # energy order, pre-prune
    kept: list[_Candidate] = field(default_factory=list)
    error: str | None = None
    seconds: float = math.nan  # CPU only: the whole stage for this compound


def load_items(args: argparse.Namespace, params: ConformerParams) -> list[Item]:
    queries: list[tuple[str, object]] = [(q, make_query(q)) for q in args.queries]
    if args.library:
        path = args.library_path or get_settings().paths.library / "compounds.parquet"
        reg = pl.read_parquet(path).select("inchikey", "working_smiles", "title")
        for r in reg.sample(min(args.library, reg.height), seed=args.seed).iter_rows(
            named=True
        ):
            queries.append(
                (r["title"] or r["inchikey"], SmilesQuery(smiles=r["working_smiles"]))
            )
    needs_lookup = any(not looks_like_smiles(q) for q in args.queries)
    chembl = chembl_lookup() if needs_lookup else None
    items: list[Item] = []
    with pubchem_client(offline=args.offline) as client:
        resolver = Resolver(client, chembl=chembl)
        for label, query in queries:
            try:
                std = standardize(resolver.resolve(query))
            except CompoundError as e:
                note(f"{label}: skipped ({e})")
                continue
            mol = std.microspecies
            items.append(
                Item(
                    label=label,
                    inchikey=std.record.inchikey,
                    heavy=mol.GetNumAtoms(),
                    isomers=_stereoisomers(mol, params),
                    count=params.target_count(
                        rdMolDescriptors.CalcNumRotatableBonds(mol)
                    ),
                )
            )
    return items


# --------------------------------------------------------------------------- #
# Shared steps 3-5
# --------------------------------------------------------------------------- #

Embedded = list[tuple[int, Chem.Mol, list[int]]]  # (stereoisomer, mol_h, conf ids)


def resolve_force_field(embedded: Embedded, params: ConformerParams) -> ForceField:
    """Same rule as ``generate_conformers``: one force field for all isomers."""
    chosen: ForceField | None = None
    for _, mol_h, _ in embedded:
        ff = _resolve_force_field(mol_h, params)
        if chosen is not None and ff is not chosen:
            msg = f"stereoisomers need different force fields ({chosen}, {ff})"
            raise ValueError(msg)
        chosen = ff
    assert chosen is not None
    return chosen


def finish(
    out: Outcome,
    embedded: Embedded,
    results: list[list[tuple[bool, float | None]]],
    params: ConformerParams,
) -> None:
    """Energy sort, energy window and pruning, as in ``generate_conformers``."""
    candidates = [
        _Candidate(s, mol_h, cid, energy, converged)
        for (s, mol_h, ids), res in zip(embedded, results, strict=True)
        for cid, (converged, energy) in zip(ids, res, strict=True)
    ]
    out.embedded = len(candidates)
    if out.force_field is not ForceField.NONE:
        candidates.sort(key=lambda c: (c.energy, c.stereoisomer, c.conf_id))
        if params.energy_window is not None:
            floor = candidates[0].energy or 0.0
            candidates = [
                c
                for c in candidates
                if (c.energy or 0.0) - floor <= params.energy_window
            ]
    out.windowed = candidates
    out.kept = _prune(candidates, params.prune_rms_threshold)


# --------------------------------------------------------------------------- #
# CPU: Modephere as it is (RDKit, compound by compound)
# --------------------------------------------------------------------------- #


def run_cpu(item: Item, params: ConformerParams) -> Outcome:
    start = time.perf_counter()
    out = _run_cpu(item, params)
    out.seconds = time.perf_counter() - start  # failures count too
    return out


def _run_cpu(item: Item, params: ConformerParams) -> Outcome:
    out = Outcome()
    embedded: Embedded = []
    for s, isomer in enumerate(item.isomers):
        mol_h = Chem.AddHs(isomer)
        ids = _embed(mol_h, params, item.count)
        if ids:
            embedded.append((s, mol_h, ids))
    if not embedded:
        out.error = "no conformer embedded"
        return out
    try:
        out.force_field = resolve_force_field(embedded, params)
    except (CompoundError, ValueError) as e:
        out.error = str(e)
        return out
    results = [_optimise(mol_h, out.force_field, params) for _, mol_h, _ in embedded]
    finish(out, embedded, results, params)
    return out


# --------------------------------------------------------------------------- #
# GPU: nvMolKit, the whole set as one batch
# --------------------------------------------------------------------------- #


@dataclass
class GpuOptions:
    hardware: HardwareOptions | None
    precision: object  # nvmolkit PrecisionMode
    minimizer: str  # "BFGS" or "FIRE"


def _embed_params(
    params: ConformerParams, *, random_coords: bool
) -> rdDistGeom.EmbedParameters:
    """The ETKDG parameters of ``conformers._embed``."""
    ps = getattr(rdDistGeom, params.method.value)()
    ps.randomSeed = params.random_seed
    ps.pruneRmsThresh = params.prune_rms_threshold
    ps.onlyHeavyAtomsForRMS = True
    ps.useSymmetryForPruning = True
    ps.useRandomCoords = random_coords
    return ps


def gpu_embed(
    jobs: list[tuple[Chem.Mol, int]], params: ConformerParams, opts: GpuOptions
) -> None:
    """Embed every (mol_h, count) in place; one call per distinct count."""
    from nvmolkit.embedMolecules import EmbedMolecules

    def batch(todo: list[tuple[Chem.Mol, int]], *, random_coords: bool) -> None:
        by_count: dict[int, list[Chem.Mol]] = defaultdict(list)
        for mol_h, count in todo:
            by_count[count].append(mol_h)
        ps = _embed_params(params, random_coords=random_coords)
        for count, mols in sorted(by_count.items()):
            EmbedMolecules(
                mols,
                ps,
                confsPerMolecule=count,
                hardwareOptions=opts.hardware,
                precision=opts.precision,
            )

    batch(jobs, random_coords=False)
    if params.use_random_coords_fallback:
        retry = [(m, c) for m, c in jobs if m.GetNumConformers() == 0]
        if retry:
            batch(retry, random_coords=True)


def gpu_optimise(
    jobs: list[tuple[Chem.Mol, ForceField]], params: ConformerParams, opts: GpuOptions
) -> list[list[tuple[bool, float | None]]]:
    """Minimise every conformer of every mol_h in place, batched per force field."""
    from nvmolkit.batchedForcefield import MMFFBatchedForcefield, UFFBatchedForcefield

    results: list[list[tuple[bool, float | None]] | None] = [None] * len(jobs)
    for ff in {ff for _, ff in jobs}:
        idx = [i for i, (_, f) in enumerate(jobs) if f is ff]
        mols = [jobs[i][0] for i in idx]
        if ff is ForceField.NONE:
            for i, m in zip(idx, mols, strict=True):
                results[i] = [(True, None)] * m.GetNumConformers()
            continue
        if ff is ForceField.UFF:
            views = [_uff_view(m) for m in mols]
            engine = UFFBatchedForcefield(
                views, hardwareOptions=opts.hardware, precision=opts.precision
            )
        else:
            views = mols
            props = [
                rdForceFieldHelpers.MMFFGetMoleculeProperties(m, mmffVariant=ff.value)
                for m in mols
            ]
            engine = MMFFBatchedForcefield(
                mols,
                properties=props,
                hardwareOptions=opts.hardware,
                precision=opts.precision,
            )
        energies, converged = engine.minimize(
            maxIters=params.max_iterations, minimizerKind=opts.minimizer
        )
        for i, m, view, e, c in zip(idx, mols, views, energies, converged, strict=True):
            if view is not m:  # UFF on the retyped copy: copy coordinates back
                for conf in view.GetConformers():
                    m.GetConformer(conf.GetId()).SetPositions(conf.GetPositions())
            results[i] = [(bool(ok), float(en)) for ok, en in zip(c, e, strict=True)]
    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        msg = f"no optimisation result for jobs {missing}"
        raise RuntimeError(msg)
    return [r for r in results if r is not None]


def run_gpu(
    items: list[Item], params: ConformerParams, opts: GpuOptions
) -> tuple[list[Outcome], dict[str, float]]:
    """The whole set as one batch; returns outcomes and stage timings (s)."""
    timing: dict[str, float] = {}
    outs = [Outcome() for _ in items]
    mols = [[Chem.AddHs(iso) for iso in item.isomers] for item in items]

    t = time.perf_counter()
    gpu_embed(
        [(m, item.count) for item, ms in zip(items, mols, strict=True) for m in ms],
        params,
        opts,
    )
    timing["embed"] = time.perf_counter() - t

    t = time.perf_counter()
    embedded_of: list[Embedded] = []
    jobs: list[tuple[Chem.Mol, ForceField]] = []
    for out, ms in zip(outs, mols, strict=True):
        embedded = [
            (s, m, [c.GetId() for c in m.GetConformers()])
            for s, m in enumerate(ms)
            if m.GetNumConformers()
        ]
        embedded_of.append(embedded)
        if not embedded:
            out.error = "no conformer embedded"
            continue
        try:
            out.force_field = resolve_force_field(embedded, params)
        except (CompoundError, ValueError) as e:
            out.error = str(e)
            continue
        jobs += [(m, out.force_field) for _, m, _ in embedded]
    timing["force field typing"] = time.perf_counter() - t

    t = time.perf_counter()
    flat = gpu_optimise(jobs, params, opts) if jobs else []
    timing["optimise"] = time.perf_counter() - t

    t = time.perf_counter()
    k = 0
    for out, embedded in zip(outs, embedded_of, strict=True):
        if out.error is not None:
            continue
        results = flat[k : k + len(embedded)]
        k += len(embedded)
        finish(out, embedded, results, params)
    timing["window + prune (RDKit)"] = time.perf_counter() - t
    return outs, timing


def warm_up(params: ConformerParams, opts: GpuOptions) -> float:
    """Pay CUDA start-up and kernel loading outside the timed run."""
    start = time.perf_counter()
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOc1ccccc1C(=O)N"))
    gpu_embed([(mol, 4)], params, opts)
    gpu_optimise([(mol, params.force_field)], params, opts)
    return time.perf_counter() - start


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #


def _heavy(cands: list[_Candidate]) -> dict[int, Chem.Mol]:
    return {c.stereoisomer: Chem.RemoveHs(c.mol) for c in cands}


def coverage(
    a: list[_Candidate], b: list[_Candidate], threshold: float
) -> tuple[float, float]:
    """Fraction of ``a`` with a same-stereoisomer twin in ``b`` (and of ``b`` in ``a``)."""
    ha, hb = _heavy(a), _heavy(b)
    best_a = [math.inf] * len(a)
    best_b = [math.inf] * len(b)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            if ca.stereoisomer != cb.stereoisomer:
                continue
            d = rdMolAlign.GetBestRMS(
                hb[cb.stereoisomer],
                ha[ca.stereoisomer],
                prbId=cb.conf_id,
                refId=ca.conf_id,
                maxMatches=_MAX_RMS_MATCHES,
            )
            best_a[i] = min(best_a[i], d)
            best_b[j] = min(best_b[j], d)
    return _within(best_a, threshold), _within(best_b, threshold)


def _within(best: list[float], threshold: float) -> float:
    return sum(d <= threshold for d in best) / len(best) if best else math.nan


def energy_check(kept: list[_Candidate], ff: ForceField | None) -> float:
    """Largest |nvMolKit energy - RDKit energy on the same coordinates|."""
    if ff in (None, ForceField.NONE):
        return math.nan
    worst = 0.0
    for c in kept:
        # The RDKit force field points into ``mol``'s coordinates: keep ``mol``
        # referenced until CalcEnergy returns (a temporary UFF view segfaults).
        mol = _uff_view(c.mol) if ff is ForceField.UFF else c.mol
        if ff is ForceField.UFF:
            engine = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=c.conf_id)
        else:
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(
                mol, mmffVariant=ff.value
            )
            engine = rdForceFieldHelpers.MMFFGetMoleculeForceField(
                mol, props, confId=c.conf_id
            )
        worst = max(worst, abs(engine.CalcEnergy() - (c.energy or 0.0)))
    return worst


def nv_rmsd_kept(out: Outcome, threshold: float) -> int:
    """Kept count if ``_prune`` used nvMolKit's RMSD matrix (no symmetry)."""
    from nvmolkit.conformerRmsd import GetConformerRMSMatrix

    cands = out.windowed
    groups: dict[int, list[_Candidate]] = defaultdict(list)
    for c in cands:
        groups[c.stereoisomer].append(c)
    rms_of = {}
    for s, group in groups.items():
        if len(group) < 2:
            continue
        heavy = Chem.RemoveHs(group[0].mol)
        want = {c.conf_id for c in group}
        for cid in [c.GetId() for c in heavy.GetConformers()]:
            if cid not in want:
                heavy.RemoveConformer(cid)
        pos = {conf.GetId(): k for k, conf in enumerate(heavy.GetConformers())}
        flat = GetConformerRMSMatrix(heavy).numpy()
        rms_of[s] = (flat, pos)
    kept: list[_Candidate] = []
    for c in cands:
        entry = rms_of.get(c.stereoisomer)
        dup = False
        if entry is not None:
            flat, pos = entry
            for o in kept:
                if o.stereoisomer != c.stereoisomer:
                    continue
                i, j = sorted((pos[c.conf_id], pos[o.conf_id]), reverse=True)
                if flat[i * (i - 1) // 2 + j] <= threshold:
                    dup = True
                    break
        if not dup:
            kept.append(c)
    return len(kept)


def same_ensemble(a: Outcome, b: Outcome) -> bool:
    """Same kept conformers, in the same order, with identical coordinates."""
    if a.error or b.error or len(a.kept) != len(b.kept):
        return False
    for ca, cb in zip(a.kept, b.kept, strict=True):
        if (ca.stereoisomer, ca.conf_id) != (cb.stereoisomer, cb.conf_id):
            return False
        pa = ca.mol.GetConformer(ca.conf_id).GetPositions()
        pb = cb.mol.GetConformer(cb.conf_id).GetPositions()
        if not np.array_equal(pa, pb):
            return False
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _min_energy(out: Outcome) -> float:
    energies = [c.energy for c in out.kept if c.energy is not None]
    return min(energies) if energies else math.nan


def _converged(out: Outcome) -> float:
    return sum(c.converged for c in out.kept) / len(out.kept) if out.kept else math.nan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("queries", nargs="*", help="names, CIDs, ChEMBL IDs or SMILES")
    p.add_argument(
        "--library",
        type=int,
        default=0,
        metavar="N",
        help="add N random compounds from the built library",
    )
    p.add_argument("--library-path", help="compounds.parquet (default: settings path)")
    p.add_argument("--seed", type=int, default=7, help="sampling seed for --library")
    p.add_argument("--offline", action="store_true", help="no PubChem network calls")
    p.add_argument("--precision", choices=("full", "single"), default="full")
    p.add_argument("--minimizer", choices=("bfgs", "fire"), default="bfgs")
    p.add_argument(
        "--batch-size", type=int, default=-1, help="nvMolKit HardwareOptions"
    )
    p.add_argument(
        "--repeat",
        action="store_true",
        help="run the GPU twice and check the seed reproduces the ensemble",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the second CPU run (seed + 1) that measures CPU-vs-CPU noise",
    )
    p.add_argument(
        "--no-coverage",
        action="store_true",
        help="skip the CPU/GPU coverage RMSDs (slow on big symmetric molecules)",
    )
    p.add_argument("--csv", help="write the per-compound table here")
    p.add_argument("--limit", type=int, default=40, help="table rows to print")
    args = p.parse_args()
    if not args.queries and not args.library:
        p.error("give compounds or --library N")
    return args


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    import nvmolkit
    import torch
    from nvmolkit.types import HardwareOptions, PrecisionMode

    params = ConformerParams()
    opts = GpuOptions(
        hardware=HardwareOptions(batchSize=args.batch_size),
        precision=PrecisionMode.SINGLE
        if args.precision == "single"
        else PrecisionMode.FULL,
        minimizer=args.minimizer.upper(),
    )

    section("nvMolKit vs RDKit: conformer stage")
    import rdkit

    kv(
        {
            "RDKit": rdkit.__version__,
            "nvMolKit": nvmolkit.__version__,
            "torch / CUDA": f"{torch.__version__} / {torch.version.cuda}",
            "GPU": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "none",
            "GPU options": f"precision={args.precision}, minimizer={opts.minimizer}, "
            f"batchSize={args.batch_size}",
            "ConformerParams": params.model_dump(mode="json"),
        }
    )

    subsection("Inputs")
    items = load_items(args, params)
    if not items:
        note("nothing to run")
        return
    kv(
        {
            "compounds": len(items),
            "stereoisomers": sum(len(i.isomers) for i in items),
            "embeddings attempted": sum(len(i.isomers) * i.count for i in items),
        }
    )

    subsection("CPU (RDKit, compound by compound)")
    cpu = []
    for n, item in enumerate(items, 1):
        cpu.append(run_cpu(item, params))
        print(f"\r  {n}/{len(items)}", end="", flush=True)
    print()
    cpu_total = sum(o.seconds for o in cpu if not math.isnan(o.seconds))
    kv({"total": f"{cpu_total:.2f} s"})

    base = None
    if not args.no_baseline:
        subsection("CPU baseline (random_seed + 1)")
        seed2 = params.model_copy(update={"random_seed": params.random_seed + 1})
        base = []
        for n, item in enumerate(items, 1):
            base.append(run_cpu(item, seed2))
            print(f"\r  {n}/{len(items)}", end="", flush=True)
        print()

    subsection("GPU (nvMolKit, one batch)")
    kv({"warm-up": f"{warm_up(params, opts):.2f} s (not counted)"})
    gpu, timing = run_gpu(items, params, opts)
    gpu_total = sum(timing.values())
    kv({**{k: f"{v:.2f} s" for k, v in timing.items()}, "total": f"{gpu_total:.2f} s"})

    repeat = None
    if args.repeat:
        repeat, _ = run_gpu(items, params, opts)

    subsection("Comparing")
    df = compare(
        items,
        cpu,
        gpu,
        params,
        base=base,
        repeat=repeat,
        coverage_on=not args.no_coverage,
    )
    print_tables(df, args.limit)
    summarize(df, cpu_total, gpu_total)
    if args.csv:
        df.write_csv(args.csv)
        note(f"table written to {args.csv}")


def compare(
    items: list[Item],
    cpu: list[Outcome],
    gpu: list[Outcome],
    params: ConformerParams,
    *,
    base: list[Outcome] | None,
    repeat: list[Outcome] | None,
    coverage_on: bool,
) -> pl.DataFrame:
    """One row per compound; columns documented in the module docstring."""
    rows = []
    for n, (item, c, g) in enumerate(zip(items, cpu, gpu, strict=True)):
        print(f"\r  {n + 1}/{len(items)}", end="", flush=True)
        row = {
            "compound": item.label,
            "heavy": item.heavy,
            "iso": len(item.isomers),
            "tried": len(item.isomers) * item.count,
            "cpu emb": c.embedded,
            "gpu emb": g.embedded,
            "cpu kept": len(c.kept),
            "gpu kept": len(g.kept),
            "cpu conv": _converged(c),
            "gpu conv": _converged(g),
            "cpu s": c.seconds,
            "cpu Emin": _min_energy(c),
            "gpu Emin": _min_energy(g),
            "error": c.error or g.error,
        }
        row["dEmin"] = row["gpu Emin"] - row["cpu Emin"]
        if not row["error"]:
            row["E check"] = energy_check(g.kept, g.force_field)
            if coverage_on:
                row["cpu in gpu"], row["gpu in cpu"] = coverage(
                    c.kept, g.kept, params.prune_rms_threshold
                )
            row["nv-RMSD kept"] = nv_rmsd_kept(g, params.prune_rms_threshold)
        if base is not None:
            b = base[n]
            row["base kept"] = len(b.kept)
            row["base dEmin"] = _min_energy(b) - row["cpu Emin"]
            if coverage_on and not (row["error"] or b.error):
                row["cpu in base"], row["base in cpu"] = coverage(
                    c.kept, b.kept, params.prune_rms_threshold
                )
        if repeat is not None:
            r = repeat[n]
            row["rep same"] = same_ensemble(g, r)
            row["rep kept"] = len(r.kept)
            row["rep dEmin"] = _min_energy(r) - row["gpu Emin"]
        rows.append(row)
    print()
    return pl.DataFrame(rows, infer_schema_length=None)


_ENSEMBLE_COLS = (
    "compound",
    "heavy",
    "iso",
    "tried",
    "cpu emb",
    "gpu emb",
    "cpu kept",
    "gpu kept",
    "cpu conv",
    "gpu conv",
    "cpu s",
)
_AGREEMENT_COLS = (
    "compound",
    "cpu Emin",
    "gpu Emin",
    "dEmin",
    "E check",
    "cpu in gpu",
    "gpu in cpu",
    "nv-RMSD kept",
)
_NOISE_COLS = (
    "compound",
    "cpu kept",
    "base kept",
    "base dEmin",
    "cpu in base",
    "base in cpu",
    "gpu kept",
    "rep kept",
    "rep dEmin",
    "rep same",
)


def print_tables(df: pl.DataFrame, limit: int) -> None:
    rounded = df.with_columns(pl.col(pl.Float64).round(3))
    show(
        rounded.select(c for c in _ENSEMBLE_COLS if c in df.columns),
        rows=limit,
        title="Ensembles (emb = embedded, conv = fraction of kept converged)",
        max_str=28,
    )
    show(
        rounded.select(c for c in _AGREEMENT_COLS if c in df.columns),
        rows=limit,
        title="GPU vs CPU (kcal/mol; dEmin < 0: GPU found a lower minimum)",
        max_str=28,
    )
    noise = [c for c in _NOISE_COLS if c in df.columns]
    if len(noise) > 3:
        show(
            rounded.select(noise),
            rows=limit,
            title="Noise: CPU with seed + 1 (base) and GPU run twice (rep)",
            max_str=28,
        )
    failed = df.filter(pl.col("error").is_not_null())
    if failed.height:
        show(failed.select("compound", "cpu emb", "gpu emb", "error"), title="Failures")


def summarize(df: pl.DataFrame, cpu_total: float, gpu_total: float) -> None:
    ok = df.filter(pl.col("error").is_null())
    section("Summary")
    summary = {
        "compounds compared": f"{ok.height} of {df.height}",
        "conformers embedded (cpu / gpu)": f"{ok['cpu emb'].sum()} / {ok['gpu emb'].sum()}",
        "conformers kept (cpu / gpu)": f"{ok['cpu kept'].sum()} / {ok['gpu kept'].sum()}",
        "same kept count": f"{(ok['cpu kept'] == ok['gpu kept']).sum()} of {ok.height}",
        "GPU lower / higher minimum (> 0.01 kcal/mol)": (
            f"{(ok['dEmin'] < -0.01).sum()} / {(ok['dEmin'] > 0.01).sum()}"
        ),
        "median |dEmin| (kcal/mol)": _median(ok["dEmin"].abs()),
        "converged (cpu / gpu)": (
            f"{_mean(ok['cpu conv']):.1%} / {_mean(ok['gpu conv']):.1%}"
        ),
    }
    if "E check" in ok.columns:
        summary["worst E check (kcal/mol)"] = _max(ok["E check"])
    if "cpu in gpu" in ok.columns:
        summary["coverage: cpu in gpu / gpu in cpu"] = (
            f"{_mean(ok['cpu in gpu']):.1%} / {_mean(ok['gpu in cpu']):.1%}"
        )
    if "nv-RMSD kept" in ok.columns:
        extra = (ok["nv-RMSD kept"] - ok["gpu kept"]).sum()
        summary["nvMolKit RMSD pruning would keep"] = (
            f"{ok['nv-RMSD kept'].sum()} ({extra:+d} vs GetBestRMS)"
        )
    if "base kept" in ok.columns:
        summary["baseline CPU seed+1: kept"] = (
            f"{ok['base kept'].sum()} (same count as CPU in "
            f"{(ok['base kept'] == ok['cpu kept']).sum()} of {ok.height})"
        )
        summary["baseline CPU seed+1: lower / higher minimum"] = (
            f"{(ok['base dEmin'] < -0.01).sum()} / {(ok['base dEmin'] > 0.01).sum()}, "
            f"median |dEmin| {_median(ok['base dEmin'].abs())}"
        )
        if "cpu in base" in ok.columns:
            summary["baseline coverage: cpu in base / base in cpu"] = (
                f"{_mean(ok['cpu in base']):.1%} / {_mean(ok['base in cpu']):.1%}"
            )
    if "rep same" in ok.columns:
        summary["GPU run twice: identical ensembles"] = (
            f"{ok['rep same'].sum()} of {ok.height}; kept count changed in "
            f"{(ok['rep kept'] != ok['gpu kept']).sum()}, minimum moved > 0.01 "
            f"kcal/mol in {(ok['rep dEmin'].abs() > 0.01).sum()}"
        )
    if gpu_total:
        summary["time cpu / gpu"] = (
            f"{cpu_total:.2f} s / {gpu_total:.2f} s ({cpu_total / gpu_total:.1f}x)"
        )
    kv(summary)


def _median(s: pl.Series) -> float | None:
    v = [x for x in s.to_list() if x is not None and not math.isnan(x)]
    return round(statistics.median(v), 4) if v else None


def _mean(s: pl.Series) -> float:
    v = [x for x in s.to_list() if x is not None and not math.isnan(x)]
    return sum(v) / len(v) if v else math.nan


def _max(s: pl.Series) -> float | None:
    v = [x for x in s.to_list() if x is not None and not math.isnan(x)]
    return round(max(v), 6) if v else None


if __name__ == "__main__":
    main()
