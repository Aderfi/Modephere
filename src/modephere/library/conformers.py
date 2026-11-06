"""Conformer generation: microspecies -> :class:`~modephere.types.ConformerEnsemble`.

Pipeline (PLAN C2.2), every knob in :class:`~modephere.types.ConformerParams`:

1. stereoisomers: unassigned stereocentres are enumerated (``ENUMERATE``, up
   to ``max_stereoisomers``) or left to the embedder (``RANDOM``);
2. per stereoisomer, add hydrogens and embed ``target_count(rotatable bonds)``
   conformers with ETKDGv3 (fixed seed, heavy-atom RMSD pruning), retrying
   from random coordinates if none embed;
3. optimise with MMFF94s, or the fallback force field (UFF) when MMFF cannot
   parametrize the molecule (boron, for instance);
4. keep conformers within ``energy_window`` of the global minimum;
5. drop near-duplicates after optimisation: in energy order, a conformer is
   kept only if its symmetry-aware heavy-atom RMSD (``GetBestRMS``) to every
   kept conformer of the same stereoisomer exceeds ``prune_rms_threshold``.

Steps 2-3 run on one of two backends (``ConformerParams.backend``):

``rdkit`` (default)
    RDKit on the CPU, compound by compound (``num_threads`` threads);
    reproducible from ``random_seed``.
``nvmolkit``
    nvMolKit on a CUDA GPU: every stereoisomer of every compound given to
    :func:`generate_conformers_batch` is embedded in one ``EmbedMolecules``
    call per conformer count and optimised in one batched force field per
    force field, with the same ETKDG parameters, MMFF94s/UFF, iteration cap
    and gradient tolerance (1e-4) as RDKit. GPU runs with the same seed do not
    reproduce each other bit for bit. The library builder batches compounds
    for it (``LibraryBuildConfig.conformer_batch_size``).

nvMolKit falls back to RDKit when no CUDA GPU is usable, for molecules over
256 atoms with hydrogens (nvMolKit's coordinate initialisation limit) and for a
whole batch when an nvMolKit call fails. The backend that ran is the one in
``ensemble.params.backend``. Steps 1, 4 and 5 are the same code on both.

The ensemble is sorted by energy (conformer ``0`` is the lowest) and its atom
order is the microspecies' after ``AddHs``: heavy atoms first, so
``positions[:, :num_heavy_atoms]`` lines up with the 2D graph.
"""

from __future__ import annotations

import functools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolAlign, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

from modephere.core import CompoundError
from modephere.types import (
    BuildStage,
    ConformerBackend,
    ConformerEnsemble,
    ConformerParams,
    ConformerRecord,
    FailureCategory,
    ForceField,
    StereoPolicy,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nvmolkit.types import HardwareOptions

logger = logging.getLogger(__name__)

_MAX_RMS_MATCHES = 10_000  # cap on symmetry permutations per RMSD
_UNCHARGER = rdMolStandardize.Uncharger()

#: nvMolKit's eigenvalue coordinate initialisation (ETKDG's default, used
#: before any random-coordinates retry) takes at most this many atoms; a larger
#: molecule makes the whole ``EmbedMolecules`` call raise.
_NVMOLKIT_MAX_ATOMS = 256

#: Per stereoisomer: optimisation (converged, energy) for each embedded conformer.
_Results = list[list[tuple[bool, float | None]]]


@dataclass(frozen=True)
class ConformerSet:
    """An ensemble plus the RDKit molecule holding its coordinates.

    ``mol`` has explicit hydrogens and one conformer per ensemble entry, with
    conformer id ``i`` = ensemble position ``i``. Its stereo tags are those of
    the first stereoisomer; use the coordinates (or
    ``Chem.AssignStereochemistryFrom3D``) for the others.
    """

    ensemble: ConformerEnsemble
    mol: Chem.Mol


@dataclass
class _Candidate:
    stereoisomer: int
    mol: Chem.Mol  # heavy+H molecule of that stereoisomer
    conf_id: int
    energy: float | None
    converged: bool


@dataclass
class _Job:
    """One compound through steps 2-3, on one backend."""

    inchikey: str
    microspecies: Chem.Mol
    isomers: list[Chem.Mol]
    count: int  # embeddings attempted per stereoisomer
    backend: ConformerBackend
    mols: list[Chem.Mol] = field(default_factory=list)  # AddHs per stereoisomer
    embedded: list[tuple[int, Chem.Mol, list[int]]] = field(default_factory=list)
    force_field: ForceField | None = None
    results: _Results = field(default_factory=list)  # parallel to ``embedded``
    error: CompoundError | None = None

    def reset(self, backend: ConformerBackend) -> None:
        """Start steps 2-3 again, from fresh molecules."""
        self.backend = backend
        self.mols = [Chem.AddHs(iso) for iso in self.isomers]
        self.embedded = []
        self.force_field = None
        self.results = []
        self.error = None


def _fail(category: FailureCategory, reason: str) -> CompoundError:
    return CompoundError(BuildStage.CONFORMERS, category, reason)


# --------------------------------------------------------------------------- #
# Step 1: stereoisomers
# --------------------------------------------------------------------------- #


def _with_stereo_of(target: Chem.Mol, source: Chem.Mol) -> Chem.Mol:
    """Copy of ``target`` carrying ``source``'s atom and bond stereo (same atoms)."""
    out = Chem.RWMol(target)
    for atom, src in zip(out.GetAtoms(), source.GetAtoms(), strict=True):
        atom.SetChiralTag(src.GetChiralTag())
    for bond, src in zip(out.GetBonds(), source.GetBonds(), strict=True):
        if src.GetStereo() != Chem.BondStereo.STEREONONE:
            bond.SetStereoAtoms(*src.GetStereoAtoms())
            bond.SetStereo(src.GetStereo())
    mol = out.GetMol()
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def _stereoisomers(mol: Chem.Mol, params: ConformerParams) -> list[Chem.Mol]:
    """Stereoisomers of the microspecies, enumerated on its neutral form.

    A protonated amine (N+-H) is a stereocentre of the microspecies only:
    it inverts through deprotonation in water, so enumerating it would double
    every ensemble with interconverting "isomers". The neutral form has the
    same atoms in the same order, so its assigned stereo is copied back.
    """
    if params.stereo_policy is StereoPolicy.RANDOM:
        return [Chem.Mol(mol)]
    neutral = _UNCHARGER.uncharge(Chem.Mol(mol))
    if neutral.GetNumAtoms() != mol.GetNumAtoms():  # never expected; stay safe
        neutral = Chem.Mol(mol)
    options = StereoEnumerationOptions(
        onlyUnassigned=True,
        maxIsomers=params.max_stereoisomers,
        unique=True,
        tryEmbedding=False,
        rand=params.random_seed,
    )
    isomers = [
        _with_stereo_of(mol, iso)
        for iso in EnumerateStereoisomers(neutral, options=options)
    ]
    # Deterministic order whatever the enumerator yields.
    isomers.sort(key=Chem.MolToSmiles)
    return isomers or [Chem.Mol(mol)]


# --------------------------------------------------------------------------- #
# Steps 2-3 shared: ETKDG parameters, force-field choice
# --------------------------------------------------------------------------- #


def _etkdg(
    params: ConformerParams, *, random_coords: bool = False
) -> rdDistGeom.EmbedParameters:
    """ETKDG parameters of both backends (nvMolKit ignores ``numThreads``)."""
    ps = getattr(rdDistGeom, params.method.value)()
    ps.randomSeed = params.random_seed
    ps.numThreads = params.num_threads
    ps.pruneRmsThresh = params.prune_rms_threshold
    ps.onlyHeavyAtomsForRMS = True
    ps.useSymmetryForPruning = True
    ps.useRandomCoords = random_coords
    return cast("rdDistGeom.EmbedParameters", ps)


#: Elements whose only UFF type (Rappe et al. 1992) is sp3: Se3+2, Te3+2.
#: RDKit's UFF typer derives the type from the hybridization, so selenium in a
#: conjugated ring (ebselen, selenophenes) asks for "Se2+2", which UFF lacks,
#: and the molecule cannot be parametrized at all.
_UFF_SP3_ONLY = frozenset({34, 52})


def _uff_view(mol_h: Chem.Mol) -> Chem.Mol:
    """``mol_h``, or a copy typed for UFF: sp3-only elements retyped as sp3.

    Only the force field sees the copy; the bonded geometry still comes from the
    ring and its sp2 neighbours (ebselen: C-Se-N 86 degrees, planar ring).
    UFF bonds to Se in such rings come out ~0.07 A short (aromatic bond order).
    """
    retype = [
        a.GetIdx()
        for a in mol_h.GetAtoms()
        if a.GetAtomicNum() in _UFF_SP3_ONLY
        and a.GetHybridization() != Chem.HybridizationType.SP3
    ]
    if not retype:
        return mol_h
    view = Chem.Mol(mol_h)
    for idx in retype:
        view.GetAtomWithIdx(idx).SetHybridization(Chem.HybridizationType.SP3)
    return view


def _copy_coordinates(source: Chem.Mol, target: Chem.Mol) -> None:
    for conf in source.GetConformers():
        target.GetConformer(conf.GetId()).SetPositions(conf.GetPositions())


def _resolve_force_field(mol_h: Chem.Mol, params: ConformerParams) -> ForceField:
    order = [params.force_field]
    if params.fallback_force_field is not None:
        order.append(params.fallback_force_field)
    for ff in order:
        if ff is ForceField.NONE:
            return ff
        if ff in (ForceField.MMFF94, ForceField.MMFF94S):
            if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol_h):
                return ff
        elif rdForceFieldHelpers.UFFHasAllMoleculeParams(_uff_view(mol_h)):
            return ff
    raise _fail(
        FailureCategory.FORCEFIELD_FAILED,
        f"no parameters in {[str(f) for f in order]}",
    )


def _collect_embedded(job: _Job) -> None:
    """``job.embedded`` from the conformers now on ``job.mols``; one force field."""
    for s, mol_h in enumerate(job.mols):
        if mol_h.GetNumConformers():
            ids = [c.GetId() for c in mol_h.GetConformers()]
            job.embedded.append((s, mol_h, ids))
        else:
            logger.debug("%s: stereoisomer %d did not embed", job.inchikey, s)
    if not job.embedded:
        job.error = _fail(
            FailureCategory.EMBEDDING_FAILED,
            f"no conformer embedded ({len(job.isomers)} stereoisomer(s), "
            f"{job.count} attempts each)",
        )


def _choose_force_field(job: _Job, params: ConformerParams) -> None:
    for _, mol_h, _ in job.embedded:
        try:
            ff = _resolve_force_field(mol_h, params)
        except CompoundError as e:
            job.error = e
            return
        if job.force_field is None:
            job.force_field = ff
        elif ff is not job.force_field:  # same constitution: never expected
            msg = f"stereoisomers need different force fields ({job.force_field}, {ff})"
            job.error = _fail(FailureCategory.FORCEFIELD_FAILED, msg)
            return


# --------------------------------------------------------------------------- #
# Steps 2-3 on RDKit (CPU)
# --------------------------------------------------------------------------- #


def _embed(mol_h: Chem.Mol, params: ConformerParams, count: int) -> list[int]:
    ids = list(
        rdDistGeom.EmbedMultipleConfs(mol_h, numConfs=count, params=_etkdg(params))
    )
    if not ids and params.use_random_coords_fallback:
        ps = _etkdg(params, random_coords=True)
        ids = list(rdDistGeom.EmbedMultipleConfs(mol_h, numConfs=count, params=ps))
    return ids


def _optimise(
    mol_h: Chem.Mol, ff: ForceField, params: ConformerParams
) -> list[tuple[bool, float | None]]:
    if ff is ForceField.NONE:
        return [(True, None)] * mol_h.GetNumConformers()
    if ff is ForceField.UFF:
        view = _uff_view(mol_h)
        results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
            view, numThreads=params.num_threads, maxIters=params.max_iterations
        )
        if view is not mol_h:  # optimised on the retyped copy: copy coordinates back
            _copy_coordinates(view, mol_h)
    else:
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            mol_h,
            numThreads=params.num_threads,
            maxIters=params.max_iterations,
            mmffVariant=ff.value,
        )
    return [(not_converged == 0, float(energy)) for not_converged, energy in results]


def _run_rdkit(job: _Job, params: ConformerParams) -> None:
    for mol_h in job.mols:
        _embed(mol_h, params, job.count)
    _collect_embedded(job)
    if job.error is None:
        _choose_force_field(job, params)
    if job.error is None:
        assert job.force_field is not None
        job.results = [
            _optimise(mol_h, job.force_field, params) for _, mol_h, _ in job.embedded
        ]


# --------------------------------------------------------------------------- #
# Steps 2-3 on nvMolKit (GPU)
# --------------------------------------------------------------------------- #


@functools.cache
def nvmolkit_available() -> bool:
    """nvMolKit imports and sees a CUDA GPU (checked once, warned once)."""
    try:
        import nvmolkit.embedMolecules  # noqa: F401, PLC0415 -- loads torch + CUDA
        import torch  # noqa: PLC0415
    except ImportError as e:
        logger.warning("nvMolKit unavailable (%s): conformers run on RDKit", e)
        return False
    if not torch.cuda.is_available():
        logger.warning("no CUDA GPU for nvMolKit: conformers run on RDKit")
        return False
    return True


def _hardware(params: ConformerParams) -> HardwareOptions:
    from nvmolkit.types import HardwareOptions  # noqa: PLC0415

    return HardwareOptions(preprocessingThreads=params.num_threads or -1)


def _nv_embed(
    jobs: list[_Job], params: ConformerParams, hardware: HardwareOptions
) -> None:
    """Embed every stereoisomer of every job in place, one call per count."""
    from nvmolkit.embedMolecules import EmbedMolecules  # noqa: PLC0415

    def embed(todo: list[tuple[Chem.Mol, int]], *, random_coords: bool) -> None:
        by_count: dict[int, list[Chem.Mol]] = defaultdict(list)
        for mol_h, count in todo:
            by_count[count].append(mol_h)
        ps = _etkdg(params, random_coords=random_coords)
        for count, mols in sorted(by_count.items()):
            EmbedMolecules(mols, ps, confsPerMolecule=count, hardwareOptions=hardware)

    todo = [(mol_h, job.count) for job in jobs for mol_h in job.mols]
    embed(todo, random_coords=False)
    if params.use_random_coords_fallback:
        retry = [(m, c) for m, c in todo if m.GetNumConformers() == 0]
        if retry:
            embed(retry, random_coords=True)


def _nv_optimise(
    mols: list[Chem.Mol],
    ff: ForceField,
    params: ConformerParams,
    hardware: HardwareOptions,
) -> _Results:
    """Minimise every conformer of ``mols`` in place, in one batch."""
    from nvmolkit.batchedForcefield import (  # noqa: PLC0415
        MMFFBatchedForcefield,
        UFFBatchedForcefield,
    )

    if ff is ForceField.NONE:
        return [[(True, None)] * m.GetNumConformers() for m in mols]
    if ff is ForceField.UFF:
        views = [_uff_view(m) for m in mols]
        engine = UFFBatchedForcefield(views, hardwareOptions=hardware)
    else:
        views = mols
        props = [
            rdForceFieldHelpers.MMFFGetMoleculeProperties(m, mmffVariant=ff.value)
            for m in mols
        ]
        engine = MMFFBatchedForcefield(mols, properties=props, hardwareOptions=hardware)
    energies, converged = engine.minimize(maxIters=params.max_iterations)
    for mol_h, view in zip(mols, views, strict=True):
        if view is not mol_h:  # optimised on the retyped copy: copy coordinates back
            _copy_coordinates(view, mol_h)
    return [
        [(bool(ok), float(e)) for ok, e in zip(oks, es, strict=True)]
        for oks, es in zip(converged, energies, strict=True)
    ]


def _run_nvmolkit(jobs: list[_Job], params: ConformerParams) -> None:
    hardware = _hardware(params)
    _nv_embed(jobs, params, hardware)
    for job in jobs:
        _collect_embedded(job)
        if job.error is None:
            _choose_force_field(job, params)
    ready = [job for job in jobs if job.error is None]
    for ff in {job.force_field for job in ready}:
        assert ff is not None
        group = [job for job in ready if job.force_field is ff]
        mols = [mol_h for job in group for _, mol_h, _ in job.embedded]
        results = iter(_nv_optimise(mols, ff, params, hardware))
        for job in group:
            job.results = [next(results) for _ in job.embedded]


# --------------------------------------------------------------------------- #
# Steps 4-5 and the ensemble
# --------------------------------------------------------------------------- #


def _prune(candidates: list[_Candidate], threshold: float) -> list[_Candidate]:
    """Greedy, energy-ordered RMSD pruning within each stereoisomer."""
    kept: list[_Candidate] = []
    heavy: dict[int, Chem.Mol] = {}
    for cand in candidates:
        if cand.stereoisomer not in heavy:
            heavy[cand.stereoisomer] = Chem.RemoveHs(cand.mol)
        ref_mol = heavy[cand.stereoisomer]
        duplicate = any(
            other.stereoisomer == cand.stereoisomer
            and rdMolAlign.GetBestRMS(
                ref_mol,
                ref_mol,
                prbId=cand.conf_id,
                refId=other.conf_id,
                maxMatches=_MAX_RMS_MATCHES,
            )
            <= threshold
            for other in kept
        )
        if not duplicate:
            kept.append(cand)
    return kept


def _assemble(job: _Job, params: ConformerParams) -> ConformerSet | CompoundError:
    if job.error is not None:
        return job.error
    force_field = job.force_field
    assert force_field is not None
    candidates = [
        _Candidate(s, mol_h, conf_id, energy, converged)
        for (s, mol_h, ids), results in zip(job.embedded, job.results, strict=True)
        for conf_id, (converged, energy) in zip(ids, results, strict=True)
    ]
    if force_field is not ForceField.NONE:
        candidates.sort(key=lambda c: (c.energy, c.stereoisomer, c.conf_id))
        if params.energy_window is not None:
            floor = candidates[0].energy or 0.0
            candidates = [
                c
                for c in candidates
                if (c.energy or 0.0) - floor <= params.energy_window
            ]
    candidates = _prune(candidates, params.prune_rms_threshold)

    out = Chem.Mol(candidates[0].mol)
    out.RemoveAllConformers()
    positions = []
    for i, cand in enumerate(candidates):
        conf = Chem.Conformer(cand.mol.GetConformer(cand.conf_id))
        conf.SetId(i)
        out.AddConformer(conf, assignId=False)
        positions.append(conf.GetPositions())

    if job.backend is not params.backend:
        params = params.model_copy(update={"backend": job.backend})
    ensemble = ConformerEnsemble(
        inchikey=job.inchikey,
        num_atoms=out.GetNumAtoms(),
        num_heavy_atoms=job.microspecies.GetNumAtoms(),
        force_field=force_field,
        params=params,
        conformers=tuple(
            ConformerRecord(
                conformer_id=i,
                stereoisomer=c.stereoisomer,
                energy=c.energy,
                converged=c.converged,
            )
            for i, c in enumerate(candidates)
        ),
        positions=np.asarray(positions, dtype=np.float32),
    )
    return ConformerSet(ensemble=ensemble, mol=out)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _job(microspecies: Chem.Mol, inchikey: str, params: ConformerParams) -> _Job:
    isomers = _stereoisomers(microspecies, params)
    count = params.target_count(rdMolDescriptors.CalcNumRotatableBonds(microspecies))
    job = _Job(inchikey, microspecies, isomers, count, params.backend)
    job.reset(params.backend)
    if job.backend is ConformerBackend.NVMOLKIT:
        if not nvmolkit_available():
            job.backend = ConformerBackend.RDKIT
        elif job.mols[0].GetNumAtoms() > _NVMOLKIT_MAX_ATOMS:
            logger.debug(
                "%s: %d atoms > %d, conformers on RDKit",
                inchikey,
                job.mols[0].GetNumAtoms(),
                _NVMOLKIT_MAX_ATOMS,
            )
            job.backend = ConformerBackend.RDKIT
    return job


def generate_conformers_batch(
    compounds: Sequence[tuple[Chem.Mol, str]], params: ConformerParams | None = None
) -> list[ConformerSet | CompoundError]:
    """Ensembles for many ``(microspecies, inchikey)`` at once, in input order.

    A compound that fails comes back as its ``CompoundError`` (stage
    conformers); the others are unaffected. With the nvMolKit backend the GPU
    sees the whole batch, which is where its speed comes from.
    """
    params = params or ConformerParams()
    jobs = [_job(mol, key, params) for mol, key in compounds]
    on_gpu = [job for job in jobs if job.backend is ConformerBackend.NVMOLKIT]
    if on_gpu:
        try:
            _run_nvmolkit(on_gpu, params)
        except (RuntimeError, ValueError) as e:  # nvMolKit raised for the batch
            logger.warning(
                "nvMolKit failed (%s: %s): %d compound(s) rerun on RDKit",
                type(e).__name__,
                e,
                len(on_gpu),
            )
            for job in on_gpu:
                job.reset(ConformerBackend.RDKIT)
    for job in jobs:
        if job.backend is ConformerBackend.RDKIT:
            _run_rdkit(job, params)
    return [_assemble(job, params) for job in jobs]


def generate_conformers(
    microspecies: Chem.Mol, inchikey: str, params: ConformerParams | None = None
) -> ConformerSet:
    """Embed, optimise and prune; raises ``CompoundError`` (stage conformers)."""
    (result,) = generate_conformers_batch([(microspecies, inchikey)], params)
    if isinstance(result, CompoundError):
        raise result
    return result


__all__ = [
    "ConformerSet",
    "generate_conformers",
    "generate_conformers_batch",
    "nvmolkit_available",
]
