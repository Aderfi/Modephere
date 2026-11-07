"""Compare the pharmacophores of two drugs of a built library, in 2D and 3D.

Usage (from the project root)::

    uv run python scripts/compare_pharmacophores.py clopidogrel prasugrel
    uv run python scripts/compare_pharmacophores.py CHEMBL1771 CHEMBL1201772
    uv run python scripts/compare_pharmacophores.py plavix effient     # ChEMBL synonyms
    uv run python scripts/compare_pharmacophores.py warfarin acenocoumarol --no-3d
    uv run python scripts/compare_pharmacophores.py aspirin ibuprofen --baseline-pairs 0
    uv run python scripts/compare_pharmacophores.py diazepam lorazepam --library data/library

Both drugs must already be in the library (``data/library`` unless
``--library``). A query is an InChIKey, a ChEMBL ID (the parent or a salt
record), a PubChem CID or a name: first the library's label / title, then the
preferred names and synonyms (INN, trade names...) of the staged ChEMBL
release, when one is staged.

Nothing is rebuilt. The features, the pharmacophore graph, the fingerprints
and the 3D feature points are read from the stored compound files, so the
report shows exactly what a model trained on the library sees. Only the atom
symbols (from the stored working SMILES) and the ECFP4 contrast fingerprint
are derived here.

The report:

1. **Each drug**: identity; features (family, member atoms, node features);
   feature-feature topological distances (bonds); and, with conformers, the
   Euclidean distances (Å) in the lowest-energy conformer, plus the feature
   pairs whose distance varies most across the ensemble.
2. **2D comparison**: family composition, the family-pair x distance profile,
   and a one-to-one feature matching (same family, greedy by node x context
   similarity), as in dev_Pgen's script.
3. **3D comparison**: the matched features of every conformer of A are
   superposed on those of every conformer of B (Kabsch, proper rotations
   only, so mirror images are not matched). The best overlay is the pair
   with the most features within ``--tolerance`` Å, ties broken by RMSD.
4. **Similarity summary**: every score next to its distribution over random
   library pairs (cached under ``data/cache/compare_pharmacophores/``), and a
   one-line verdict.

This is an inspection utility, not part of the library pipeline. It is a port
of dev_Pgen's ``scripts/compare_pharmacophores.py`` with a 3D part added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from _common import chembl_lookup, is_inchikey, kv, note, section, subsection
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from modephere.config import get_settings
from modephere.library.pharmacophore import fingerprints as _fp_module
from modephere.library.pharmacophore import graph2d as _graph_module
from modephere.library.pharmacophore.fingerprints import DIST3D_UPPER, tanimoto
from modephere.library.pharmacophore.graph2d import DIST_CAP, PAIR_INDEX
from modephere.library.standardize import working_mols
from modephere.library.storage import LibraryStore
from modephere.types import PharmacophoreFamily

if TYPE_CHECKING:
    from rdkit import Chem

    from modephere.chembl.lookup import ChemblLookup
    from modephere.types import CompoundArtifacts

_FAMILIES = list(PharmacophoreFamily)
_N_FAM = len(_FAMILIES)
_ABBR = {
    PharmacophoreFamily.DONOR: "Don",
    PharmacophoreFamily.ACCEPTOR: "Acc",
    PharmacophoreFamily.POS_IONIZABLE: "Pos",
    PharmacophoreFamily.NEG_IONIZABLE: "Neg",
    PharmacophoreFamily.AROMATIC: "Aro",
    PharmacophoreFamily.HYDROPHOBE: "Hyd",
}
# Coarse topological bins for the readable pair table and the pair Tanimoto
# only (the library's global_feats use the 12 bins of graph2d.TOPO_UPPER).
_PAIR_UPPER: tuple[int, ...] = (3, 6, 10)
_PAIR_BIN_LABELS = ("<=3", "4-6", "7-10", ">=11")
# 3D pair profile: the Å bins of the stored triplet fingerprint.
_N_BINS_3D = len(DIST3D_UPPER) + 1
_PAIR3D_WIDTH = len(PAIR_INDEX) * _N_BINS_3D
_UPPER_3D = np.asarray(DIST3D_UPPER)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_BASELINE_MAX_ATOMS = 70  # heavy atoms; larger compounds are left out of the baseline
_MIN_OVERLAY = 3  # matched features needed for a determined superposition
_FLEXIBLE_SHOWN = 5


# --------------------------------------------------------------------------- #
# Resolution: query -> library row
# --------------------------------------------------------------------------- #


@dataclass
class Resolved:
    row: dict[str, Any]
    matched_on: str
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def inchikey(self) -> str:
        return str(self.row["inchikey"])

    @property
    def name(self) -> str:
        return str(self.row["label"] or self.row["title"] or self.row["inchikey"])


def resolve(query: str, table: pl.DataFrame, chembl: ChemblLookup | None) -> Resolved:
    """Find ``query`` among the compounds of the library."""
    text = query.strip()
    upper = text.upper()
    if is_inchikey(upper):
        return _pick(table.filter(pl.col("inchikey") == upper), "InChIKey", text)
    if upper.startswith("CHEMBL") and upper[6:].isdigit():
        rows = table.filter(
            (pl.col("chembl_id") == upper) | (pl.col("chembl_record_id") == upper)
        )
        if rows.height == 0 and chembl is not None:
            hit = chembl.by_chembl_id(upper)  # a salt record the library never saw
            if hit is not None:
                rows = table.filter(pl.col("chembl_id") == hit.parent_chembl_id)
        return _pick(rows, "ChEMBL ID", text)
    if text.isdigit():
        return _pick(table.filter(pl.col("cid") == int(text)), "PubChem CID", text)

    needle = " ".join(text.casefold().split())
    for column in ("label", "title"):
        rows = table.filter(
            (pl.col(column).str.to_lowercase() == needle).fill_null(value=False)
        )
        if rows.height:
            return _pick(rows, f"library {column}", text)
    if chembl is not None:
        parents: list[str] = []
        kinds: dict[str, str] = {}
        for hit in chembl.by_name(text):
            parent = hit.parent_chembl_id
            if parent not in kinds:
                parents.append(parent)
                kinds[parent] = hit.matched
        rows = table.filter(pl.col("chembl_id").is_in(parents))
        if rows.height:
            # keep ChEMBL's ranking (preferred name, then max phase)
            rank = {p: i for i, p in enumerate(parents)}
            rows = rows.sort(
                pl.col("chembl_id").replace_strict(rank, default=len(rank))
            )
            first = str(rows["chembl_id"][0])
            return _pick(rows, f"ChEMBL {kinds[first].lower()}", text, keep_order=True)
    return _pick(table.clear(), "name", text)


def _pick(
    rows: pl.DataFrame, how: str, text: str, *, keep_order: bool = False
) -> Resolved:
    if rows.height == 0:
        msg = f"{text!r} is not in the library (looked up by {how})"
        raise LookupError(msg)
    if not keep_order:
        rows = rows.sort("inchikey")
    found = rows.to_dicts()
    return Resolved(found[0], matched_on=how, alternatives=found[1:])


def suggestions(text: str, table: pl.DataFrame, limit: int = 8) -> list[str]:
    """Library labels containing ``text`` (for the not-found message)."""
    needle = text.strip().casefold()
    if len(needle) < 3:  # noqa: PLR2004 -- shorter matches everything
        return []
    hits = table.filter(
        pl.col("label").str.to_lowercase().str.contains(needle, literal=True)
    )
    return hits["label"].drop_nulls().sort().head(limit).to_list()


# --------------------------------------------------------------------------- #
# Profile: one stored compound, decoded
# --------------------------------------------------------------------------- #


@dataclass
class Profile:
    """A library compound's pharmacophore, in arrays convenient to compare."""

    art: CompoundArtifacts
    families: list[PharmacophoreFamily]
    atoms: list[tuple[int, ...]]
    topo: np.ndarray  # [K, K] bonds between the closest member atoms (capped)
    points: np.ndarray | None  # [C, K, 3] Å, ensemble order; None for a 2D build
    relative_energy: np.ndarray | None  # [C] kcal/mol; None without a force field

    @property
    def k(self) -> int:
        return len(self.families)

    @property
    def num_conformers(self) -> int:
        return 0 if self.points is None else int(self.points.shape[0])

    @property
    def lowest(self) -> int:
        """Ensemble position of the lowest-energy conformer."""
        return 0 if self.art.conformers is None else self.art.conformers.lowest

    @cached_property
    def microspecies(self) -> Chem.Mol:
        """The pH 7.4 form, atoms in stored order (for atom symbols)."""
        return working_mols(self.art.compound)[1]

    @cached_property
    def morgan_fp(self) -> DataStructs.ExplicitBitVect:
        fp: DataStructs.ExplicitBitVect = _MORGAN.GetFingerprint(
            working_mols(self.art.compound)[0]
        )
        return fp

    @cached_property
    def pair3d(self) -> np.ndarray | None:
        """[C, 21 x 8] family-pair x Å-bin counts per conformer."""
        return None if self.points is None else pair3d_hist(self.points, self.families)

    def family_counts(self) -> Counter[PharmacophoreFamily]:
        return Counter(self.families)

    def pair_profile(self) -> Counter[tuple[str, str, str]]:
        """(family a, family b, coarse topological bin) -> count, unordered pairs."""
        prof: Counter[tuple[str, str, str]] = Counter()
        for i in range(self.k):
            for j in range(i + 1, self.k):
                fa, fb = sorted(
                    (self.families[i], self.families[j]), key=lambda f: f.one_hot_index
                )
                prof[fa.value, fb.value, _pair_bin(float(self.topo[i, j]))] += 1
        return prof

    def atom_label(self, idx: int) -> str:
        return f"{self.microspecies.GetAtomWithIdx(idx).GetSymbol()}{idx}"

    def feature_label(self, i: int) -> str:
        return f"{i}{_ABBR[self.families[i]]}"


def profile(art: CompoundArtifacts, *, use_3d: bool = True) -> Profile:
    feats = art.pharmacophore_2d.features
    k = len(feats)
    graph = art.pharmacophore
    topo = np.zeros((k, k))
    for e, (i, j) in enumerate(graph.edge_index.T.tolist()):
        topo[i, j] = round(float(graph.edge_attr[e, -1]) * DIST_CAP)

    points = energy = None
    ens = art.conformers
    if use_3d and ens is not None:
        points = np.zeros((ens.num_conformers, k, 3))
        for c, p3d in enumerate(art.pharmacophores_3d):
            for i, point in enumerate(p3d.points):
                points[c, i] = point.position
        energies = ens.energies
        if energies is not None:
            energy = energies - energies.min()
    return Profile(
        art=art,
        families=[f.family for f in feats],
        atoms=[f.atom_ids for f in feats],
        topo=topo,
        points=points,
        relative_energy=energy,
    )


def _pair_bin(d: float) -> str:
    for upper, label in zip(_PAIR_UPPER, _PAIR_BIN_LABELS, strict=False):
        if d <= upper:
            return label
    return _PAIR_BIN_LABELS[-1]


def pair3d_hist(points: np.ndarray, families: list[PharmacophoreFamily]) -> np.ndarray:
    """Per conformer, counts of (family pair, Å bin) over feature pairs.

    The Å bins are those of the stored 3D triplet fingerprint
    (``DIST3D_UPPER``), so this is the pair half of that fingerprint, as counts.
    """
    n_conf, k = points.shape[0], len(families)
    if k < 2:  # noqa: PLR2004 -- no pair
        return np.zeros((n_conf, _PAIR3D_WIDTH), dtype=np.int64)
    i, j = np.triu_indices(k, 1)
    fam = np.asarray([f.one_hot_index for f in families])
    lo, hi = np.minimum(fam[i], fam[j]), np.maximum(fam[i], fam[j])
    pair = np.asarray([PAIR_INDEX[int(a), int(b)] for a, b in zip(lo, hi, strict=True)])
    dist = np.linalg.norm(points[:, i] - points[:, j], axis=-1)  # [C, P]
    bins = np.searchsorted(_UPPER_3D, dist, side="left")
    flat = np.arange(n_conf)[:, None] * _PAIR3D_WIDTH + pair[None] * _N_BINS_3D + bins
    counts = np.bincount(flat.ravel(), minlength=n_conf * _PAIR3D_WIDTH)
    return counts.reshape(n_conf, _PAIR3D_WIDTH)


# --------------------------------------------------------------------------- #
# 2D similarity (as in dev_Pgen)
# --------------------------------------------------------------------------- #


def multiset_tanimoto(a: Counter[Any], b: Counter[Any]) -> float:
    """Sum of minima over sum of maxima (1.0 = identical counts)."""
    keys = set(a) | set(b)
    num = sum(min(a[k], b[k]) for k in keys)
    den = sum(max(a[k], b[k]) for k in keys)
    return num / den if den else 1.0


def global_cosine(a: Profile, b: Profile) -> float:
    """Cosine of the stored ``global_feats`` (0.0 if either is all zeros).

    Non-negative log counts: unrelated drug-like compounds already score well
    above 0, so read it against the random-pair percentile.
    """
    ga = a.art.pharmacophore.global_feats.ravel().astype(np.float64)
    gb = b.art.pharmacophore.global_feats.ravel().astype(np.float64)
    na, nb = np.linalg.norm(ga), np.linalg.norm(gb)
    return 0.0 if not na or not nb else float(ga @ gb / (na * nb))


@dataclass(frozen=True)
class Match:
    a: int
    b: int
    node: float  # similarity of the node features
    context: float  # similarity of the topological environment

    @property
    def score(self) -> float:
        """Node x context: high only when the feature *and* its environment agree
        (same-family features nearly always have node similarity ~1)."""
        return self.node * self.context


def node_similarity(a: Profile, i: int, b: Profile, j: int) -> float:
    """1 - mean |difference| of (size, aromatic, ring, charge), in [0, 1].

    Charge lives in [-1, 1], so its difference is halved.
    """
    xa = a.art.pharmacophore.x[i, _N_FAM + 1 :].astype(np.float64)
    xb = b.art.pharmacophore.x[j, _N_FAM + 1 :].astype(np.float64)
    diff = np.abs(xa - xb)
    diff[-1] /= 2.0
    return 1.0 - float(diff.mean())


def _context(p: Profile) -> list[Counter[tuple[PharmacophoreFamily, str]]]:
    """Per feature: (other family, coarse topological bin) counts."""
    ctx: list[Counter[tuple[PharmacophoreFamily, str]]] = [
        Counter() for _ in range(p.k)
    ]
    for i in range(p.k):
        for j in range(p.k):
            if i != j:
                ctx[i][p.families[j], _pair_bin(float(p.topo[i, j]))] += 1
    return ctx


def match_features(a: Profile, b: Profile) -> list[Match]:
    """Greedy one-to-one matching of same-family features, best score first."""
    if not a.k or not b.k:
        return []
    ctx_a, ctx_b = _context(a), _context(b)
    candidates = [
        Match(
            i,
            j,
            node=node_similarity(a, i, b, j),
            context=multiset_tanimoto(ctx_a[i], ctx_b[j]),
        )
        for i in range(a.k)
        for j in range(b.k)
        if a.families[i] == b.families[j]
    ]
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[Match] = []
    for m in sorted(candidates, key=lambda m: (-m.score, m.a, m.b)):
        if m.a in used_a or m.b in used_b:
            continue
        matches.append(m)
        used_a.add(m.a)
        used_b.add(m.b)
    return sorted(matches, key=lambda m: m.a)


# --------------------------------------------------------------------------- #
# 3D similarity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Overlay:
    """The best superposition of the matched features, over conformer pairs."""

    conf_a: int  # ensemble positions
    conf_b: int
    within: int  # matched features within the tolerance
    rmsd: float  # Å, over the matched features
    distances: np.ndarray  # [M] Å per matched feature, in match order
    pairs: int  # conformer pairs tried


def best_overlay(
    a: Profile, b: Profile, matches: list[Match], tolerance: float
) -> Overlay | None:
    """Superpose matched features for every conformer pair (batched Kabsch).

    The 2D matching fixes which feature goes onto which, so this measures
    whether the same features can also sit in the same place in space.
    """
    if a.points is None or b.points is None or not matches:
        return None
    p = a.points[:, [m.a for m in matches]]  # [Ca, M, 3]
    q = b.points[:, [m.b for m in matches]]  # [Cb, M, 3]
    p = p - p.mean(axis=1, keepdims=True)
    q = q - q.mean(axis=1, keepdims=True)
    h = np.einsum("amx,bmy->abxy", p, q)  # covariance per conformer pair
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    d[d == 0] = 1.0
    u[..., :, 2] *= d[..., None]  # no reflections
    rot = u @ vt  # row vectors: aligned = p @ rot
    aligned = np.einsum("amx,abxy->abmy", p, rot)
    dist = np.linalg.norm(aligned - q[None], axis=-1)  # [Ca, Cb, M]
    within = (dist <= tolerance).sum(axis=-1)
    rmsd = np.sqrt((dist**2).mean(axis=-1))
    best = int(np.lexsort((rmsd.ravel(), -within.ravel()))[0])
    ca, cb = divmod(best, q.shape[0])
    return Overlay(
        conf_a=ca,
        conf_b=cb,
        within=int(within[ca, cb]),
        rmsd=float(rmsd[ca, cb]),
        distances=dist[ca, cb],
        pairs=int(within.size),
    )


def pair3d_tanimoto(a: Profile, b: Profile) -> tuple[float, float] | None:
    """(best over conformer pairs, lowest-energy pair) multiset Tanimoto of the
    3D pair profiles."""
    ha, hb = a.pair3d, b.pair3d
    if ha is None or hb is None:
        return None
    num = np.minimum(ha[:, None], hb[None]).sum(axis=-1)
    den = np.maximum(ha[:, None], hb[None]).sum(axis=-1)
    t = np.where(den > 0, num / np.maximum(den, 1), 1.0)
    return float(t.max()), float(t[a.lowest, b.lowest])


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scores:
    family_tanimoto: float
    pair_tanimoto: float
    match_coverage: float
    match_similarity: float
    global_cosine: float
    gobbi_tanimoto: float
    morgan_tanimoto: float
    triplet3d_tanimoto: float | None = None
    pair3d_tanimoto: float | None = None
    overlay_coverage: float | None = None


_SCORE_LABELS: dict[str, str] = {
    "family_tanimoto": "family composition (multiset Tanimoto)",
    "pair_tanimoto": "pair x topological distance (multiset Tanimoto)",
    "match_coverage": "feature matching coverage",
    "match_similarity": "feature matching similarity",
    "global_cosine": "global_feats cosine (stored graph vector)",
    "gobbi_tanimoto": "Gobbi 2D pharmacophore FP Tanimoto (stored)",
    "morgan_tanimoto": "ECFP4 structural Tanimoto (contrast)",
    "triplet3d_tanimoto": "3D triplet FP Tanimoto (stored, ensembles)",
    "pair3d_tanimoto": "3D pair x Å profile, best conformer pair",
    "overlay_coverage": "3D overlay: matched features within tolerance",
}
_3D_SCORES = ("triplet3d_tanimoto", "pair3d_tanimoto", "overlay_coverage")


def compute_scores(
    a: Profile,
    b: Profile,
    matches: list[Match],
    overlay: Overlay | None,
) -> Scores:
    """Every score of a pair, each in [0, 1] with 1 = identical.

    Matching and overlay scores are normalised by the larger feature count, so
    features left unmatched in either drug pull them down. 3D scores are None
    unless both drugs have conformers.
    """
    longest = max(a.k, b.k)
    has_3d = a.points is not None and b.points is not None
    fa, fb = a.art.pharm3d_fingerprint, b.art.pharm3d_fingerprint
    pair3d = pair3d_tanimoto(a, b)
    return Scores(
        family_tanimoto=multiset_tanimoto(a.family_counts(), b.family_counts()),
        pair_tanimoto=multiset_tanimoto(a.pair_profile(), b.pair_profile()),
        match_coverage=len(matches) / longest if longest else 1.0,
        match_similarity=sum(m.score for m in matches) / longest if longest else 1.0,
        global_cosine=global_cosine(a, b),
        gobbi_tanimoto=tanimoto(a.art.pharm2d_fingerprint, b.art.pharm2d_fingerprint),
        morgan_tanimoto=DataStructs.TanimotoSimilarity(a.morgan_fp, b.morgan_fp),
        triplet3d_tanimoto=(
            tanimoto(fa, fb) if has_3d and fa is not None and fb is not None else None
        ),
        pair3d_tanimoto=None if pair3d is None else pair3d[0],
        overlay_coverage=(
            None
            if not has_3d
            else (overlay.within / longest if overlay is not None and longest else 0.0)
        ),
    )


def score_pair(a: Profile, b: Profile, tolerance: float) -> Scores:
    matches = match_features(a, b)
    return compute_scores(a, b, matches, best_overlay(a, b, matches, tolerance))


# --------------------------------------------------------------------------- #
# Random-pair baseline
# --------------------------------------------------------------------------- #


@dataclass
class Baseline:
    """Each score's distribution over random library pairs (sorted ascending)."""

    n_molecules: int
    n_pairs: int
    values: dict[str, list[float]]

    def percentile(self, metric: str, value: float) -> float | None:
        """Share of random pairs scoring <= ``value`` (None without data)."""
        dist = self.values.get(metric) or []
        return bisect_right(dist, value) / len(dist) if dist else None

    def quantile(self, metric: str, q: float) -> float | None:
        dist = self.values.get(metric) or []
        return dist[min(len(dist) - 1, int(q * len(dist)))] if dist else None


@dataclass(frozen=True)
class BaselineSpec:
    n_molecules: int
    n_pairs: int
    tolerance: float
    seed: int = 0


def random_pair_baseline(store: LibraryStore, spec: BaselineSpec) -> Baseline:
    """Score ``n_pairs`` random pairs drawn from ``n_molecules`` library compounds.

    Compounds without features or above ``_BASELINE_MAX_ATOMS`` heavy atoms
    (peptides, macrolides) are left out. 3D scores are always collected (for
    pairs where both compounds have conformers), so ``--no-3d`` shares the
    cache: its 2D distributions are the same sample.
    """
    rng = random.Random(spec.seed)
    keys = store.inchikeys()
    profiles = []
    for key in rng.sample(keys, min(spec.n_molecules, len(keys))):
        p = profile(store.load(key))
        if p.k and p.art.compound.num_heavy_atoms <= _BASELINE_MAX_ATOMS:
            profiles.append(p)
    if len(profiles) < 2:  # noqa: PLR2004
        msg = "not enough library compounds with features for a random-pair baseline"
        raise ValueError(msg)

    values: dict[str, list[float]] = {f.name: [] for f in fields(Scores)}
    for _ in range(spec.n_pairs):
        a, b = rng.sample(profiles, 2)
        scores = score_pair(a, b, spec.tolerance)
        for name, dist in values.items():
            value = getattr(scores, name)
            if value is not None:
                dist.append(value)
    return Baseline(
        n_molecules=len(profiles),
        n_pairs=spec.n_pairs,
        values={k: sorted(v) for k, v in values.items()},
    )


def cached_baseline(store: LibraryStore, spec: BaselineSpec) -> Baseline:
    """:func:`random_pair_baseline`, cached under ``data/cache/compare_pharmacophores``.

    The key covers the library registry, the sampling parameters and the
    source of this script and of the graph / fingerprint code, so a rebuilt
    library or a change in how scores are computed recomputes it.
    """
    key = hashlib.sha256()
    registry = store.table_path("compounds")
    stat = registry.stat()
    key.update(f"{registry.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    key.update(f"{spec}|{_BASELINE_MAX_ATOMS}".encode())
    for src in (
        Path(__file__),
        Path(_graph_module.__file__),
        Path(_fp_module.__file__),
    ):
        key.update(src.read_bytes())
    cache = (
        get_settings().paths.cache
        / "compare_pharmacophores"
        / f"baseline_{key.hexdigest()[:16]}.json"
    )
    if cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return Baseline(data["n_molecules"], data["n_pairs"], data["values"])

    print(
        f"Computing the random-pair baseline ({spec.n_pairs} pairs, cached in "
        f"{cache})...",
        file=sys.stderr,
    )
    baseline = random_pair_baseline(store, spec)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "n_molecules": baseline.n_molecules,
                "n_pairs": baseline.n_pairs,
                "values": baseline.values,
            }
        ),
        encoding="utf-8",
    )
    return baseline


# --------------------------------------------------------------------------- #
# Printing: one drug
# --------------------------------------------------------------------------- #


def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_identity(tag: str, r: Resolved, p: Profile) -> None:
    c = p.art.compound
    ens = p.art.conformers
    section(f"Drug {tag}: {r.name}")
    kv(
        {
            "matched on": r.matched_on,
            "InChIKey": c.inchikey,
            "ChEMBL parent": c.resolved.chembl_id,
            "ChEMBL record": c.resolved.chembl_record_id,
            "PubChem CID": c.resolved.cid,
            "working SMILES": c.working_smiles,
            "pH 7.4 microspecies": c.microspecies_smiles,
            "formal charge": f"{c.formal_charge:+d}",
            "heavy atoms": c.num_heavy_atoms,
            "features (nodes)": p.k,
            "edges (directed)": p.art.pharmacophore.num_edges,
            "atoms covered": len({a for ids in p.atoms for a in ids}),
            "conformers": (
                "none (2D-only build)"
                if ens is None
                else f"{ens.num_conformers} ({ens.force_field}, "
                f"{ens.params.backend} backend; lowest = #{p.lowest})"
            ),
        }
    )
    if r.alternatives:
        alts = ", ".join(
            f"{x['label']} ({x['chembl_id'] or x['inchikey']})"
            for x in r.alternatives[:6]
        )
        more = len(r.alternatives) - 6
        note(f"also matched: {alts}{f' (+{more} more)' if more > 0 else ''}")


def print_features(tag: str, p: Profile) -> None:
    subsection(f"{tag}: pharmacophore features")
    if not p.k:
        note("no pharmacophore feature perceived")
        return
    x = p.art.pharmacophore.x
    print(
        f"  {'#':>3}  {'family':<13} {'n':>2}  {'arom':>5} {'ring':>5} {'charge':>7}  atoms"
    )
    for i, (fam, ids) in enumerate(zip(p.families, p.atoms, strict=True)):
        atoms = " ".join(p.atom_label(a) for a in ids)
        print(
            f"  {i:>3}  {fam.value:<13} {len(ids):>2}  {x[i, _N_FAM + 2]:>5.2f} "
            f"{x[i, _N_FAM + 3]:>5.2f} {x[i, _N_FAM + 4]:>+7.3f}  {_short(atoms, 48)}"
        )


def _matrix(p: Profile, cell: Any) -> None:
    labels = [p.feature_label(i) for i in range(p.k)]
    w = max(len(s) for s in labels) + 2
    print("  " + " " * w + "".join(f"{s:>{w}}" for s in labels))
    for i, lab in enumerate(labels):
        cells = ["." if i == j else cell(i, j) for j in range(p.k)]
        print(f"  {lab:<{w}}" + "".join(f"{c:>{w}}" for c in cells))


def print_topology(tag: str, p: Profile, max_features: int) -> None:
    subsection(f"{tag}: topological distance between features (bonds)")
    if p.k < 2:  # noqa: PLR2004
        note("fewer than two features: no pairs")
        return
    if p.k > max_features:
        note(f"{p.k} features > --max-matrix {max_features}; skipped")
        return

    def cell(i: int, j: int) -> str:
        d = int(p.topo[i, j])
        return f">{int(DIST_CAP)}" if d >= DIST_CAP else str(d)

    _matrix(p, cell)


def print_geometry(tag: str, p: Profile, max_features: int) -> None:
    if p.points is None or p.k < 2:  # noqa: PLR2004
        return
    subsection(f"{tag}: Euclidean distance between features (Å), conformer #{p.lowest}")
    dist = np.linalg.norm(p.points[:, :, None] - p.points[:, None], axis=-1)  # [C,K,K]
    if p.k <= max_features:
        _matrix(p, lambda i, j: f"{dist[p.lowest, i, j]:.1f}")
    else:
        note(f"{p.k} features > --max-matrix {max_features}; matrix skipped")
    if p.num_conformers < 2:  # noqa: PLR2004
        return
    i, j = np.triu_indices(p.k, 1)
    lo, hi = dist[:, i, j].min(axis=0), dist[:, i, j].max(axis=0)
    order = np.argsort(lo - hi, kind="stable")[:_FLEXIBLE_SHOWN]
    print(f"\n  most variable pairs over the {p.num_conformers} conformers:")
    for n in order:
        pair = f"{p.feature_label(i[n])} - {p.feature_label(j[n])}"
        print(
            f"    {pair:<14} {lo[n]:>5.1f} - {hi[n]:>5.1f} Å  (range {hi[n] - lo[n]:.1f})"
        )


# --------------------------------------------------------------------------- #
# Printing: the comparison
# --------------------------------------------------------------------------- #


def print_family_comparison(a: Profile, b: Profile) -> None:
    section("2D: family composition")
    ca, cb = a.family_counts(), b.family_counts()
    print(f"  {'family':<13} {'A':>4} {'B':>4} {'B-A':>5}  {'shared':>6}")
    for fam in _FAMILIES:
        if ca[fam] or cb[fam]:
            print(
                f"  {fam.value:<13} {ca[fam]:>4} {cb[fam]:>4} {cb[fam] - ca[fam]:>+5}  "
                f"{min(ca[fam], cb[fam]):>6}"
            )
    print(f"  {'total':<13} {a.k:>4} {b.k:>4} {b.k - a.k:>+5}  {(ca & cb).total():>6}")

    def names(fams: set[PharmacophoreFamily]) -> str:
        return (
            ", ".join(f.value for f in sorted(fams, key=lambda f: f.one_hot_index))
            or "-"
        )

    print(f"\n  families shared : {names(set(ca) & set(cb))}")
    print(f"  only in A       : {names(set(ca) - set(cb))}")
    print(f"  only in B       : {names(set(cb) - set(ca))}")


def print_pair_profile(a: Profile, b: Profile) -> None:
    section("2D: family-pair x topological distance (unordered feature pairs)")
    pa, pb = a.pair_profile(), b.pair_profile()
    order = {f.value: f.one_hot_index for f in _FAMILIES}
    keys = sorted(
        set(pa) | set(pb),
        key=lambda k: (order[k[0]], order[k[1]], _PAIR_BIN_LABELS.index(k[2])),
    )
    if not keys:
        note("no feature pairs in either drug")
        return
    print(f"  {'pair':<30} {'bonds':>5} {'A':>4} {'B':>4}  note")
    for fa, fb, dbin in keys:
        na, nb = pa[fa, fb, dbin], pb[fa, fb, dbin]
        tag = "same" if na == nb else "only A" if not nb else "only B" if not na else ""
        print(f"  {fa + ' - ' + fb:<30} {dbin:>5} {na:>4} {nb:>4}  {tag}")
    same = sum(1 for k in keys if pa[k] == pb[k])
    print(
        f"\n  pair/bin cells: {len(keys)} | identical: {same} "
        f"| only A: {sum(1 for k in keys if not pb[k])} "
        f"| only B: {sum(1 for k in keys if not pa[k])}"
    )


def print_matching(a: Profile, b: Profile, matches: list[Match]) -> None:
    section("2D: feature matching (same family, greedy by node x context similarity)")
    if not matches:
        note("no feature can be matched")
    for m in matches:
        atoms_a = " ".join(a.atom_label(x) for x in a.atoms[m.a])
        atoms_b = " ".join(b.atom_label(x) for x in b.atoms[m.b])
        print(
            f"  A#{m.a:<3} {a.families[m.a].value:<13} [{_short(atoms_a, 22):<22}]  <->  "
            f"B#{m.b:<3} [{_short(atoms_b, 22):<22}]  "
            f"score={m.score:.2f} (node={m.node:.2f} ctx={m.context:.2f})"
        )
    matched_a, matched_b = {m.a for m in matches}, {m.b for m in matches}
    un_a = [f"#{i} {a.families[i].value}" for i in range(a.k) if i not in matched_a]
    un_b = [f"#{j} {b.families[j].value}" for j in range(b.k) if j not in matched_b]
    print(f"\n  unmatched in A: {', '.join(un_a) or '-'}")
    print(f"  unmatched in B: {', '.join(un_b) or '-'}")


def print_overlay(
    a: Profile,
    b: Profile,
    matches: list[Match],
    overlay: Overlay | None,
    tolerance: float,
) -> None:
    section("3D: overlay of the matched features (best conformer pair)")
    if a.points is None or b.points is None:
        missing = " and ".join(t for t, p in (("A", a), ("B", b)) if p.points is None)
        note(f"no conformers for {missing} (2D-only build or --no-3d)")
        return
    if overlay is None:
        note("no matched features to superpose")
        return

    def energy(p: Profile, c: int) -> str:
        e = p.relative_energy
        return "" if e is None else f", {e[c]:.2f} kcal/mol above the minimum"

    kv(
        {
            "conformer pairs tried": f"{overlay.pairs} ({a.num_conformers} x "
            f"{b.num_conformers})",
            "best pair": f"A #{overlay.conf_a}{energy(a, overlay.conf_a)}; "
            f"B #{overlay.conf_b}{energy(b, overlay.conf_b)}",
            "within tolerance": f"{overlay.within} of {len(matches)} matched features "
            f"(<= {tolerance:.2f} Å)",
            "RMSD": f"{overlay.rmsd:.2f} Å over the matched features",
        }
    )
    if len(matches) < _MIN_OVERLAY:
        note(
            f"only {len(matches)} matched feature(s): fewer than {_MIN_OVERLAY} points "
            "always superpose well, so this overlay says little"
        )
    print(f"\n  {'A':<10} {'family':<13} {'B':<8} {'distance':>9}")
    for m, d in zip(matches, overlay.distances, strict=True):
        flag = "" if d <= tolerance else "  off"
        print(f"  A#{m.a:<8} {a.families[m.a].value:<13} B#{m.b:<6} {d:>7.2f} Å{flag}")
    pair3d = pair3d_tanimoto(a, b)
    if pair3d is not None:
        print(
            f"\n  3D pair x Å profile Tanimoto: best conformer pair {pair3d[0]:.3f}, "
            f"lowest-energy pair {pair3d[1]:.3f}"
        )


# --------------------------------------------------------------------------- #
# Summary and verdict
# --------------------------------------------------------------------------- #

_HIGH_PCTL = 0.95  # above 95% of random pairs: clearly similar
_MODERATE_PCTL = 0.80  # above 80%: somewhat similar
# Fallback without a baseline (rough literature-style cut-offs).
_GOBBI_SIMILAR = 0.20
_MORGAN_SIMILAR = 0.35
_LEVEL_RANK = {"typical": 0, "moderate": 1, "high": 2}


def _level(baseline: Baseline | None, metric: str, value: float | None) -> str | None:
    """'high' / 'moderate' / 'typical' (of unrelated drugs); None if unknown."""
    if value is None:
        return None
    if baseline is None:
        cuts = {"gobbi_tanimoto": _GOBBI_SIMILAR, "morgan_tanimoto": _MORGAN_SIMILAR}
        if metric not in cuts:
            return None
        return "high" if value >= cuts[metric] else "typical"
    pct = baseline.percentile(metric, value)
    if pct is None:
        return None
    if pct >= _HIGH_PCTL:
        return "high"
    return "moderate" if pct >= _MODERATE_PCTL else "typical"


def verdict(s: Scores, baseline: Baseline | None) -> str:
    """One line from Gobbi 2D, the 3D scores (best of triplet FP and overlay) and
    ECFP4 (structure)."""
    pharm2d = _level(baseline, "gobbi_tanimoto", s.gobbi_tanimoto) or "typical"
    struct = _level(baseline, "morgan_tanimoto", s.morgan_tanimoto) or "typical"
    levels_3d = [
        lvl
        for lvl in (
            _level(baseline, "triplet3d_tanimoto", s.triplet3d_tanimoto),
            _level(baseline, "overlay_coverage", s.overlay_coverage),
        )
        if lvl is not None
    ]
    pharm3d = max(levels_3d, key=_LEVEL_RANK.__getitem__) if levels_3d else None

    parts = [f"2D {pharm2d}"] + ([f"3D {pharm3d}"] if pharm3d else [])
    pharm = ", ".join(parts)
    pharm_sim = pharm2d != "typical" or (pharm3d is not None and pharm3d != "typical")
    struct_sim = struct != "typical"
    if pharm_sim and struct_sim:
        line = f"Similar pharmacophorically ({pharm}) and structurally ({struct})."
    elif pharm_sim:
        line = (
            f"Pharmacophorically similar ({pharm}) but not structurally: "
            "possible scaffold hop."
        )
    elif struct_sim:
        line = (
            f"Structurally similar ({struct}) but not pharmacophorically ({pharm}): "
            "shared scaffold, different interaction pattern."
        )
    else:
        reference = (
            "the range of unrelated library pairs"
            if baseline
            else f"below Gobbi {_GOBBI_SIMILAR} / ECFP4 {_MORGAN_SIMILAR}"
        )
        line = f"NOT similar: pharmacophore and structure scores are in {reference}."
    if pharm3d in {"moderate", "high"} and pharm2d == "typical":
        line += (
            " The similarity is 3D only: the features can sit in the same places "
            "although the bond paths between them differ."
        )
    return line


def _fmt(value: float | None, width: int, spec: str = ".3f") -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}{spec}}"


def print_scores(s: Scores, baseline: Baseline | None) -> None:
    section("Similarity summary")
    head = f"  {'score':<50} {'value':>7}"
    if baseline is not None:
        head += f"  {'pctl':>5}  {'random p50':>10} {'p95':>6}"
    print(head)
    for f in fields(Scores):
        if f.name == _3D_SCORES[0]:
            print("  " + "-" * 48)
        value = getattr(s, f.name)
        line = f"  {_SCORE_LABELS[f.name]:<50} {_fmt(value, 7)}"
        if baseline is not None:
            pct = None if value is None else baseline.percentile(f.name, value)
            line += (
                f"  {_fmt(pct, 5, '.0%')}  {_fmt(baseline.quantile(f.name, 0.5), 10)} "
                f"{_fmt(baseline.quantile(f.name, 0.95), 6)}"
            )
        print(line)

    print()
    if baseline is not None:
        n3d = len(baseline.values.get(_3D_SCORES[0]) or [])
        print(
            f"  pctl = share of {baseline.n_pairs} random library pairs "
            f"({baseline.n_molecules} compounds; {n3d} pairs with 3D) scoring <= this\n"
            "  pair. Unrelated drugs already share the common families (donor, "
            "acceptor, aromatic),\n  so a raw value means something only next to the "
            "random-pair median."
        )
    else:
        note("no random-pair baseline: raw values are not calibrated")
    if s.overlay_coverage is None:
        note("3D scores need conformers for both drugs")
    print(f"\n  -> {verdict(s, baseline)}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Options:
    library: Path
    tolerance: float = 1.5
    use_3d: bool = True
    max_matrix: int = 24
    baseline_molecules: int = 400
    baseline_pairs: int = 3000
    seed: int = 0


def compare(query_a: str, query_b: str, opts: Options) -> Scores:
    """Resolve both drugs in the library, print the report, return the scores."""
    store = LibraryStore(opts.library)
    table = store.read_table("compounds")
    if table.height == 0:
        msg = f"no library at {opts.library} (build one with `python -m modephere.library`)"
        raise LookupError(msg)
    chembl = (
        chembl_lookup() if not all(_is_key(q) for q in (query_a, query_b)) else None
    )

    resolved = []
    for query in (query_a, query_b):
        try:
            resolved.append(resolve(query, table, chembl))
        except LookupError as e:
            close = suggestions(query, table)
            hint = f"; similar library labels: {', '.join(close)}" if close else ""
            raise LookupError(f"{e}{hint}") from None
    ra, rb = resolved
    pa = profile(store.load(ra.inchikey), use_3d=opts.use_3d)
    pb = profile(store.load(rb.inchikey), use_3d=opts.use_3d)

    section(f"PHARMACOPHORE COMPARISON   A = {ra.name}   vs   B = {rb.name}")
    kv({"library": f"{opts.library} ({table.height} compounds)"})
    if ra.inchikey == rb.inchikey:
        note("both queries are the same library compound")

    for tag, r, p in (("A", ra, pa), ("B", rb, pb)):
        print_identity(tag, r, p)
        print_features(tag, p)
        print_topology(tag, p, opts.max_matrix)
        print_geometry(tag, p, opts.max_matrix)

    matches = match_features(pa, pb)
    overlay = best_overlay(pa, pb, matches, opts.tolerance)
    print_family_comparison(pa, pb)
    print_pair_profile(pa, pb)
    print_matching(pa, pb, matches)
    print_overlay(pa, pb, matches, overlay, opts.tolerance)

    scores = compute_scores(pa, pb, matches, overlay)
    baseline = (
        cached_baseline(
            store,
            BaselineSpec(
                n_molecules=opts.baseline_molecules,
                n_pairs=opts.baseline_pairs,
                tolerance=opts.tolerance,
                seed=opts.seed,
            ),
        )
        if opts.baseline_pairs > 0
        else None
    )
    print_scores(scores, baseline)
    return scores


def _is_key(query: str) -> bool:
    """Queries resolved without the ChEMBL release (InChIKey, CID)."""
    text = query.strip()
    return is_inchikey(text.upper()) or text.isdigit()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the pharmacophores (2D and 3D) of two library drugs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("drug_a", help="InChIKey, ChEMBL ID, PubChem CID or name")
    parser.add_argument("drug_b", help="InChIKey, ChEMBL ID, PubChem CID or name")
    parser.add_argument(
        "--library",
        type=Path,
        default=get_settings().paths.library,
        help="Library root.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.5,
        help="Å within which an overlaid feature counts as matched in space.",
    )
    parser.add_argument(
        "--no-3d", action="store_true", help="2D only (ignore stored conformers)."
    )
    parser.add_argument(
        "--max-matrix",
        type=int,
        default=24,
        help="Skip the distance matrices beyond this many features.",
    )
    parser.add_argument(
        "--baseline-pairs",
        type=int,
        default=3000,
        help="Random library pairs used to calibrate the scores (0 = skip).",
    )
    parser.add_argument(
        "--baseline-molecules",
        type=int,
        default=400,
        help="Library compounds sampled for the random-pair baseline.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Baseline sampling seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    opts = Options(
        library=args.library,
        tolerance=args.tolerance,
        use_3d=not args.no_3d,
        max_matrix=args.max_matrix,
        baseline_molecules=args.baseline_molecules,
        baseline_pairs=args.baseline_pairs,
        seed=args.seed,
    )
    try:
        compare(args.drug_a, args.drug_b, opts)
    except (LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
