"""Topological pharmacophore graph (port of dev_Pgen ``pharmacophore_graph``).

Layout ``pharmacophore/v1``; must stay byte-identical to dev_Pgen
(``tests/unit/library/test_parity.py``). One node per feature, a complete
graph over the features, edges carrying the topological distance between the
closest atom pair of the two features.

    Node features (11):
        family one-hot[6 families, other]                                   (7)
        n_atoms in the feature / 6 (clamped)                                (1)
        aromatic fraction, ring fraction of member atoms                    (2)
        mean Gasteiger charge of member atoms                               (1)
    Edge features (13):
        topological-distance bin one-hot (11 bins + overflow)              (12)
        distance / 30 (clamped)                                             (1)
    Global features (258):
        6 family counts + 21 family-pair x 12 distance-bin counts, log1p,
        unbounded; all zeros for a featureless molecule. Not standardised:
        z-score with statistics fitted on training compounds only.
"""

from __future__ import annotations

import math
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem

from modephere.library.molecules import gasteiger_charge, one_hot
from modephere.types import (
    PHARMACOPHORE_GRAPH_V1,
    PharmacophoreFamily,
    PharmacophoreGraph,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from modephere.types import PharmacophoreFeature

_FAMILIES = list(PharmacophoreFamily)
_N_FAM = len(_FAMILIES)
#: Topological-distance bins (inclusive upper bounds); one extra overflow slot.
TOPO_UPPER: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17)
DIST_CAP = 30.0
N_TOPO_BINS = len(TOPO_UPPER) + 1
PAIR_INDEX = {
    pair: i for i, pair in enumerate(combinations_with_replacement(range(_N_FAM), 2))
}

_layout = PHARMACOPHORE_GRAPH_V1
if (
    _N_FAM + 1 + 4 != _layout.node_dim
    or N_TOPO_BINS + 1 != _layout.edge_dim
    or _N_FAM + len(PAIR_INDEX) * N_TOPO_BINS != _layout.global_dim
):  # pragma: no cover
    msg = "pharmacophore bins disagree with the pharmacophore/v1 layout"
    raise RuntimeError(msg)


def distance_bin(value: float, uppers: tuple[int, ...] = TOPO_UPPER) -> int:
    for i, upper in enumerate(uppers):
        if value <= upper:
            return i
    return len(uppers)


def _node_features(mol: Chem.Mol, feature: PharmacophoreFeature) -> list[float]:
    atoms = [mol.GetAtomWithIdx(i) for i in feature.atom_ids]
    n = len(atoms)
    feats = one_hot(feature.family, _FAMILIES, other=True)
    feats.append(min(n, 6) / 6.0)
    feats.append(sum(a.GetIsAromatic() for a in atoms) / n)
    feats.append(sum(a.IsInRing() for a in atoms) / n)
    feats.append(sum(gasteiger_charge(a) for a in atoms) / n)
    return feats


def pharmacophore_graph(
    microspecies: Chem.Mol, features: Sequence[PharmacophoreFeature]
) -> PharmacophoreGraph:
    """Graph over ``features`` (canonical order) of a microspecies.

    ``microspecies`` must carry Gasteiger charges (``protonate`` computes them).
    """
    k = len(features)
    num_atoms = microspecies.GetNumAtoms()
    if k == 0:
        return PharmacophoreGraph(
            x=np.empty((0, _layout.node_dim), dtype=np.float32),
            edge_index=np.empty((2, 0), dtype=np.int64),
            edge_attr=np.empty((0, _layout.edge_dim), dtype=np.float32),
            global_feats=np.zeros((1, _layout.global_dim), dtype=np.float32),
            atom2feat=np.empty((0, 2), dtype=np.int64),
            num_drug_atoms=num_atoms,
        )

    x = [_node_features(microspecies, f) for f in features]
    dist = Chem.GetDistanceMatrix(microspecies)  # 1e8 between fragments
    fam_counts = [0] * _N_FAM
    pair_hist = [0] * (len(PAIR_INDEX) * N_TOPO_BINS)
    for f in features:
        fam_counts[f.family.one_hot_index] += 1

    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            fi, fj = features[i], features[j]
            d = min(float(dist[np.ix_(fi.atom_ids, fj.atom_ids)].min()), DIST_CAP)
            dbin = distance_bin(d)
            bins = [0.0] * N_TOPO_BINS
            bins[dbin] = 1.0
            attr = [*bins, d / DIST_CAP]
            edge_index += [[i, j], [j, i]]
            edge_attr += [attr, attr]
            a, b = fi.family.one_hot_index, fj.family.one_hot_index
            pair_hist[PAIR_INDEX[(min(a, b), max(a, b))] * N_TOPO_BINS + dbin] += 1

    return PharmacophoreGraph(
        x=np.asarray(x, dtype=np.float32),
        edge_index=np.asarray(edge_index, dtype=np.int64).reshape(-1, 2).T,
        edge_attr=np.asarray(edge_attr, dtype=np.float32).reshape(-1, _layout.edge_dim),
        global_feats=np.asarray(
            [[math.log1p(c) for c in (*fam_counts, *pair_hist)]], dtype=np.float32
        ),
        atom2feat=np.asarray(
            [[a, n] for n, f in enumerate(features) for a in f.atom_ids], dtype=np.int64
        ),
        num_drug_atoms=num_atoms,
    )


__all__ = [
    "DIST_CAP",
    "N_TOPO_BINS",
    "PAIR_INDEX",
    "TOPO_UPPER",
    "distance_bin",
    "pharmacophore_graph",
]
