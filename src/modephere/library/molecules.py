"""Molecular graph featurizer: RDKit Mol -> :class:`~modephere.types.MoleculeGraph`.

Port of dev_Pgen ``src/data/library/drugs.py`` (layout ``molecule/v1``, must
stay byte-identical: ``tests/unit/library/test_parity.py``). Every categorical
field carries an explicit "other" bucket and every continuous field is
normalised, so out-of-vocabulary values are never silently all-zeros, except
the bounded numeric bins, whose overflow is counted (:class:`FeatureSaturation`).

    Node features (61):
        element one-hot[C,N,O,S,P,F,Cl,Br,I,B,Se,Si,other]   (13)
        degree one-hot[0..6]                                  (7)
        total-valence one-hot[0..6]                           (7)
        formal-charge scalar + one-hot[-1,0,+1,other]         (1+4)
        hybridization one-hot[SP,SP2,SP3,SP3D,SP3D2,S,other]  (7)
        total-Hs one-hot[0..4]                                (5)
        chirality one-hot[unspec,CW,CCW,other]                (4)
        is_aromatic, is_in_ring                               (2)
        ring-size membership[3,4,5,6,7]                       (5)
        Gasteiger partial charge (scalar, clamped)            (1)
        electronegativity (Pauling, normalised)               (1)
        is_H_donor, is_H_acceptor                             (2)
        mass (normalised), num_radical_electrons              (2)
    Edge features (18), both directions:
        bond-type one-hot[SINGLE,DOUBLE,TRIPLE,AROMATIC,other](5)
        is_conjugated, is_in_ring                             (2)
        ring-size membership[3,4,5,6,7]                       (5)
        stereo one-hot[NONE,Z,E,CIS,TRANS,other]              (6)
    Global features (1038): see :mod:`modephere.library.descriptors`.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, rdPartialCharges

from modephere.library.descriptors import molecular_descriptors
from modephere.types import MOLECULE_GRAPH_V1, MoleculeGraph

_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Se", "Si"]
_HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.S,
]
_CHIRAL_TAGS = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
_BOND_STEREO = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]
_DEGREE_BINS = [0, 1, 2, 3, 4, 5, 6]
_VALENCE_BINS = [0, 1, 2, 3, 4, 5, 6]
_CHARGE_BINS = [-1, 0, 1]
_HS_BINS = [0, 1, 2, 3, 4]
_RING_SIZES = [3, 4, 5, 6, 7]

# Pauling electronegativity by atomic number (normalised by max ~3.98 = F).
_PAULING_EN: dict[int, float] = {
    1: 2.20,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.31,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    34: 2.55,
    35: 2.96,
    53: 2.66,
}
_DONOR_ELEMENTS = {"N", "O", "S"}
_ACCEPTOR_ELEMENTS = {"N", "O"}

#: Numeric bins that can overflow to all-zeros (the categorical fields cannot).
SATURATING_FEATURES: tuple[str, ...] = ("degree", "valence", "total_hs")


class FeatureSaturation:
    """Tally of one-hot saturation events (out-of-range values encoded as zeros)."""

    def __init__(self) -> None:
        self.by_feature: Counter[str] = Counter()
        self.by_value: Counter[str] = Counter()

    def record(self, feature: str, value: Any) -> None:
        self.by_feature[feature] += 1
        self.by_value[f"{feature}={value}"] += 1

    @property
    def total(self) -> int:
        return sum(self.by_feature.values())


def one_hot(
    value: Any,
    choices: list[Any],
    *,
    other: bool = False,
    sink: FeatureSaturation | None = None,
    feature: str | None = None,
) -> list[float]:
    """One-hot encode ``value`` over ``choices`` (``other``: trailing catch-all)."""
    encoding = [0.0] * (len(choices) + (1 if other else 0))
    if value in choices:
        encoding[choices.index(value)] = 1.0
    elif other:
        encoding[-1] = 1.0
    elif sink is not None and feature is not None:
        sink.record(feature, value)
    return encoding


def gasteiger_charge(atom: rdchem.Atom) -> float:
    """Gasteiger partial charge, clamped to [-1, 1] (NaN/inf/absent -> 0)."""
    try:
        q = float(atom.GetDoubleProp("_GasteigerCharge"))
    except KeyError:
        return 0.0
    if math.isnan(q) or math.isinf(q):
        return 0.0
    return max(-1.0, min(1.0, q))


def _atom_features(
    atom: rdchem.Atom, saturation: FeatureSaturation | None
) -> list[float]:
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    feats: list[float] = []
    feats += one_hot(symbol, _ELEMENTS, other=True)
    feats += one_hot(atom.GetDegree(), _DEGREE_BINS, sink=saturation, feature="degree")
    feats += one_hot(
        atom.GetTotalValence(), _VALENCE_BINS, sink=saturation, feature="valence"
    )
    feats.append(max(-1.0, min(1.0, charge / 2.0)))
    feats += one_hot(charge, _CHARGE_BINS, other=True)
    feats += one_hot(atom.GetHybridization(), _HYBRIDIZATIONS, other=True)
    feats += one_hot(
        atom.GetTotalNumHs(), _HS_BINS, sink=saturation, feature="total_hs"
    )
    feats += one_hot(atom.GetChiralTag(), _CHIRAL_TAGS, other=True)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats += [1.0 if atom.IsInRingSize(n) else 0.0 for n in _RING_SIZES]
    feats.append(gasteiger_charge(atom))
    feats.append(_PAULING_EN.get(atom.GetAtomicNum(), 0.0) / 3.98)
    feats.append(1.0 if symbol in _DONOR_ELEMENTS and atom.GetTotalNumHs() > 0 else 0.0)
    feats.append(1.0 if symbol in _ACCEPTOR_ELEMENTS and charge <= 0 else 0.0)
    feats.append(atom.GetMass() / 120.0)
    feats.append(float(atom.GetNumRadicalElectrons()))
    return feats


def _bond_features(bond: rdchem.Bond) -> list[float]:
    feats: list[float] = []
    feats += one_hot(bond.GetBondType(), _BOND_TYPES, other=True)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    feats += [1.0 if bond.IsInRingSize(n) else 0.0 for n in _RING_SIZES]
    feats += one_hot(bond.GetStereo(), _BOND_STEREO, other=True)
    return feats


def molecule_graph(
    mol: Chem.Mol, *, saturation: FeatureSaturation | None = None
) -> MoleculeGraph:
    """Featurize ``mol`` (hydrogens implicit); node ``i`` is atom ``i``.

    Gasteiger charges are computed on a copy, so ``mol`` is not modified.
    """
    mol = Chem.Mol(mol)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    x = [_atom_features(atom, saturation) for atom in mol.GetAtoms()]

    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = _bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feats, feats]

    layout = MOLECULE_GRAPH_V1
    return MoleculeGraph(
        x=np.asarray(x, dtype=np.float32).reshape(-1, layout.node_dim),
        edge_index=np.asarray(edge_index, dtype=np.int64).reshape(-1, 2).T,
        edge_attr=np.asarray(edge_attr, dtype=np.float32).reshape(-1, layout.edge_dim),
        global_feats=molecular_descriptors(mol),
        atomic_numbers=np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()]),
    )


__all__ = [
    "SATURATING_FEATURES",
    "FeatureSaturation",
    "gasteiger_charge",
    "molecule_graph",
    "one_hot",
]
