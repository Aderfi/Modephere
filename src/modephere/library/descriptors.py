# Descriptors.MolWt & co. are created at runtime; RDKit's stubs omit them.
# mypy: disable-error-code="attr-defined"
"""Per-molecule global descriptors (port of dev_Pgen ``molecular_descriptors``).

14 normalised QSAR physicochemical descriptors followed by a 1024-bit
Morgan/ECFP4 fingerprint: the ``global_feats`` [1, 1038] of ``molecule/v1``.
The fingerprint makes structurally similar molecules close in this vector.
NaN/inf descriptors (rare) are coerced to 0.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

from modephere.types import MOLECULE_GRAPH_V1

if TYPE_CHECKING:
    from rdkit import Chem

N_PHYSCHEM: int = 14
ECFP_RADIUS: int = 2
ECFP_BITS: int = 1024
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=ECFP_RADIUS, fpSize=ECFP_BITS
)

if N_PHYSCHEM + ECFP_BITS != MOLECULE_GRAPH_V1.global_dim:  # pragma: no cover
    msg = "descriptor width disagrees with the molecule/v1 layout"
    raise RuntimeError(msg)


def _num_stereo_centers(mol: Chem.Mol) -> int:
    try:
        return rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    except RuntimeError, ValueError:
        return 0


def physchem(mol: Chem.Mol) -> list[float]:
    """The 14 normalised physicochemical descriptors."""
    values = [
        Descriptors.MolWt(mol) / 500.0,
        Descriptors.MolLogP(mol) / 10.0,
        Descriptors.TPSA(mol) / 150.0,
        Descriptors.NumHDonors(mol) / 10.0,
        Descriptors.NumHAcceptors(mol) / 20.0,
        Descriptors.NumRotatableBonds(mol) / 20.0,
        Descriptors.FractionCSP3(mol),
        Descriptors.RingCount(mol) / 10.0,
        Descriptors.NumAromaticRings(mol) / 10.0,
        Descriptors.NumAliphaticRings(mol) / 10.0,
        Descriptors.NumHeteroatoms(mol) / 30.0,
        Descriptors.NumSaturatedRings(mol) / 10.0,
        Descriptors.qed(mol),
        _num_stereo_centers(mol) / 10.0,
    ]
    return [0.0 if math.isnan(v) or math.isinf(v) else v for v in values]


def molecular_descriptors(mol: Chem.Mol) -> np.ndarray:
    """``[1, 1038]`` float32: physchem + ECFP4 bits."""
    ecfp = [float(b) for b in _MORGAN.GetFingerprint(mol)]
    return np.asarray([physchem(mol) + ecfp], dtype=np.float32)


__all__ = [
    "ECFP_BITS",
    "ECFP_RADIUS",
    "N_PHYSCHEM",
    "molecular_descriptors",
    "physchem",
]
