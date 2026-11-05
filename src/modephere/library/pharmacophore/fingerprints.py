"""Pharmacophore fingerprints, stored as sorted on-bit indices (PLAN C2.3b/d).

* **2D** — RDKit's Gobbi–Poppinga 2D pharmacophore fingerprint
  (``Gobbi_Pharm2D.factory``, 39 972 bits, layout ``gobbi_pharm2d/v1``),
  computed on the pH 7.4 microspecies. It uses Gobbi's own feature
  definitions: it is a standard reference fingerprint, comparable with other
  tools, not a view of Modephere's six families.
* **3D** — Modephere's feature-triplet fingerprint (``triplet_pharm3d/v1``,
  4 096 bits), built from the perceived features placed in every conformer.
  Every feature pair and triplet gives a key: its families plus its
  Euclidean distances binned in Å (``DIST3D_UPPER``), canonicalised over
  vertex permutations, hashed (BLAKE2b, so stable across runs and machines)
  to a bit. The ensemble fingerprint is the union over conformers. Distances
  only, so it is invariant to rotation and translation.
"""

from __future__ import annotations

import hashlib
from itertools import combinations, permutations
from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from modephere.types import GOBBI_PHARM2D_V1, TRIPLET_PHARM3D_V1

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from rdkit import Chem

    from modephere.types import Pharmacophore3D

#: Euclidean distance bins (inclusive upper bounds, Å); one extra overflow bin.
DIST3D_UPPER: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 17.0)

_GOBBI = Gobbi_Pharm2D.factory
if _GOBBI.GetSigSize() != GOBBI_PHARM2D_V1.num_bits:  # pragma: no cover
    msg = (
        f"Gobbi factory has {_GOBBI.GetSigSize()} bits, layout "
        f"{GOBBI_PHARM2D_V1.key} expects {GOBBI_PHARM2D_V1.num_bits}"
    )
    raise RuntimeError(msg)


def gobbi_pharm2d(mol: Chem.Mol) -> np.ndarray:
    """Sorted on-bits (int64) of the Gobbi 2D pharmacophore fingerprint."""
    fp = Generate.Gen2DFingerprint(mol, _GOBBI)
    return np.asarray(sorted(fp.GetOnBits()), dtype=np.int64)


def _bins(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    bins: np.ndarray = np.searchsorted(np.asarray(DIST3D_UPPER), dist, side="left")
    return bins


def _pair_key(fam: Sequence[int], bins: np.ndarray, i: int, j: int) -> tuple[int, ...]:
    a, b = sorted((fam[i], fam[j]))
    return (2, a, b, int(bins[i, j]))


def _triplet_key(
    fam: Sequence[int], bins: np.ndarray, i: int, j: int, k: int
) -> tuple[int, ...]:
    return min(
        (3, fam[a], fam[b], fam[c], int(bins[a, b]), int(bins[a, c]), int(bins[b, c]))
        for a, b, c in permutations((i, j, k))
    )


def _bit(key: tuple[int, ...]) -> int:
    digest = hashlib.blake2b(bytes(key), digest_size=8).digest()
    return int.from_bytes(digest, "little") % TRIPLET_PHARM3D_V1.num_bits


def _keys(p3d: Pharmacophore3D) -> Iterable[tuple[int, ...]]:
    fam = [p.family.one_hot_index for p in p3d.points]
    if len(fam) < 2:
        return
    bins = _bins(np.asarray([p.position for p in p3d.points], dtype=np.float64))
    for i, j in combinations(range(len(fam)), 2):
        yield _pair_key(fam, bins, i, j)
    for i, j, k in combinations(range(len(fam)), 3):
        yield _triplet_key(fam, bins, i, j, k)


def triplet_pharm3d(pharmacophores: Iterable[Pharmacophore3D]) -> np.ndarray:
    """Sorted on-bits (int64) of the ensemble's 3D triplet fingerprint."""
    keys: set[tuple[int, ...]] = set()
    for p3d in pharmacophores:
        keys.update(_keys(p3d))
    return np.asarray(sorted({_bit(k) for k in keys}), dtype=np.int64)


def to_dense(on_bits: np.ndarray, num_bits: int) -> np.ndarray:
    """Dense uint8 bit vector from on-bits (for similarity / ML input)."""
    dense = np.zeros(num_bits, dtype=np.uint8)
    dense[on_bits] = 1
    return dense


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity of two on-bit arrays (1.0 for two empty prints)."""
    union = np.union1d(a, b).size
    return 1.0 if union == 0 else np.intersect1d(a, b).size / union


__all__ = ["DIST3D_UPPER", "gobbi_pharm2d", "tanimoto", "to_dense", "triplet_pharm3d"]
