"""Place perceived features in each conformer (PLAN C2.3c).

For every conformer, each feature becomes a :class:`PharmacophorePoint` at the
centroid of its member atoms, plus a direction where the chemistry defines one:

* **Donor** (single atom): mean unit vector from the atom to its hydrogens;
* **Acceptor** (single atom): from the mean of its neighbours to the atom (the
  lone-pair bisector, e.g. along C=O for a carbonyl O);
* **Aromatic**: the ring normal (least-variance axis of the ring atoms), with
  its sign fixed by the right-hand rule over the ring's first atoms in index
  order. The sign depends on atom labels only, so it rotates with the molecule.

Directions are equivariant (they rotate with the conformer); positions are
equivariant too, so distances, and the fingerprints built from them, are
invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from modephere.types import (
    Pharmacophore3D,
    PharmacophoreFamily,
    PharmacophorePoint,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rdkit import Chem

    from modephere.types import PharmacophoreFeature

_EPS = 1e-6


def _unit(v: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(v))
    return None if norm < _EPS else v / norm


def _donor_direction(mol: Chem.Mol, idx: int, xyz: np.ndarray) -> np.ndarray | None:
    hs = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(idx).GetNeighbors()
        if n.GetAtomicNum() == 1
    ]
    if not hs:
        return None
    vectors = [u for h in hs if (u := _unit(xyz[h] - xyz[idx])) is not None]
    return _unit(np.mean(vectors, axis=0)) if vectors else None


def _acceptor_direction(mol: Chem.Mol, idx: int, xyz: np.ndarray) -> np.ndarray | None:
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
    if not nbrs:
        return None
    return _unit(xyz[idx] - xyz[nbrs].mean(axis=0))


def _ring_normal(atom_ids: tuple[int, ...], xyz: np.ndarray) -> np.ndarray | None:
    ring = xyz[list(atom_ids)]
    centre = ring.mean(axis=0)
    _, _, vt = np.linalg.svd(ring - centre)
    normal = vt[-1]
    # Orient by the first non-collinear pair of ring atoms (label-defined).
    for j in range(1, len(atom_ids)):
        ref = np.cross(ring[0] - centre, ring[j] - centre)
        if np.linalg.norm(ref) > _EPS:
            return _unit(normal if float(normal @ ref) >= 0 else -normal)
    return _unit(normal)


def _direction(
    mol: Chem.Mol, feature: PharmacophoreFeature, xyz: np.ndarray
) -> np.ndarray | None:
    family, ids = feature.family, feature.atom_ids
    if family is PharmacophoreFamily.AROMATIC and len(ids) >= 3:
        return _ring_normal(ids, xyz)
    if len(ids) != 1:
        return None
    if family is PharmacophoreFamily.DONOR:
        return _donor_direction(mol, ids[0], xyz)
    if family is PharmacophoreFamily.ACCEPTOR:
        return _acceptor_direction(mol, ids[0], xyz)
    return None


def _vec(v: np.ndarray) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def place_features(
    mol_h: Chem.Mol,
    features: Sequence[PharmacophoreFeature],
    *,
    inchikey: str,
    num_heavy_atoms: int,
) -> tuple[Pharmacophore3D, ...]:
    """One :class:`Pharmacophore3D` per conformer of ``mol_h`` (explicit Hs)."""
    out = []
    for conf in mol_h.GetConformers():
        xyz = conf.GetPositions()
        points = []
        for f in features:
            direction = _direction(mol_h, f, xyz)
            points.append(
                PharmacophorePoint(
                    family=f.family,
                    atom_ids=f.atom_ids,
                    position=_vec(xyz[list(f.atom_ids)].mean(axis=0)),
                    direction=None if direction is None else _vec(direction),
                )
            )
        out.append(
            Pharmacophore3D(
                inchikey=inchikey,
                conformer_id=conf.GetId(),
                num_heavy_atoms=num_heavy_atoms,
                points=tuple(points),
            )
        )
    return tuple(out)


__all__ = ["place_features"]
