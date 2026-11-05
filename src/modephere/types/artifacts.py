"""Everything the library stores for one compound, checked for consistency.

:class:`CompoundArtifacts` is the unit the builder produces and the storage
layer writes (one ``compounds/<InChIKey>.safetensors`` file). Its validator
holds the cross-object contracts that no single schema can see:

* every part carries the same InChIKey;
* the molecule graph has one node per heavy atom of the working form, and the
  pharmacophore graph indexes into exactly that many atoms;
* pharmacophore node ``i`` is 2D feature ``i``, member for member;
* there is one 3D pharmacophore per conformer, in ensemble order, each with
  exactly the 2D features.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field, model_validator

from modephere.types.arrays import Int64Array
from modephere.types.base import ArrayModel
from modephere.types.compound import StandardizedCompound
from modephere.types.conformer import ConformerEnsemble
from modephere.types.graph import MoleculeGraph, PharmacophoreGraph
from modephere.types.layouts import (
    GOBBI_PHARM2D_V1,
    TRIPLET_PHARM3D_V1,
    FingerprintLayout,
)
from modephere.types.pharmacophore import Pharmacophore2D, Pharmacophore3D


def _check_on_bits(bits: np.ndarray, layout: FingerprintLayout) -> None:
    if bits.ndim != 1:
        msg = f"{layout.key} on-bits must be 1D, got shape {bits.shape}"
        raise ValueError(msg)
    if bits.size and (bits.min() < 0 or bits.max() >= layout.num_bits):
        msg = f"{layout.key} on-bits fall outside [0, {layout.num_bits})"
        raise ValueError(msg)
    if bits.size > 1 and not (np.diff(bits) > 0).all():
        msg = f"{layout.key} on-bits must be strictly increasing"
        raise ValueError(msg)


class CompoundArtifacts(ArrayModel):
    """All representations of one compound."""

    compound: StandardizedCompound
    molecule: MoleculeGraph
    pharmacophore: PharmacophoreGraph
    pharmacophore_2d: Pharmacophore2D
    pharm2d_fingerprint: Int64Array = Field(
        ..., description="Sorted on-bits of the Gobbi 2D pharmacophore fingerprint."
    )
    conformers: ConformerEnsemble | None = Field(
        default=None, description="None for a 2D-only build."
    )
    pharmacophores_3d: tuple[Pharmacophore3D, ...] = ()
    pharm3d_fingerprint: Int64Array | None = Field(
        default=None, description="Sorted on-bits of the 3D triplet fingerprint."
    )

    @property
    def inchikey(self) -> str:
        return self.compound.inchikey

    @property
    def has_3d(self) -> bool:
        return self.conformers is not None

    @model_validator(mode="after")
    def _check(self) -> Self:
        self._check_keys()
        self._check_2d()
        self._check_3d()
        return self

    def _check_keys(self) -> None:
        keys = {self.pharmacophore_2d.inchikey}
        keys |= {p.inchikey for p in self.pharmacophores_3d}
        if self.conformers is not None:
            keys.add(self.conformers.inchikey)
        if keys - {self.inchikey}:
            msg = f"artifact parts disagree on the InChIKey: {sorted(keys)}"
            raise ValueError(msg)

    def _check_2d(self) -> None:
        n = self.compound.num_heavy_atoms
        if self.molecule.num_nodes != n:
            msg = f"molecule graph has {self.molecule.num_nodes} nodes, compound {n} heavy atoms"
            raise ValueError(msg)
        if self.pharmacophore.num_drug_atoms != n:
            msg = f"pharmacophore graph indexes {self.pharmacophore.num_drug_atoms} atoms, not {n}"
            raise ValueError(msg)
        if self.pharmacophore_2d.num_heavy_atoms != n:
            msg = "pharmacophore_2d atom count differs from the compound's"
            raise ValueError(msg)
        members = {
            (atom, i)
            for i, f in enumerate(self.pharmacophore_2d.features)
            for atom in f.atom_ids
        }
        graph_members = {(int(a), int(f)) for a, f in self.pharmacophore.atom2feat}
        if members != graph_members:
            msg = "pharmacophore graph nodes do not match the 2D features"
            raise ValueError(msg)
        _check_on_bits(self.pharm2d_fingerprint, GOBBI_PHARM2D_V1)

    def _check_3d(self) -> None:
        ensemble = self.conformers
        if ensemble is None:
            if self.pharmacophores_3d or self.pharm3d_fingerprint is not None:
                msg = "3D pharmacophores need a conformer ensemble"
                raise ValueError(msg)
            return
        if ensemble.num_heavy_atoms != self.compound.num_heavy_atoms:
            msg = "conformer ensemble heavy-atom count differs from the compound's"
            raise ValueError(msg)
        ids = [c.conformer_id for c in ensemble.conformers]
        if [p.conformer_id for p in self.pharmacophores_3d] != ids:
            msg = "need one 3D pharmacophore per conformer, in ensemble order"
            raise ValueError(msg)
        features = self.pharmacophore_2d.features
        if any(p.features != features for p in self.pharmacophores_3d):
            msg = "a 3D pharmacophore has features the 2D pharmacophore lacks"
            raise ValueError(msg)
        if self.pharm3d_fingerprint is None:
            msg = "a 3D build needs pharm3d_fingerprint"
            raise ValueError(msg)
        _check_on_bits(self.pharm3d_fingerprint, TRIPLET_PHARM3D_V1)


__all__ = ["CompoundArtifacts"]
