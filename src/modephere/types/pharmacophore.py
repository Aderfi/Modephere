"""Pharmacophore features, 2D and per-conformer 3D (PLAN C2.3).

Perception runs once per compound, on the pH 7.4 microspecies of the working
form, and yields :class:`PharmacophoreFeature` s (a family + its atoms).
Feature *identity* is ``(family, atom_ids)``. The 2D pharmacophore is the
feature set. Each conformer's 3D pharmacophore places the same features in
space (:class:`PharmacophorePoint`), so ``p3d.features == p2d.features`` for
every conformer of a compound.

Atom indices are heavy-atom indices of the working form, the same as the 2D
molecular graph nodes (and dev_Pgen's ``atom2feat``).

Features are stored in canonical order: by first atom, then family one-hot
index, then atom tuple, the same order as dev_Pgen ``_perceive``. Node ``i``
of the pharmacophore graph is therefore feature ``i`` whatever order
perception produced them in.
"""

from __future__ import annotations

from collections import Counter
from typing import Self

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from modephere.types.base import FrozenModel
from modephere.types.enums import PharmacophoreFamily
from modephere.types.scalars import AtomIds, InChIKey, UnitVec3, Vec3


class PharmacophoreFeature(FrozenModel):
    """One pharmacophore feature: a family over a group of atoms."""

    family: PharmacophoreFamily
    atom_ids: AtomIds

    @property
    def identity(self) -> tuple[PharmacophoreFamily, tuple[int, ...]]:
        return (self.family, self.atom_ids)

    def as_feature(self) -> PharmacophoreFeature:
        """This feature without geometry (identity on a plain feature)."""
        return PharmacophoreFeature(family=self.family, atom_ids=self.atom_ids)


class PharmacophorePoint(PharmacophoreFeature):
    """A feature placed in one conformer."""

    position: Vec3 = Field(..., description="Centroid of the member atoms (Å).")
    direction: UnitVec3 | None = Field(
        default=None,
        description="H-bond direction (donor/acceptor) or ring normal (aromatic).",
    )

    @model_validator(mode="after")
    def _check_direction(self) -> Self:
        if self.direction is not None and not self.family.is_directional:
            msg = f"{self.family} features are isotropic and carry no direction"
            raise ValueError(msg)
        return self


def _canonical_order[F: PharmacophoreFeature](features: tuple[F, ...]) -> tuple[F, ...]:
    """Sort into canonical order; reject a feature perceived twice."""
    seen = Counter(f.identity for f in features)
    dupes = [ident for ident, n in seen.items() if n > 1]
    if dupes:
        msg = f"duplicate pharmacophore features: {dupes}"
        raise ValueError(msg)
    return tuple(
        sorted(
            features, key=lambda f: (f.atom_ids[0], f.family.one_hot_index, f.atom_ids)
        )
    )


def _check_atom_range(features: tuple[PharmacophoreFeature, ...], n: int) -> None:
    for f in features:
        if f.atom_ids[-1] >= n:
            msg = f"{f.family} feature {f.atom_ids} indexes past {n} heavy atoms"
            raise ValueError(msg)


def _family_counts(
    features: tuple[PharmacophoreFeature, ...],
) -> dict[PharmacophoreFamily, int]:
    counts = Counter(f.family for f in features)
    return {family: counts[family] for family in PharmacophoreFamily}


class Pharmacophore2D(FrozenModel):
    """The feature set of a compound (topology only).

    An empty feature set is a valid record; the builder reports it as
    ``FailureCategory.NO_PHARMACOPHORE_FEATURES``.
    """

    inchikey: InChIKey
    num_heavy_atoms: PositiveInt
    features: tuple[PharmacophoreFeature, ...] = ()

    @field_validator("features")
    @classmethod
    def _order(
        cls, v: tuple[PharmacophoreFeature, ...]
    ) -> tuple[PharmacophoreFeature, ...]:
        return _canonical_order(tuple(f.as_feature() for f in v))

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_atom_range(self.features, self.num_heavy_atoms)
        return self

    @property
    def family_counts(self) -> dict[PharmacophoreFamily, int]:
        """Count per family, every family present (zeros included)."""
        return _family_counts(self.features)


class Pharmacophore3D(FrozenModel):
    """The features of a compound placed in one conformer."""

    inchikey: InChIKey
    conformer_id: NonNegativeInt
    num_heavy_atoms: PositiveInt
    points: tuple[PharmacophorePoint, ...] = ()

    @field_validator("points")
    @classmethod
    def _order(
        cls, v: tuple[PharmacophorePoint, ...]
    ) -> tuple[PharmacophorePoint, ...]:
        return _canonical_order(v)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_atom_range(self.points, self.num_heavy_atoms)
        return self

    @property
    def features(self) -> tuple[PharmacophoreFeature, ...]:
        """The points without geometry, comparable to ``Pharmacophore2D.features``."""
        return tuple(p.as_feature() for p in self.points)

    @property
    def family_counts(self) -> dict[PharmacophoreFamily, int]:
        return _family_counts(self.points)

    def to_2d(self) -> Pharmacophore2D:
        return Pharmacophore2D(
            inchikey=self.inchikey,
            num_heavy_atoms=self.num_heavy_atoms,
            features=self.features,
        )


__all__ = [
    "Pharmacophore2D",
    "Pharmacophore3D",
    "PharmacophoreFeature",
    "PharmacophorePoint",
]
