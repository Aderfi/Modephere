"""Conformer generation parameters and conformer ensembles (PLAN C2.2).

Defaults are the PLAN defaults (Q5/Q6): an ensemble of 10–50 conformers
scaled by rotatable bonds, ETKDGv3, MMFF94s with a UFF fallback, a
10 kcal/mol energy window, 0.5 Å heavy-atom RMSD pruning, stereoisomers
enumerated up to 8.

Coordinates are one ``float32`` array ``[C, N, 3]`` per ensemble, not one
object per conformer: that is the shape the storage layer writes and the PyG
adapter samples from (:class:`~modephere.types.tensors.ConformerTensors`).
"""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from modephere.types.arrays import Float32Array
from modephere.types.base import ArrayModel, FrozenModel
from modephere.types.enums import EmbeddingMethod, ForceField, StereoPolicy
from modephere.types.scalars import (
    Angstrom,
    InChIKey,
    KcalPerMol,
    PositiveKcalPerMol,
    Seed,
)

_CONFORMERS_PER_ROTOR = 5
_ENERGY_TOLERANCE = 1e-6


class ConformerParams(FrozenModel):
    """Everything that decides a conformer ensemble, validated up front."""

    method: EmbeddingMethod = EmbeddingMethod.ETKDG_V3
    num_conformers: PositiveInt | None = Field(
        default=None,
        description="Fixed number of embeddings; None scales it with rotatable "
        "bonds between min_conformers and max_conformers.",
    )
    min_conformers: PositiveInt = 10
    max_conformers: PositiveInt = 50
    prune_rms_threshold: Angstrom = Field(
        default=0.5, description="Heavy-atom RMSD under which conformers merge."
    )
    energy_window: PositiveKcalPerMol | None = Field(
        default=10.0, description="Keep conformers within this of the minimum."
    )
    force_field: ForceField = ForceField.MMFF94S
    fallback_force_field: ForceField | None = Field(
        default=ForceField.UFF,
        description="Used when the primary force field cannot parametrize.",
    )
    max_iterations: PositiveInt = 2000
    random_seed: Seed = 42
    num_threads: NonNegativeInt = Field(default=0, description="0 = all cores.")
    use_random_coords_fallback: bool = Field(
        default=True, description="Retry failed embeddings from random coordinates."
    )
    stereo_policy: StereoPolicy = StereoPolicy.ENUMERATE
    max_stereoisomers: PositiveInt = 8

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min_conformers > self.max_conformers:
            msg = (
                f"min_conformers ({self.min_conformers}) > "
                f"max_conformers ({self.max_conformers})"
            )
            raise ValueError(msg)
        if self.fallback_force_field is ForceField.NONE:
            msg = "fallback_force_field must be a force field or None, not 'none'"
            raise ValueError(msg)
        if self.fallback_force_field is self.force_field:
            msg = f"fallback_force_field repeats force_field ({self.force_field})"
            raise ValueError(msg)
        if self.force_field is ForceField.NONE and (
            self.energy_window is not None or self.fallback_force_field is not None
        ):
            msg = "without a force field there are no energies: set energy_window "
            msg += "and fallback_force_field to None"
            raise ValueError(msg)
        return self

    def target_count(self, num_rotatable_bonds: int) -> int:
        """Number of embeddings to attempt for one stereoisomer."""
        if self.num_conformers is not None:
            return self.num_conformers
        n = self.min_conformers + _CONFORMERS_PER_ROTOR * max(0, num_rotatable_bonds)
        return min(self.max_conformers, n)


class ConformerRecord(FrozenModel):
    """Metadata of one conformer; its coordinates live in the ensemble array."""

    conformer_id: NonNegativeInt
    stereoisomer: NonNegativeInt = Field(
        default=0, description="Index among the enumerated stereoisomers."
    )
    energy: KcalPerMol | None = Field(
        ..., description="Force-field energy; None when no force field ran."
    )
    converged: bool


class ConformerEnsemble(ArrayModel):
    """The conformers of one compound, with hydrogens.

    Atom order is that of the working-form molecule after ``AddHs``: heavy
    atoms first, so ``positions[:, :num_heavy_atoms]`` lines up with the 2D
    molecular graph.
    """

    inchikey: InChIKey
    num_atoms: PositiveInt = Field(..., description="Atoms including hydrogens.")
    num_heavy_atoms: PositiveInt
    force_field: ForceField = Field(
        ..., description="Force field actually used (primary or fallback)."
    )
    params: ConformerParams
    conformers: tuple[ConformerRecord, ...] = Field(..., min_length=1)
    positions: Float32Array = Field(..., description="[C, N, 3] in ångström.")

    @property
    def num_conformers(self) -> int:
        return len(self.conformers)

    @property
    def energies(self) -> np.ndarray | None:
        """Energies ``[C]`` (float64), or None without a force field."""
        if self.force_field is ForceField.NONE:
            return None
        return np.array([c.energy for c in self.conformers], dtype=np.float64)

    @property
    def lowest(self) -> int:
        """Position (not ``conformer_id``) of the lowest-energy conformer."""
        energies = self.energies
        return 0 if energies is None else int(np.argmin(energies))

    @model_validator(mode="after")
    def _check(self) -> Self:
        self._check_shape()
        self._check_force_field()
        ids = [c.conformer_id for c in self.conformers]
        if len(set(ids)) != len(ids):
            msg = f"duplicate conformer ids: {ids}"
            raise ValueError(msg)
        return self

    def _check_shape(self) -> None:
        expected = (self.num_conformers, self.num_atoms, 3)
        if self.positions.shape != expected:
            msg = f"positions shape {self.positions.shape} != {expected}"
            raise ValueError(msg)
        if self.num_heavy_atoms > self.num_atoms:
            msg = f"num_heavy_atoms ({self.num_heavy_atoms}) > num_atoms ({self.num_atoms})"
            raise ValueError(msg)

    def _check_force_field(self) -> None:
        allowed = {self.params.force_field, self.params.fallback_force_field}
        if self.force_field not in allowed:
            msg = f"force_field {self.force_field} is neither primary nor fallback"
            raise ValueError(msg)
        has_energy = [c.energy is not None for c in self.conformers]
        if self.force_field is ForceField.NONE:
            if any(has_energy):
                msg = "conformers carry energies but no force field ran"
                raise ValueError(msg)
            return
        if not all(has_energy):
            msg = f"every conformer needs an energy under {self.force_field}"
            raise ValueError(msg)
        window = self.params.energy_window
        energies = self.energies
        if window is not None and energies is not None:
            spread = float(energies.max() - energies.min())
            if spread > window + _ENERGY_TOLERANCE:
                msg = f"energy spread {spread:.3f} exceeds the {window} kcal/mol window"
                raise ValueError(msg)


__all__ = ["ConformerEnsemble", "ConformerParams", "ConformerRecord"]
