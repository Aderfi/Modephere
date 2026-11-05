"""Annotated scalar types: identifiers, chemistry units and geometry.

Every alias is a PEP 695 ``type`` over ``Annotated[...]`` so the constraint
travels with the type wherever it is used (fields, ``TypeAdapter``, nested
tuples). They validate shape and range only: whether a SMILES actually parses
is the standardization layer's job (RDKit), not this one's.
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    FiniteFloat,
    StringConstraints,
)

# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #

#: PubChem Compound ID (same bound as dev_Pgen's ``PubChemCID``).
type PubChemCID = Annotated[int, Field(gt=0, le=10**9)]


def _strip_upper(value: object) -> object:
    # Runs before the pattern: StringConstraints(to_upper=...) applies the
    # pattern to the raw input, which would reject lowercase spellings.
    return value.strip().upper() if isinstance(value, str) else value


#: ChEMBL molecule ID, e.g. ``CHEMBL25``; upper-cased on input.
type ChemblID = Annotated[
    str, BeforeValidator(_strip_upper), StringConstraints(pattern=r"^CHEMBL\d+$")
]

#: Standard InChIKey: 14-char skeleton hash, 8-char stereo/isotope hash +
#: standard flag (S, or N for non-standard) + version (A), protonation char.
type InChIKey = Annotated[
    str,
    BeforeValidator(_strip_upper),
    StringConstraints(pattern=r"^[A-Z]{14}-[A-Z]{8}[SN]A-[A-Z]$"),
]

type NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


def _check_smiles(value: str) -> str:
    smiles = value.strip()
    if not smiles:
        msg = "SMILES cannot be empty"
        raise ValueError(msg)
    if any(c.isspace() for c in smiles):
        # A space inside a SMILES line starts its title ("CCO ethanol"): the
        # caller passed a SMILES-file line, not a SMILES.
        msg = f"SMILES must not contain whitespace: {smiles!r}"
        raise ValueError(msg)
    return smiles


#: A SMILES string, syntactically unchecked (RDKit parses it downstream).
type SmilesStr = Annotated[str, AfterValidator(_check_smiles)]

# --------------------------------------------------------------------------- #
# Atoms
# --------------------------------------------------------------------------- #

#: 0-based atom index into an RDKit molecule (heavy atoms come first: ``AddHs``
#: appends hydrogens after them, so heavy-atom indices survive protonation and
#: hydrogen addition unchanged).
type AtomIndex = Annotated[int, Field(ge=0)]


def _sorted_unique(ids: tuple[int, ...]) -> tuple[int, ...]:
    if len(set(ids)) != len(ids):
        msg = f"duplicate atom indices: {ids}"
        raise ValueError(msg)
    return tuple(sorted(ids))


#: The atoms of one feature: non-empty, unique, stored sorted (canonical form).
type AtomIds = Annotated[
    tuple[AtomIndex, ...], Field(min_length=1), AfterValidator(_sorted_unique)
]

# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #

#: A non-negative length in ångström (RMSD thresholds, distances).
type Angstrom = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]

#: An energy in kcal/mol (force-field energies can be negative).
type KcalPerMol = Annotated[float, Field(allow_inf_nan=False)]

#: A strictly positive energy in kcal/mol (energy windows).
type PositiveKcalPerMol = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]

#: Deterministic RDKit seed. RDKit reads ``-1`` as "random"; that is not allowed
#: here — every build is reproducible.
type Seed = Annotated[int, Field(ge=0, le=2**31 - 1)]

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

#: A point in Cartesian space (ångström).
type Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]

_MIN_NORM = 1e-6
_UNIT_TOLERANCE = 1e-12


def _normalise(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if norm < _MIN_NORM:
        msg = f"direction vector has (near-)zero length: {v}"
        raise ValueError(msg)
    if abs(norm - 1.0) < _UNIT_TOLERANCE:
        return v  # already unit: idempotent, so a stored vector reloads bit-exact
    return (v[0] / norm, v[1] / norm, v[2] / norm)


#: A direction; any non-zero vector is accepted and stored normalised.
type UnitVec3 = Annotated[Vec3, AfterValidator(_normalise)]


__all__ = [
    "Angstrom",
    "AtomIds",
    "AtomIndex",
    "ChemblID",
    "InChIKey",
    "KcalPerMol",
    "NonEmptyStr",
    "PositiveKcalPerMol",
    "PubChemCID",
    "Seed",
    "SmilesStr",
    "UnitVec3",
    "Vec3",
]
