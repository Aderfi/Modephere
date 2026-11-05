"""Annotated scalars, numpy array fields and the base-model contracts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from modephere.types import (
    ArrayModel,
    AtomIds,
    ChemblID,
    Float32Array,
    FrozenModel,
    InChIKey,
    Int64Array,
    PubChemCID,
    Seed,
    SmilesStr,
    UInt8Array,
    UnitVec3,
)
from modephere.types.base import values_equal


def _adapter(tp: Any) -> TypeAdapter[Any]:
    return TypeAdapter(tp)


class TestIdentifiers:
    def test_inchikey_normalised(self) -> None:
        key = " bsynrymutxbxsq-uhfffaoysa-n "
        assert _adapter(InChIKey).validate_python(key) == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    @pytest.mark.parametrize(
        "bad",
        [
            "BSYNRYMUTXBXSQ-UHFFFAOYSA",  # truncated
            "BSYNRYMUTXBXSQ-UHFFFAOYXA-N",  # flag must be S or N
            "BSYNRYMUTXBXSQ-UHFFFAOYSB-N",  # version must be A
            "BSYNRYMUTXBXS1-UHFFFAOYSA-N",  # digits are not allowed
        ],
    )
    def test_inchikey_rejects(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _adapter(InChIKey).validate_python(bad)

    def test_cid_bounds(self) -> None:
        cid = _adapter(PubChemCID)
        assert cid.validate_python("2244") == 2244
        for bad in (0, -1, 10**9 + 1):
            with pytest.raises(ValidationError):
                cid.validate_python(bad)

    def test_chembl_id(self) -> None:
        assert _adapter(ChemblID).validate_python("chembl25") == "CHEMBL25"
        with pytest.raises(ValidationError):
            _adapter(ChemblID).validate_python("CHEMBL")

    def test_smiles_strips_but_rejects_inner_whitespace(self) -> None:
        smiles = _adapter(SmilesStr)
        assert smiles.validate_python("  CCO\n") == "CCO"
        for bad in ("", "   ", "CCO ethanol"):
            with pytest.raises(ValidationError):
                smiles.validate_python(bad)

    def test_seed_is_deterministic_only(self) -> None:
        with pytest.raises(ValidationError):
            _adapter(Seed).validate_python(-1)  # RDKit's "random" seed


class TestGeometry:
    def test_atom_ids_sorted(self) -> None:
        assert _adapter(AtomIds).validate_python([5, 1, 3]) == (1, 3, 5)

    @pytest.mark.parametrize("bad", [[], [1, 1], [-1]])
    def test_atom_ids_rejects(self, bad: list[int]) -> None:
        with pytest.raises(ValidationError):
            _adapter(AtomIds).validate_python(bad)

    def test_unit_vector_normalised(self) -> None:
        v = _adapter(UnitVec3).validate_python((3.0, 0.0, 4.0))
        assert v == pytest.approx((0.6, 0.0, 0.8))
        assert math.isclose(sum(c * c for c in v), 1.0)

    @pytest.mark.parametrize("bad", [(0.0, 0.0, 0.0), (math.nan, 1.0, 0.0)])
    def test_unit_vector_rejects(self, bad: tuple[float, float, float]) -> None:
        with pytest.raises(ValidationError):
            _adapter(UnitVec3).validate_python(bad)


class _Arrays(ArrayModel):
    f: Float32Array
    i: Int64Array | None = None
    bits: UInt8Array | None = None


class TestArrays:
    def test_lists_cast_to_declared_dtype(self) -> None:
        m = _Arrays(f=[[1, 2], [3, 4]], i=[1, 2], bits=[0, 1, 1])
        assert m.f.dtype == np.float32
        assert m.i is not None and m.i.dtype == np.int64
        assert m.bits is not None and m.bits.dtype == np.uint8

    def test_stored_copy_is_read_only(self) -> None:
        src = np.zeros(3, dtype=np.float64)
        m = _Arrays(f=src)
        src[0] = 9.0
        assert m.f[0] == 0.0  # private copy
        with pytest.raises(ValueError, match="read-only"):
            m.f[0] = 1.0

    def test_float_never_truncates_into_int(self) -> None:
        with pytest.raises(ValidationError, match="cannot cast"):
            _Arrays(f=[0.0], i=[1.5])

    @pytest.mark.parametrize("bad", [[300], [-1]])
    def test_integer_range_checked(self, bad: list[int]) -> None:
        with pytest.raises(ValidationError, match="range"):
            _Arrays(f=[0.0], bits=bad)

    def test_non_finite_rejected(self) -> None:
        with pytest.raises(ValidationError, match="NaN"):
            _Arrays(f=[1.0, np.nan])

    def test_ragged_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _Arrays(f=[[1.0], [1.0, 2.0]])

    def test_equality_is_elementwise(self) -> None:
        assert _Arrays(f=[1.0, 2.0]) == _Arrays(f=np.array([1.0, 2.0]))
        assert _Arrays(f=[1.0, 2.0]) != _Arrays(f=[1.0, 3.0])

    def test_unhashable(self) -> None:
        with pytest.raises(TypeError):
            hash(_Arrays(f=[1.0]))

    def test_json_round_trip(self) -> None:
        m = _Arrays(f=[[1.0, 2.0]], i=[3])
        assert _Arrays.model_validate_json(m.model_dump_json()) == m

    def test_values_equal_checks_dtype(self) -> None:
        a = np.array([1, 2], dtype=np.int64)
        assert not values_equal(a, a.astype(np.int32))
        assert values_equal((a, 1), (a.copy(), 1))


class _Frozen(FrozenModel):
    name: str


class TestBaseModels:
    def test_unknown_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            _Frozen(name="x", typo=1)  # type: ignore[call-arg]

    def test_frozen_and_hashable(self) -> None:
        m = _Frozen(name=" x ")
        assert m.name == "x"
        assert hash(m) == hash(_Frozen(name="x"))
        with pytest.raises(ValidationError):
            m.name = "y"  # type: ignore[misc]
