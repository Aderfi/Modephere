"""Numpy array field types for Pydantic models.

The chemistry layers hand numpy arrays between them (PLAN C3: they never
import torch). These aliases let an :class:`~modephere.types.base.ArrayModel`
declare an array field with a fixed dtype:

* input may be an array or a nested sequence;
* it is cast to the declared dtype only under numpy's ``same_kind`` rule, so a
  float never truncates silently into an integer field;
* float arrays must be finite;
* the stored array is a private, read-only copy, so a frozen model stays
  frozen (mutating the caller's array cannot change it, and vice versa).

Shapes are checked by each model's own validator, where the expected
dimensions (atoms, conformers) are known.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import PlainSerializer, PlainValidator, WithJsonSchema


def _coerce(dtype: type[np.generic]) -> Callable[[Any], np.ndarray]:
    target = np.dtype(dtype)

    def validate(value: Any) -> np.ndarray:
        src = np.asarray(value)
        if src.dtype == object:
            msg = "array input is ragged or contains non-numeric objects"
            raise ValueError(msg)
        if target.kind in "iu" and src.dtype.kind in "iu":
            # Integer to integer (either signedness): the values decide, since a
            # plain cast would wrap (300 -> 44 in uint8).
            info = np.iinfo(target)
            if src.size and (src.min() < info.min or src.max() > info.max):
                msg = f"values outside the {target} range [{info.min}, {info.max}]"
                raise ValueError(msg)
        elif not np.can_cast(src.dtype, target, casting="same_kind"):
            msg = f"cannot cast {src.dtype} to {target} without loss of kind"
            raise ValueError(msg)
        arr = np.array(src, dtype=target, copy=True)
        if target.kind == "f" and not np.isfinite(arr).all():
            msg = "array contains NaN or infinite values"
            raise ValueError(msg)
        arr.setflags(write=False)
        return arr

    return validate


def _json_schema(item: str) -> WithJsonSchema:
    return WithJsonSchema({"type": "array", "description": f"nested {item} array"})


_TO_LIST = PlainSerializer(lambda a: a.tolist(), when_used="json")

type Float32Array = Annotated[
    NDArray[np.float32],
    PlainValidator(_coerce(np.float32)),
    _TO_LIST,
    _json_schema("float32"),
]
type Float64Array = Annotated[
    NDArray[np.float64],
    PlainValidator(_coerce(np.float64)),
    _TO_LIST,
    _json_schema("float64"),
]
type Int64Array = Annotated[
    NDArray[np.int64],
    PlainValidator(_coerce(np.int64)),
    _TO_LIST,
    _json_schema("int64"),
]
#: Bit vectors (fingerprints).
type UInt8Array = Annotated[
    NDArray[np.uint8],
    PlainValidator(_coerce(np.uint8)),
    _TO_LIST,
    _json_schema("uint8"),
]

__all__ = ["Float32Array", "Float64Array", "Int64Array", "UInt8Array"]
