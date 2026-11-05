"""Base models shared by every Modephere schema.

Three flavours, picked by what the model holds:

* :class:`ModephereModel` — mutable records (revalidated on every assignment).
* :class:`FrozenModel` — immutable, hashable value objects (queries, params,
  features). Most schemas use this one.
* :class:`ArrayModel` — immutable models holding numpy arrays or torch
  tensors. Pydantic's default ``__eq__`` compares the field dicts with ``==``,
  which is ambiguous for arrays, so equality is element-wise here; arrays are
  unhashable, so these models are too.

All three reject unknown fields: a typo in a config or a catalog column is a
validation error, never a silently ignored key.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict


class ModephereModel(BaseModel):
    """Mutable base: strips strings and revalidates on assignment."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


class FrozenModel(BaseModel):
    """Immutable, hashable value object."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=True,
        extra="forbid",
    )


class ArrayModel(FrozenModel):
    """Immutable model whose fields may be numpy arrays or torch tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(
            values_equal(getattr(self, name), getattr(other, name))
            for name in type(self).model_fields
        )


def _is_tensor(value: object) -> bool:
    # Duck-typed so this module never imports torch.
    return type(value).__module__.startswith("torch") and hasattr(value, "dtype")


def values_equal(a: Any, b: Any) -> bool:
    """Equality that is exact (dtype + shape + values) for arrays and tensors."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    if _is_tensor(a) or _is_tensor(b):
        import torch

        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and bool(torch.equal(a, b))
        )
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        return len(a) == len(b) and all(
            values_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    return bool(a == b)


__all__ = ["ArrayModel", "FrozenModel", "ModephereModel", "values_equal"]
