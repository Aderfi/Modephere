"""Per-element provenance of the pharmacogenomic build.

:class:`~modephere.kg.provenance.Stamp` is per write batch; a pgx element
also records, for derived facts, the ``method`` and the raw ``evidence`` it
was derived from. :class:`ElementStamp` carries all of it per row; the writer
(:mod:`modephere.pgx.write`) groups rows by ``(sources, method)`` into batch
stamps and stores ``evidence`` as an ordinary property.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pathlib import Path

_CHUNK = 1 << 20


class ElementStamp(BaseModel):
    """Provenance carried by one node or relationship."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[str, ...] = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    method: str | None = None
    evidence: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ElementStamp", "sha256_file"]
