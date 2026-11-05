"""Per-compound failure records (manifest entries and failure-table rows)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import AwareDatetime, Field

from modephere.types.base import FrozenModel
from modephere.types.enums import BuildStage, FailureCategory
from modephere.types.scalars import NonEmptyStr


class FailureRecord(FrozenModel):
    """Why one compound did not make it through one stage."""

    key: NonEmptyStr = Field(
        ...,
        description="Query key before standardization ('cid:2244'), InChIKey after.",
    )
    stage: BuildStage
    category: FailureCategory
    reason: NonEmptyStr
    occurred_at: AwareDatetime = Field(default_factory=lambda: datetime.now(tz=UTC))


__all__ = ["FailureRecord"]
