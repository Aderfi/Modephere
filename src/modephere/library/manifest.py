"""Build manifest: what has been built, with which options and toolkit versions.

JSON, written atomically (temp file then rename) after every flush, so an
interrupted build leaves a consistent file and resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_serializer

logger = logging.getLogger(__name__)


class ManifestStats(BaseModel):
    queries: int = 0
    built: int = 0
    skipped: int = 0
    duplicates: int = 0
    failed: int = 0
    failures_by_category: dict[str, int] = Field(default_factory=dict)


class BuildManifest(BaseModel):
    """Persistent record of a library's builds."""

    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    versions: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] | None = Field(
        default=None,
        description="PipelineOptions the stored compounds were built with.",
    )
    completed: set[str] = Field(default_factory=set, description="InChIKeys built.")
    failed: dict[str, str] = Field(
        default_factory=dict, description="Failure key -> 'category: reason'."
    )
    stats: ManifestStats = Field(default_factory=ManifestStats)

    @field_serializer("completed")
    def _sorted(self, v: set[str]) -> list[str]:
        return sorted(v)

    @classmethod
    def load_or_empty(cls, path: Path) -> BuildManifest:
        """Read an existing manifest, or start fresh if absent / unreadable."""
        if not path.exists():
            return cls()
        try:
            with path.open(encoding="utf-8") as f:
                return cls.model_validate(json.load(f))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Manifest at %s is unreadable (%s); starting fresh.", path, e
            )
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(tz=UTC)
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, prefix=".manifest.", suffix=".tmp"
        ) as tf:
            tf.write(self.model_dump_json(indent=2))
            tmp = Path(tf.name)
        tmp.replace(path)

    def mark_done(self, inchikey: str) -> None:
        self.completed.add(inchikey)

    def mark_failed(self, key: str, reason: str) -> None:
        self.failed[key] = reason


__all__ = ["BuildManifest", "ManifestStats"]
