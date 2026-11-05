"""Append-only JSONL cache of raw upstream responses (port of dev_Pgen's KG cache).

Every PubChem answer is written here verbatim, with the request and retrieval
time, before anything uses it. A build then resolves from this cache alone,
so it is reproducible offline and every structure traces back to the response
it came from. The last line for a key wins on reload, so refreshing a key keeps
the full retrieval history in the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import AwareDatetime, BaseModel, Field

from modephere.core import DataError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """One upstream response."""

    key: str
    request: str = Field(..., description="Request URL (+ body), never secrets.")
    status: Literal["ok", "not_found"]
    retrieved_at: AwareDatetime
    response: dict[str, Any] | None = None


class RawCache:
    """Key -> :class:`CacheEntry` map persisted as one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, CacheEntry] = {}
        self._needs_newline = False
        # Byte offset of a truncated final line, dropped before the next append:
        # appending after it would leave it mid-file and corrupt the next load.
        self._truncate_at: int | None = None
        if path.exists():
            self._load()

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        last = max((n for n, line in enumerate(lines) if line.strip()), default=-1)
        for n, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                entry = CacheEntry.model_validate_json(line)
            except ValueError as e:
                if n == last:
                    # A process killed mid-write leaves a partial final line.
                    logger.warning(
                        "%s: skipping truncated final line %d", self.path, n + 1
                    )
                    data = self.path.read_bytes().rstrip(b"\n")
                    self._truncate_at = data.rfind(b"\n") + 1
                    continue
                msg = f"{self.path}:{n + 1}: corrupt cache line: {e}"
                raise DataError(msg) from e
            self._entries[entry.key] = entry
        if self._truncate_at is None:
            self._needs_newline = bool(text) and not text.endswith("\n")

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> CacheEntry | None:
        return self._entries.get(key)

    def put(self, entry: CacheEntry, *, overwrite: bool = False) -> None:
        """Append ``entry``; an existing key is replaced only with ``overwrite``."""
        if entry.key in self._entries and not overwrite:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._truncate_at is not None:
            with self.path.open("r+b") as f:
                f.truncate(self._truncate_at)
            self._truncate_at = None
            self._needs_newline = False
        line = json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(("\n" if self._needs_newline else "") + line)
        self._needs_newline = False
        self._entries[entry.key] = entry

    def keys(self) -> set[str]:
        return set(self._entries)

    def sha256(self) -> str:
        """Checksum of the cache file on disk (of an empty file if absent)."""
        data = self.path.read_bytes() if self.path.exists() else b""
        return hashlib.sha256(data).hexdigest()


__all__ = ["CacheEntry", "RawCache"]
