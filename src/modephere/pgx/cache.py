"""Offline guarantee of the pharmacogenomic build.

Raw NCBI responses live in :class:`~modephere.library.ingest.raw_cache.RawCache`
files under ``<paths.cache>/ncbi``; ``ingest`` reads them through an offline
:class:`~modephere.pgx.ncbi.NCBIClient`, which raises :class:`CacheMissError`
for anything ``fetch`` never recorded.
"""

from __future__ import annotations

from modephere.core import DataError


class CacheMissError(DataError):
    """Raised in offline mode when required responses were never fetched."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        preview = ", ".join(missing[:10])
        super().__init__(
            f"{len(missing)} required upstream responses are not cached "
            f"(first: {preview}); run 'python -m modephere.pgx fetch'"
        )


__all__ = ["CacheMissError"]
