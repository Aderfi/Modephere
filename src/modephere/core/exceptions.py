"""Exception hierarchy for Modephere.

Every library error derives from :class:`ModephereException`, so a caller can
catch the whole family at once and still single out a subtype. Per-compound
pipeline failures raise :class:`CompoundError`, which carries the stage and
:class:`~modephere.types.FailureCategory` the builder records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from modephere.types.enums import BuildStage, FailureCategory


class ModephereException(Exception):
    """Base for all Modephere-specific errors."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})

    def __str__(self) -> str:
        if self.details:
            ctx = " | ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} | Context: {{{ctx}}}"
        return self.message


class ConfigurationError(ModephereException):
    """Invalid or missing configuration, or an optional extra not installed."""


class DataError(ModephereException):
    """Invalid or incompatible data (bad catalog, corrupt cache or artifact)."""


class ResourceError(ModephereException):
    """Insufficient or unreachable resources (network, disk)."""


class UpstreamError(ResourceError):
    """An upstream service (PubChem) kept failing after every retry."""


class StorageError(DataError):
    """A library artifact or table is missing, corrupt or schema-mismatched."""


class CompoundError(DataError):
    """One compound failed one pipeline stage; the build records it and goes on."""

    def __init__(
        self,
        stage: BuildStage,
        category: FailureCategory,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason, details)
        self.stage = stage
        self.category = category
        self.reason = reason


__all__ = [
    "CompoundError",
    "ConfigurationError",
    "DataError",
    "ModephereException",
    "ResourceError",
    "StorageError",
    "UpstreamError",
]
