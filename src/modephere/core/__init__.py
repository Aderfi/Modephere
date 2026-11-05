"""Cross-cutting primitives: exception hierarchy and logging setup."""

from modephere.core.exceptions import (
    CompoundError,
    ConfigurationError,
    DataError,
    ModephereException,
    ResourceError,
    StorageError,
    UpstreamError,
)
from modephere.core.log import build_file_logger, setup_logging

__all__ = [
    "CompoundError",
    "ConfigurationError",
    "DataError",
    "ModephereException",
    "ResourceError",
    "StorageError",
    "UpstreamError",
    "build_file_logger",
    "setup_logging",
]
