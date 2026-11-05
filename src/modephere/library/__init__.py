"""Modephere library builder: identifiers -> molecule + 2D/3D pharmacophore library.

From code::

    >>> from modephere.library import build_compound           # one compound, no disk
    >>> art = build_compound("CC(=O)Oc1ccccc1C(=O)O")

    >>> from modephere.library import LibraryBuilder, LibraryBuildConfig
    >>> cfg = LibraryBuildConfig.from_settings(catalog=Path("drugs.tsv"))
    >>> LibraryBuilder(cfg).fetch(); LibraryBuilder(cfg).run()

    >>> from modephere.library import LibraryStore
    >>> art = LibraryStore(cfg.library_root).load("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")

Or the CLI: ``python -m modephere.library --help``.

Exports are resolved lazily, so importing one submodule (e.g. the
standardizer) does not pull in the whole builder stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modephere.library.builder import BuildSummary, FetchSummary, LibraryBuilder
    from modephere.library.config import LibraryBuildConfig
    from modephere.library.pipeline import (
        PipelineOptions,
        build_artifacts,
        build_compound,
    )
    from modephere.library.standardize import StandardizeOptions, standardize
    from modephere.library.storage import LibraryStore

_EXPORTS = {
    "BuildSummary": "modephere.library.builder",
    "FetchSummary": "modephere.library.builder",
    "LibraryBuilder": "modephere.library.builder",
    "LibraryBuildConfig": "modephere.library.config",
    "LibraryStore": "modephere.library.storage",
    "PipelineOptions": "modephere.library.pipeline",
    "StandardizeOptions": "modephere.library.standardize",
    "build_artifacts": "modephere.library.pipeline",
    "build_compound": "modephere.library.pipeline",
    "standardize": "modephere.library.standardize",
}

__all__ = [
    "BuildSummary",
    "FetchSummary",
    "LibraryBuildConfig",
    "LibraryBuilder",
    "LibraryStore",
    "PipelineOptions",
    "StandardizeOptions",
    "build_artifacts",
    "build_compound",
    "standardize",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
