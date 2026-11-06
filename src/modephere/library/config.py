"""Pydantic configuration for a library build run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from modephere.chembl.extract import latest_staging
from modephere.config import PubChemSettings, get_settings
from modephere.library.pipeline import PipelineOptions
from modephere.types import AmbiguityPolicy, FrozenModel


class LibraryBuildConfig(FrozenModel):
    """All inputs, outputs and behaviour of one build, validated up front.

    ``from_settings()`` roots everything at the project's standard paths; any
    field can be overridden.
    """

    # Inputs: a catalog file and/or identifiers given directly.
    catalog: Path | None = Field(default=None, description="tsv/csv/txt/parquet/json.")
    smiles: tuple[str, ...] = ()
    cids: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    chembl_ids: tuple[str, ...] = ()

    # Locations
    library_root: Path
    cache_dir: Path = Field(..., description="Raw PubChem responses (JSONL).")
    chembl_staging: Path | None = Field(
        default=None,
        description="Staged ChEMBL release (data/chembl/chembl_<N>). None: names "
        "resolve through PubChem only and compounds get no chembl_id.",
    )
    failures_log: Path | None = None
    saturation_log: Path | None = None

    # Behaviour
    offline: bool = Field(
        default=True,
        description="Resolve from the raw cache only (run 'fetch' first). "
        "False lets the build query PubChem itself.",
    )
    force: bool = Field(default=False, description="Rebuild compounds already stored.")
    ambiguity: AmbiguityPolicy = AmbiguityPolicy.FIRST_HIT
    pipeline: PipelineOptions = Field(default_factory=PipelineOptions)
    pubchem: PubChemSettings = Field(default_factory=PubChemSettings)
    flush_every: int = Field(
        default=200, gt=0, description="Compounds per table flush."
    )
    conformer_batch_size: int = Field(
        default=64,
        gt=0,
        description="Compounds per GPU conformer batch (nvmolkit backend only; "
        "the CPU backend builds one compound at a time).",
    )

    @model_validator(mode="after")
    def _has_input(self) -> Self:
        if self.catalog is None and not (
            self.smiles or self.cids or self.names or self.chembl_ids
        ):
            msg = "give a catalog and/or SMILES / CIDs / names / ChEMBL IDs to build"
            raise ValueError(msg)
        return self

    @property
    def manifest_path(self) -> Path:
        return self.library_root / "build_manifest.json"

    @classmethod
    def from_settings(
        cls, *, use_chembl: bool = True, **overrides: object
    ) -> LibraryBuildConfig:
        """Config rooted at ``get_settings().paths``; ``overrides`` win.

        With ``use_chembl`` the newest staged ChEMBL release under
        ``paths.chembl`` is used, if there is one.
        """
        settings = get_settings()
        stamp = datetime.now(tz=UTC).strftime("%Y%m%d")
        logs = settings.paths.logs / "library"
        defaults: dict[str, object] = {
            "library_root": settings.paths.library,
            "cache_dir": settings.paths.cache / "pubchem",
            "failures_log": logs / f"build_failures_{stamp}.log",
            "saturation_log": logs / f"feature_saturation_{stamp}.log",
            "pubchem": settings.pubchem,
            "chembl_staging": (
                latest_staging(settings.paths.chembl) if use_chembl else None
            ),
        }
        given = {k: v for k, v in overrides.items() if v is not None}
        return cls.model_validate(defaults | given)


__all__ = ["LibraryBuildConfig"]
