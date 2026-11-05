"""Top-level orchestrator: catalog -> resolved -> standardized -> stored library.

    >>> from modephere.library import LibraryBuilder, LibraryBuildConfig
    >>> cfg = LibraryBuildConfig.from_settings(catalog=Path("drugs.tsv"))
    >>> LibraryBuilder(cfg).fetch()     # the only networked step
    >>> LibraryBuilder(cfg).run()       # offline, resumable

Contract (as dev_Pgen's builders): resumable, since a compound already stored is
skipped unless ``force``; per-compound failures never stop the run. They are
counted by :class:`~modephere.types.FailureCategory`, written one per line to a
dedicated, non-propagating file logger, recorded on the manifest and in
``failures.parquet``. Two queries that standardize to the same InChIKey build
it once (``duplicates``).
"""

from __future__ import annotations

import logging
import platform
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import rdkit
from tqdm.auto import tqdm

from modephere.chembl.lookup import ChemblLookup
from modephere.core import CompoundError, build_file_logger
from modephere.library.ingest.catalog import (
    CatalogLoad,
    load_catalog,
    queries_from_values,
)
from modephere.library.ingest.pubchem import PubChemClient
from modephere.library.ingest.resolve import Resolver
from modephere.library.manifest import BuildManifest
from modephere.library.molecules import SATURATING_FEATURES, FeatureSaturation
from modephere.library.pipeline import build_artifacts
from modephere.library.standardize import standardize
from modephere.library.storage import LibraryStore
from modephere.types import BuildStage, FailureCategory, FailureRecord

if TYPE_CHECKING:
    import httpx

    from modephere.library.config import LibraryBuildConfig
    from modephere.library.ingest.resolve import Resolution
    from modephere.types import CompoundArtifacts

logger = logging.getLogger(__name__)

_FAILURE_LOGGER = "modephere.library.build_failures"
_SATURATION_LOGGER = "modephere.library.feature_saturation"


def toolkit_versions() -> dict[str, str]:
    from importlib.metadata import version

    return {
        "modephere": version("modephere"),
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "polars": pl.__version__,
        "python": platform.python_version(),
    }


@dataclass(frozen=True)
class BuildSummary:
    queries: int
    built: int
    skipped: int
    duplicates: int
    failed: int
    failures: dict[FailureCategory, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failed == 0


@dataclass(frozen=True)
class FetchSummary:
    queries: int
    resolved: int
    failed: int
    failures: dict[FailureCategory, int] = field(default_factory=dict)


class LibraryBuilder:
    """End-to-end orchestration; ``fetch()`` and ``run()`` are the public steps."""

    def __init__(
        self,
        config: LibraryBuildConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.store = LibraryStore(config.library_root)
        self._transport = transport  # tests inject httpx.MockTransport
        self.failures: list[FailureRecord] = []
        self.saturation = FeatureSaturation()
        self._failure_logger = build_file_logger(_FAILURE_LOGGER, config.failures_log)
        self._saturation_logger = build_file_logger(
            _SATURATION_LOGGER, config.saturation_log
        )

    # ----- inputs --------------------------------------------------------- #

    def _load_queries(self) -> CatalogLoad:
        cfg = self.config
        load = load_catalog(cfg.catalog) if cfg.catalog is not None else CatalogLoad()
        direct = queries_from_values(
            smiles=cfg.smiles, cids=cfg.cids, names=cfg.names, chembl_ids=cfg.chembl_ids
        )
        load.queries += direct.queries
        load.failures += direct.failures
        return load

    def _resolver(self, client: PubChemClient) -> Resolver:
        return Resolver(client, ambiguity=self.config.ambiguity, chembl=self.chembl)

    @cached_property
    def chembl(self) -> ChemblLookup | None:
        """The staged ChEMBL release's lookup (loaded once), if configured."""
        staging = self.config.chembl_staging
        if staging is None:
            return None
        lookup = ChemblLookup(staging)
        logger.info(
            "ChEMBL: %s (%d molecules) from %s", lookup.release, len(lookup), staging
        )
        return lookup

    def _client(self, *, offline: bool) -> PubChemClient:
        return PubChemClient(
            self.config.pubchem,
            self.config.cache_dir,
            offline=offline,
            transport=self._transport,
        )

    def _record(self, record: FailureRecord) -> None:
        self.failures.append(record)
        self._failure_logger.warning(
            "build failed | key=%s | stage=%s | nature=%s | reason=%s",
            record.key,
            record.stage,
            record.category,
            record.reason,
        )

    def _record_error(self, key: str, error: CompoundError) -> None:
        self._record(
            FailureRecord(
                key=key, stage=error.stage, category=error.category, reason=error.reason
            )
        )

    # ----- fetch ---------------------------------------------------------- #

    def fetch(self) -> FetchSummary:
        """Resolve every query online, filling the raw cache. Builds nothing."""
        self.failures = []
        load = self._load_queries()
        for record in load.failures:
            self._record(record)
        with self._client(offline=False) as client:
            results = self._resolver(client).resolve_many(load.queries)
        for query, result in zip(load.queries, results, strict=True):
            if isinstance(result, CompoundError):
                self._record_error(query.key, result)
        counts = Counter(f.category for f in self.failures)
        summary = FetchSummary(
            queries=len(load.queries) + len(load.failures),
            resolved=sum(not isinstance(r, CompoundError) for r in results),
            failed=len(self.failures),
            failures=dict(counts),
        )
        logger.info(
            "Fetch done: %d queries, %d resolved, %d failed (cache: %s)",
            summary.queries,
            summary.resolved,
            summary.failed,
            self.config.cache_dir,
        )
        return summary

    # ----- build ---------------------------------------------------------- #

    def run(self) -> BuildSummary:
        """Build the library; returns the per-outcome counts."""
        cfg = self.config
        self.failures = []
        self.saturation = FeatureSaturation()
        manifest = BuildManifest.load_or_empty(cfg.manifest_path)
        options = cfg.pipeline.model_dump(mode="json")
        if (
            manifest.options not in (None, options)
            and not cfg.force
            and len(self.store)
        ):
            logger.warning(
                "The library at %s was built with other pipeline options; stored "
                "compounds are kept as they are. Use force to rebuild them.",
                cfg.library_root,
            )
        manifest.options = options
        manifest.versions = toolkit_versions()

        load = self._load_queries()
        for record in load.failures:
            self._record(record)
        logger.info(
            "Library build: %d queries -> %s (offline=%s, 3D=%s, force=%s)",
            len(load.queries),
            cfg.library_root,
            cfg.offline,
            cfg.pipeline.build_3d,
            cfg.force,
        )
        with self._client(offline=cfg.offline) as client:
            resolutions = self._resolver(client).resolve_many(load.queries)

        tally: Counter[str] = Counter()
        seen: set[str] = set()
        pending: list[CompoundArtifacts] = []
        pairs = zip(load.queries, resolutions, strict=True)
        for query, resolved in tqdm(pairs, total=len(load.queries), desc="Compounds"):
            outcome, art = self._build_one(query.key, resolved, seen, manifest)
            tally[outcome] += 1
            if art is not None:
                pending.append(art)
            if len(pending) >= cfg.flush_every:
                self._flush(pending, manifest)
                pending = []
        self._flush(pending, manifest)

        for record in self.failures:
            manifest.mark_failed(record.key, f"{record.category}: {record.reason}")
        self.store.write_failures(self.failures)
        counts = Counter(f.category for f in self.failures)
        summary = BuildSummary(
            queries=len(load.queries) + len(load.failures),
            built=tally["built"],
            skipped=tally["skipped"],
            duplicates=tally["duplicate"],
            failed=len(self.failures),
            failures=dict(counts),
        )
        manifest.stats.queries = summary.queries
        manifest.stats.built = summary.built
        manifest.stats.skipped = summary.skipped
        manifest.stats.duplicates = summary.duplicates
        manifest.stats.failed = summary.failed
        manifest.stats.failures_by_category = {str(k): v for k, v in counts.items()}
        manifest.save(cfg.manifest_path)
        self._log_summary(summary)
        return summary

    def _build_one(  # noqa: PLR0911 -- one return per outcome
        self,
        key: str,
        resolved: Resolution,
        seen: set[str],
        manifest: BuildManifest,
    ) -> tuple[str, CompoundArtifacts | None]:
        if isinstance(resolved, CompoundError):
            self._record_error(key, resolved)
            return "failed", None
        try:
            std = standardize(resolved, self.config.pipeline.standardize)
        except CompoundError as e:
            self._record_error(key, e)
            return "failed", None

        inchikey = std.record.inchikey
        if inchikey in seen:
            logger.info(
                "%s duplicates an earlier query (%s); built once", key, inchikey
            )
            return "duplicate", None
        seen.add(inchikey)
        if self.store.exists(inchikey) and not self.config.force:
            manifest.mark_done(inchikey)
            return "skipped", None

        before = self.saturation.total
        try:
            art = build_artifacts(std, self.config.pipeline, saturation=self.saturation)
        except CompoundError as e:
            self._record_error(inchikey, e)
            return "failed", None
        if (events := self.saturation.total - before) > 0:
            self._saturation_logger.warning(
                "one-hot saturation | inchikey=%s | events=%d | smiles=%s",
                inchikey,
                events,
                std.record.working_smiles,
            )
        try:
            self.store.save(art)
        except OSError as e:
            self._record(
                FailureRecord(
                    key=inchikey,
                    stage=BuildStage.SAVE,
                    category=FailureCategory.SAVE_ERROR,
                    reason=str(e),
                )
            )
            return "failed", None
        manifest.mark_done(inchikey)
        return "built", art

    def _flush(self, arts: list[CompoundArtifacts], manifest: BuildManifest) -> None:
        if arts:
            self.store.upsert(arts)
        manifest.save(self.config.manifest_path)

    def _log_summary(self, s: BuildSummary) -> None:
        logger.info(
            "Library build done: %d built, %d skipped, %d duplicates, %d failed.",
            s.built,
            s.skipped,
            s.duplicates,
            s.failed,
        )
        breakdown = ", ".join(f"{c}={n}" for c, n in sorted(s.failures.items()))
        self._failure_logger.info(
            "Build run finished | built=%d skipped=%d failed=%d | %s",
            s.built,
            s.skipped,
            s.failed,
            breakdown or "no failures",
        )
        if s.failed:
            logger.warning("Failure breakdown: %s", breakdown)
        if s.failures.get(FailureCategory.CACHE_MISS):
            logger.warning(
                "%d queries were never fetched: run 'python -m modephere.library "
                "fetch' (or build with --online).",
                s.failures[FailureCategory.CACHE_MISS],
            )
        total = self.saturation.total
        by_feature = ", ".join(
            f"{f}={self.saturation.by_feature[f]}" for f in SATURATING_FEATURES
        )
        self._saturation_logger.info(
            "Build run finished | events=%d | %s", total, by_feature
        )
        for value, count in self.saturation.by_value.most_common():
            self._saturation_logger.info("  out-of-vocab | %s | count=%d", value, count)
        if total:
            logger.warning(
                "One-hot saturation: %d events (%s); see the saturation log.",
                total,
                by_feature,
            )


__all__ = ["BuildSummary", "FetchSummary", "LibraryBuilder", "toolkit_versions"]
