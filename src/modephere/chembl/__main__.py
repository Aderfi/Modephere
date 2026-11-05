"""CLI for the ChEMBL staging step.

Usage::

    # SQLite release -> data/chembl/chembl_<N>/*.parquet (path from settings)
    python -m modephere.chembl extract
    python -m modephere.chembl extract --sqlite ~/chembl_37.db --min-confidence 8
    # re-aggregate bioactivity at another confidence, without re-reading SQLite
    python -m modephere.chembl aggregate --min-confidence 9
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from modephere.chembl.extract import (
    MANIFEST,
    StagingManifest,
    aggregate_bioactivity,
    extract,
    latest_staging,
    read_table,
)
from modephere.chembl.source import ChemblRelease
from modephere.config import get_settings
from modephere.core import ConfigurationError, ModephereException, setup_logging

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modephere.chembl",
        description="Stage a local ChEMBL SQLite release as Parquet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="SQLite release -> staging Parquet.")
    ex.add_argument("--sqlite", type=Path, help="chembl_<N>.db (default: settings).")
    ex.add_argument("--no-checksum", action="store_true", help="Skip the SHA-256.")
    agg = sub.add_parser("aggregate", help="Recompute bioactivity.parquet.")
    agg.add_argument("--release", help="Staging directory name, e.g. chembl_37.")
    for p in (ex, agg):
        p.add_argument("--staging", type=Path, help="Staging root (default: settings).")
        p.add_argument("--min-confidence", type=int, help="Assay target confidence.")
        p.add_argument("--verbose", "-v", action="store_true")
    return parser


def _latest(root: Path) -> Path:
    staging = latest_staging(root)
    if staging is None:
        msg = f"no extracted ChEMBL release under {root}; run 'extract' first"
        raise ConfigurationError(msg)
    return staging


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(name="chembl", level=logging.DEBUG if args.verbose else logging.INFO)
    settings = get_settings()
    root = args.staging or settings.paths.chembl
    min_conf = (
        settings.chembl.min_confidence
        if args.min_confidence is None
        else args.min_confidence
    )
    try:
        if args.command == "extract":
            path = args.sqlite or settings.chembl.sqlite
            if path is None:
                msg = "no ChEMBL SQLite file: pass --sqlite or set MODEPHERE_CHEMBL__SQLITE"
                raise ConfigurationError(msg)
            release = ChemblRelease.open(path.expanduser())
            logger.info("Extracting %s from %s", release.id, release.path)
            out = extract(
                release, root, min_confidence=min_conf, checksum=not args.no_checksum
            )
            logger.info("Staged %s in %s", release.id, out)
            return 0
        staging = root / args.release if args.release else _latest(root)
        manifest = StagingManifest.read(staging)
        summary = aggregate_bioactivity(read_table(staging, "activities"), min_conf)
        summary.write_parquet(staging / "bioactivity.parquet", compression="zstd")
        manifest = manifest.model_copy(
            update={
                "min_confidence": min_conf,
                "rows": manifest.rows | {"bioactivity": summary.height},
            }
        )
        (staging / MANIFEST).write_text(manifest.model_dump_json(indent=2), "utf-8")
        logger.info(
            "bioactivity: %d pairs at confidence >= %d", summary.height, min_conf
        )
        return 0
    except ModephereException as e:
        logger.error("%s", e)  # noqa: TRY400 -- user-facing message
        return 1


if __name__ == "__main__":
    sys.exit(main())
