"""Resolution report: what resolved, how, and — when not — why.

Writes ``resolution_report.tsv`` (one row per PharmGKB allele label),
``disagreements.tsv`` (PharmVar vs dbSNP cross-check findings) and
``unresolved.log`` (every unresolved case with its reason), and logs the
headline numbers: resolved label mentions, counts per kind x status x reason,
the strand-mismatch rate of the allele-specific inference, and how many
genotype mentions (``*1/*22``, ``CT``) resolved, per reason.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import polars as pl

from modephere.core import build_file_logger

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.pgx.build import KGBuild

logger = logging.getLogger(__name__)

RESOLUTION_FILE = "resolution_report.tsv"
DISAGREEMENTS_FILE = "disagreements.tsv"
UNRESOLVED_LOG = "unresolved.log"
UNRESOLVED_LOGGER = "modephere.pgx.unresolved"


def write_unresolved_log(build: KGBuild, out_dir: Path) -> Path:
    """Every unresolved case of the run, one line each, in ``unresolved.log``.

    Rewritten per ingest (it describes one run). Line format::

        <time> | WARNING | <category> | <reason> | <subject> | annotation=<id> | <detail>

    sorted by category, reason and subject, so two runs diff cleanly.
    """
    path = out_dir / UNRESOLVED_LOG
    path.unlink(missing_ok=True)
    audit = build_file_logger(UNRESOLVED_LOGGER, path)
    audit.info("pgx run %s: %d unresolved cases", build.run.id, len(build.unresolved))
    cases = sorted(
        build.unresolved,
        key=lambda c: (c.category, c.reason, c.subject, c.annotation_id or ""),
    )
    for c in cases:
        audit.warning(
            "%s | %s | %s | annotation=%s | %s",
            c.category,
            c.reason,
            c.subject,
            c.annotation_id or "-",
            c.detail or "-",
        )
    for handler in list(audit.handlers):
        audit.removeHandler(handler)
        handler.close()
    by_category = Counter(c.category for c in build.unresolved)
    logger.info(
        "Unresolved cases: %d (%s) -> %s",
        len(build.unresolved),
        ", ".join(f"{k} {n}" for k, n in sorted(by_category.items())) or "none",
        path,
    )
    return path


def write_report(build: KGBuild, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = pl.DataFrame(
        [r.model_dump() for r in build.resolution],
        schema={
            "label": pl.String,
            "kind": pl.String,
            "status": pl.String,
            "reason": pl.String,
            "n_annotations": pl.Int64,
            "n_variants": pl.Int64,
            "path": pl.String,
        },
    ).sort(["status", "n_annotations"], descending=[False, True])
    path = out_dir / RESOLUTION_FILE
    resolution.write_csv(path, separator="\t")

    disagreements = pl.DataFrame(
        [
            {**d.model_dump(), "dbsnp_spdis": ",".join(d.dbsnp_spdis)}
            for d in build.disagreements
        ],
        schema={
            "issue": pl.String,
            "haplotype": pl.String,
            "rsid": pl.String,
            "pharmvar_spdi": pl.String,
            "dbsnp_spdis": pl.String,
        },
    )
    disagreements.write_csv(out_dir / DISAGREEMENTS_FILE, separator="\t")

    mentions: Counter[str] = Counter()
    for r in build.resolution:
        mentions[r.status] += r.n_annotations
    total = sum(mentions.values())
    logger.info(
        "KG run %s: resolved mentions %d/%d (%.1f%%), %d labels",
        build.run.id,
        mentions["resolved"],
        total,
        100 * mentions["resolved"] / total if total else 0.0,
        len(build.resolution),
    )
    by_kind = Counter((r.kind, r.status, r.reason or "-") for r in build.resolution)
    for (kind, status, reason), n in sorted(by_kind.items()):
        logger.info("  %-9s %-10s %-28s %6d labels", kind, status, reason, n)

    attempted = sum(build.inference_counts.values())
    mismatches = build.inference_counts.get("allele_strand_mismatch", 0)
    logger.info(
        "Allele inference: %d attempted, %d linked, strand mismatches %d/%d (%.4f)",
        attempted,
        build.inference_counts.get("linked", 0),
        mismatches,
        attempted,
        mismatches / attempted if attempted else 0.0,
    )
    for reason, n in sorted(build.inference_counts.items()):
        logger.info("  inference %-24s %6d", reason, n)
    genotypes = build.genotype_counts
    mentions_gt = {k: n for k, n in genotypes.items() if not k.startswith("skipped:")}
    resolved_gt = sum(n for k, n in mentions_gt.items() if k.startswith("resolved:"))
    logger.info(
        "Genotypes: %d/%d mentions resolved",
        resolved_gt,
        sum(mentions_gt.values()),
    )
    for key, n in sorted(genotypes.items()):
        logger.info("  genotype %-36s %6d", key, n)
    write_unresolved_log(build, out_dir)
    for issue, n in sorted(Counter(d.issue for d in build.disagreements).items()):
        logger.info("PharmVar/dbSNP cross-check: %d %s", n, issue)
    logger.info("Reports written to %s", out_dir)
    return path


__all__ = [
    "DISAGREEMENTS_FILE",
    "RESOLUTION_FILE",
    "UNRESOLVED_LOG",
    "write_report",
    "write_unresolved_log",
]
