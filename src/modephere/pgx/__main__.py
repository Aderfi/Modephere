"""CLI for the pharmacogenomic layer of the knowledge graph.

Usage::

    python -m modephere.pgx fetch              # network: dbSNP, NCBI SPDI, ClinPGx -> cache
    python -m modephere.pgx ingest [--replace] # offline: cache + inputs -> Neo4j, audit
    python -m modephere.pgx export             # Neo4j -> data/pgx/export/*.parquet
    # afterwards, tie the PharmGKB drugs to ChEMBL (Drug nodes already in the graph):
    python -m modephere.kg link-drugs

Only ``fetch`` touches the network. ``ingest`` builds the whole layer in
memory first; if any upstream response is missing it exits 1 before
connecting to Neo4j. Inputs and outputs: :class:`~modephere.pgx.inputs.PgxLayout`.
NCBI identity: ``[ncbi]`` in settings.toml (``MODEPHERE_NCBI__API_KEY``,
``MODEPHERE_NCBI__EMAIL``); Neo4j: ``[kg]`` (``MODEPHERE_KG__PASSWORD``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

from modephere.config import PROJECT_ROOT, get_settings
from modephere.core import ModephereException, setup_logging
from modephere.kg.graph import KnowledgeGraph
from modephere.kg.provenance import IngestRun
from modephere.pgx.build import build_kg
from modephere.pgx.cache import CacheMissError
from modephere.pgx.export import export_tables
from modephere.pgx.fetch import (
    download_allele_tables,
    plan_fetch,
    read_table_index,
    run_fetch,
)
from modephere.pgx.inputs import Inputs, PgxLayout, load_inputs
from modephere.pgx.ncbi import NCBIClient
from modephere.pgx.report import write_report
from modephere.pgx.write import LAYER, write_build

logger = logging.getLogger(__name__)

REFRESH_STATUSES = frozenset({"not_found", "error"})


def _client(layout: PgxLayout, *, offline: bool, refresh: bool = False) -> NCBIClient:
    return NCBIClient(
        get_settings().ncbi,
        layout.ncbi_cache,
        offline=offline,
        refresh=REFRESH_STATUSES if refresh else frozenset(),
    )


def _fetch(layout: PgxLayout, refresh: bool) -> int:
    inputs = load_inputs(layout)
    pharmvar_genes = {d.gene for d in inputs.pharmvar}
    plan = plan_fetch(inputs.annotations, inputs.pharmvar, pharmvar_genes)
    download_allele_tables(
        plan.allele_table_genes, layout.allele_tables, refresh=refresh
    )
    inputs = load_inputs(layout)  # pick up the freshly downloaded tables
    with _client(layout, offline=False, refresh=refresh) as client:
        run_fetch(plan, inputs.pharmvar + inputs.tables, client)
    logger.info("Fetch complete; NCBI cache at %s", layout.ncbi_cache)
    return 0


def _table_gaps(layout: PgxLayout, inputs: Inputs) -> list[str]:
    """Genes whose ClinPGx table the labels need but the cache cannot supply."""
    needed = plan_fetch(
        inputs.annotations, inputs.pharmvar, {d.gene for d in inputs.pharmvar}
    ).allele_table_genes
    index = read_table_index(layout.allele_tables)
    return sorted(
        g
        for g in needed
        if g not in index or index[g].status == "error" or g in inputs.failed_tables
    )


def _ingest(layout: PgxLayout, replace: bool) -> int:
    inputs = load_inputs(layout)
    gaps = _table_gaps(layout, inputs)
    if gaps:
        logger.error(
            "Ingest aborted before writing anything: ClinPGx allele tables missing or "
            "unreadable for %s; run 'python -m modephere.pgx fetch'",
            gaps,
        )
        return 1
    run = IngestRun.start(
        {"replace": str(replace), "ncbi_cache": str(layout.ncbi_cache)},
        repo=PROJECT_ROOT,
        layer=LAYER,
    )
    try:
        build = build_kg(
            annotations=inputs.annotations,
            definitions=inputs.pharmvar + inputs.tables,
            client=_client(layout, offline=True),
            drug_cids=inputs.drug_cids,
            datasets=inputs.datasets,
            run=run,
            annotation_datasets=inputs.annotation_datasets,
        )
    except CacheMissError as e:
        logger.error("Ingest aborted before writing anything: %s", e)
        return 1
    build.run.finished_at = datetime.now(UTC)
    with KnowledgeGraph.connect() as graph:
        summary = write_build(graph, build, replace=replace)
        audit = graph.audit()
    write_report(build, layout.reports)
    logger.info(
        "pgx run %s: %d nodes, %d relationships (%d older run(s) replaced); "
        "run 'python -m modephere.kg link-drugs' to tie its drugs to ChEMBL",
        summary.run_id,
        summary.nodes,
        summary.relationships,
        summary.replaced_runs,
    )
    return _report_audit(audit)


def _report_audit(audit: dict[str, object]) -> int:
    if any(audit.values()):
        logger.error("Provenance audit failed: %s", audit)
        return 1
    logger.info("Provenance audit passed")
    return 0


def _export(layout: PgxLayout) -> int:
    with KnowledgeGraph.connect() as graph:
        export_tables(graph, layout.export)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m modephere.pgx",
        description="Pharmacogenomic layer: PharmGKB, PharmVar, dbSNP -> Neo4j.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="dbSNP / NCBI / ClinPGx -> raw cache.")
    fetch.add_argument(
        "--refresh",
        action="store_true",
        help="Re-request cached not_found / error responses (history kept).",
    )
    ingest = sub.add_parser("ingest", help="Build the layer offline, write to Neo4j.")
    ingest.add_argument(
        "--replace",
        action="store_true",
        help="Withdraw the previous pgx run(s) first (ChEMBL data is untouched).",
    )
    sub.add_parser("export", help="Annotation rows + standardization tables.")
    for parser in sub.choices.values():
        parser.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(name="pgx", level=logging.DEBUG if args.verbose else logging.INFO)
    layout = PgxLayout.from_settings()
    try:
        if args.command == "fetch":
            return _fetch(layout, args.refresh)
        if args.command == "ingest":
            return _ingest(layout, args.replace)
        return _export(layout)
    except ModephereException as e:
        logger.error("%s", e)  # noqa: TRY400 -- user-facing message
        return 1


if __name__ == "__main__":
    sys.exit(main())
