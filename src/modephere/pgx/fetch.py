"""The networked step: fill the raw cache with everything the build needs.

``run_fetch`` executes the same resolution functions the offline build uses
(:mod:`modephere.pgx.resolve`) against an online client, so every response the
build will read is cached here. ClinPGx allele-definition tables are downloaded
to ``<cache>/clinpgx/allele_tables/<GENE>.xlsx`` with an append-only
``index.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import BaseModel, Field

from modephere.core import ResourceError
from modephere.pgx.labels import AlleleKind, Structure, classify_label
from modephere.pgx.ncbi import ESUMMARY_BATCH
from modephere.pgx.resolve import canonicalize_row, resolve_refsnps

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from modephere.pgx.annotation import PGxAnnotation
    from modephere.pgx.ncbi import NCBIClient
    from modephere.pgx.sources.definitions import HaplotypeDefinition

logger = logging.getLogger(__name__)

CLINPGX_TABLE_URL = "https://api.clinpgx.org/v1/download/file/attachment/{gene}_allele_definition_table.xlsx"
TABLE_INDEX = "index.jsonl"
_TABLE_RETRIES = (1, 2, 4, 8)


class FetchPlan(BaseModel):
    rsids: set[int]
    allele_table_genes: set[str]
    #: Genes with a whole-gene deletion allele: their span is fetched from NCBI Gene.
    deletion_genes: set[str] = Field(default_factory=set)


class TableIndexEntry(BaseModel):
    gene: str
    url: str
    status: Literal["ok", "not_found", "error"]
    retrieved_at: datetime
    sha256: str | None = None
    path: str | None = None


def plan_fetch(
    annotations: Iterable[PGxAnnotation],
    pharmvar: list[HaplotypeDefinition],
    pharmvar_genes: set[str],
) -> FetchPlan:
    """rsIDs from labels and PharmVar rows; table genes of star labels PharmVar lacks."""
    rsids: set[int] = set()
    table_genes: set[str] = set()
    deletion_genes: set[str] = set()
    for annotation in annotations:
        for label in annotation.variants:
            c = classify_label(label)
            if c.kind is AlleleKind.RSID and c.rsid is not None:
                rsids.add(c.rsid)
            elif c.structure is Structure.GENE_DELETION and c.haplotype_name:
                deletion_genes.add(c.haplotype_name.split("*", 1)[0])
            elif c.kind is AlleleKind.STAR and c.haplotype_name:
                gene = c.haplotype_name.split("*", 1)[0]
                if gene not in pharmvar_genes:
                    table_genes.add(gene)
    for definition in pharmvar:
        for row in definition.rows:
            if row.rsid and row.rsid.startswith("rs") and row.rsid[2:].isdigit():
                rsids.add(int(row.rsid[2:]))
    return FetchPlan(
        rsids=rsids, allele_table_genes=table_genes, deletion_genes=deletion_genes
    )


def run_fetch(
    plan: FetchPlan, definitions: list[HaplotypeDefinition], client: NCBIClient
) -> None:
    """Fill the cache; an input that keeps failing is skipped, not fatal to the rest.

    Every failure is collected and re-raised at the end as one ResourceError,
    so a rerun retries exactly those keys (nothing failed is ever cached).
    """
    failures: list[str] = []
    rsids = sorted(plan.rsids)
    logger.info("Fetching dbSNP records for %d rsIDs", len(rsids))
    for start in range(0, len(rsids), ESUMMARY_BATCH):
        try:
            resolve_refsnps(rsids[start : start + ESUMMARY_BATCH], client)
        except ResourceError as e:
            failures.append(str(e))
    logger.info("Fetching NCBI Gene spans for %d genes", len(plan.deletion_genes))
    for symbol in sorted(plan.deletion_genes):
        try:
            client.gene_summary(symbol)
        except ResourceError as e:
            failures.append(str(e))
    rows = {(r.spdi_input, r.hgvs_input): r for d in definitions for r in d.rows}
    logger.info("Canonicalizing %d distinct definition variants", len(rows))
    for n, row in enumerate(rows.values(), start=1):
        try:
            canonicalize_row(row, client)
        except ResourceError as e:
            failures.append(str(e))
        if n % 250 == 0:
            logger.info("Canonicalized %d/%d", n, len(rows))
    if failures:
        for failure in failures:
            logger.error("fetch failure: %s", failure)
        msg = f"{len(failures)} request(s) failed; rerun 'fetch' to retry them (first: {failures[0]})"
        raise ResourceError(msg)


def read_table_index(dest: Path) -> dict[str, TableIndexEntry]:
    index = dest / TABLE_INDEX
    if not index.exists():
        return {}
    entries = [
        TableIndexEntry.model_validate_json(line)
        for line in index.read_text().splitlines()
        if line.strip()
    ]
    return {e.gene: e for e in entries}


def download_allele_tables(
    genes: Iterable[str],
    dest: Path,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    refresh: bool = False,
) -> dict[str, TableIndexEntry]:
    """Download missing ClinPGx tables; a 404 is recorded as ``not_found``.

    With ``refresh``, tables whose last attempt was not ``ok`` are retried; the
    new attempt is appended to the index (last entry per gene wins).
    """
    dest.mkdir(parents=True, exist_ok=True)
    known = read_table_index(dest)
    todo = {g for g in genes if g not in known or (refresh and known[g].status != "ok")}
    if not todo:
        return known
    with httpx.Client(timeout=30, transport=transport, follow_redirects=True) as http:
        for gene in sorted(todo):
            known[gene] = _download_table(http, gene, dest, sleep)
    return known


def _download_table(
    http: httpx.Client, gene: str, dest: Path, sleep: Callable[[float], None]
) -> TableIndexEntry:
    url = CLINPGX_TABLE_URL.format(gene=gene)
    response = None
    for delay in (*_TABLE_RETRIES, None):
        sleep(1.0)
        try:
            response = http.get(url)
        except httpx.TransportError as e:
            logger.warning("ClinPGx %s: %s", gene, e)
            response = None
        if (
            response is not None
            and response.status_code < 500
            and response.status_code != 429
        ):
            break
        if delay is None:
            msg = f"ClinPGx download failed after retries: {url}"
            raise ResourceError(msg)
        sleep(delay)
    assert response is not None
    entry = TableIndexEntry(
        gene=gene, url=url, status="error", retrieved_at=datetime.now(UTC)
    )
    if response.status_code == 200:
        path = dest / f"{gene}.xlsx"
        path.write_bytes(response.content)
        entry = entry.model_copy(
            update={
                "status": "ok",
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "path": path.name,
            }
        )
    elif response.status_code == 404:
        entry = entry.model_copy(update={"status": "not_found"})
    with (dest / TABLE_INDEX).open("a") as f:
        f.write(json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n")
    logger.info("ClinPGx table %s: %s", gene, entry.status)
    return entry


__all__ = [
    "CLINPGX_TABLE_URL",
    "FetchPlan",
    "TableIndexEntry",
    "download_allele_tables",
    "plan_fetch",
    "read_table_index",
    "run_fetch",
]
