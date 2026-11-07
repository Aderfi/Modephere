"""Where the pharmacogenomic inputs live, and loading them with their Datasets.

Layout (``paths.pgx`` and ``paths.cache`` from the settings)::

    data/pgx/pharmgkb/var_{drug,pheno,fa}_ann.tsv   PharmGKB variant annotations
    data/pgx/pharmgkb/CREATED_<date>.txt           release stamp of that download
    data/pgx/pharmvar/<GENE>/*.haplotypes.tsv      PharmVar haplotype definitions
    data/pgx/drugs/drug_name_cache.json            drug name -> PubChem CID
    data/pgx/drugs/train_data.tsv                  |-aligned drugs / drugs_cid
    data/cache/ncbi/{dbsnp_esummary,ncbi_spdi}.jsonl   raw NCBI responses (fetch)
    data/cache/clinpgx/allele_tables/<GENE>.xlsx    ClinPGx tables + index.jsonl
    data/pgx/reports/                              resolution report + unresolved.log (ingest)
    data/pgx/export/                               parquet tables (export)

The NCBI cache files and table index have the same format as dev_Pgen's
``data/library/kg/raw/``, so copying that directory replays it offline.
"""

from __future__ import annotations

import logging
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from modephere.config import get_settings
from modephere.core import DataError
from modephere.kg.provenance import Dataset
from modephere.pgx.annotation import AnnotationSource, PGxAnnotation
from modephere.pgx.build import (
    allele_table_dataset,
    local_file_dataset,
    pharmgkb_dataset,
)
from modephere.pgx.fetch import read_table_index
from modephere.pgx.sources.allele_tables import load_allele_table
from modephere.pgx.sources.definitions import HaplotypeDefinition
from modephere.pgx.sources.pharmgkb import (
    ALIGNED_TSV_DATASET,
    NAME_CACHE_DATASET,
    iter_annotations,
    load_drug_cid_provenance,
)
from modephere.pgx.sources.pharmvar import (
    grch38_haplotype_files,
    load_pharmvar,
    pharmvar_dataset,
)

if TYPE_CHECKING:
    from modephere.config import Settings

logger = logging.getLogger(__name__)


class PgxLayout(BaseModel):
    """Every path the pgx layer reads or writes."""

    model_config = ConfigDict(frozen=True)

    pharmgkb: Path
    pharmvar: Path
    name_cache: Path
    aligned_tsv: Path
    ncbi_cache: Path
    allele_tables: Path
    reports: Path
    export: Path

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> PgxLayout:
        s = settings or get_settings()
        root, cache = s.paths.pgx, s.paths.cache
        return cls(
            pharmgkb=root / "pharmgkb",
            pharmvar=root / "pharmvar",
            name_cache=root / "drugs" / "drug_name_cache.json",
            aligned_tsv=root / "drugs" / "train_data.tsv",
            ncbi_cache=cache / "ncbi",
            allele_tables=cache / "clinpgx" / "allele_tables",
            reports=root / "reports",
            export=root / "export",
        )


class Inputs(BaseModel):
    annotations: list[PGxAnnotation]
    pharmvar: list[HaplotypeDefinition]
    tables: list[HaplotypeDefinition]
    datasets: list[Dataset]
    drug_cids: dict[str, tuple[int, str]]
    annotation_datasets: dict[str, str] = Field(default_factory=dict)
    failed_tables: list[str] = Field(default_factory=list)


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


def _load_tables(
    layout: PgxLayout,
) -> tuple[list[HaplotypeDefinition], list[Dataset], list[str]]:
    from openpyxl.utils.exceptions import InvalidFileException

    tables: list[HaplotypeDefinition] = []
    datasets: list[Dataset] = []
    failed: list[str] = []
    for gene, entry in read_table_index(layout.allele_tables).items():
        if entry.status != "ok" or entry.path is None:
            continue
        xlsx = layout.allele_tables / entry.path
        ds = allele_table_dataset(xlsx, gene, entry.url, entry.retrieved_at)
        try:
            tables.extend(load_allele_table(xlsx, ds.id))
        except (
            DataError,
            zipfile.BadZipFile,
            InvalidFileException,
            KeyError,
        ) as e:
            logger.error("Unreadable ClinPGx table %s: %s", gene, e)
            failed.append(gene)
            continue
        datasets.append(ds)
    return tables, datasets, failed


def load_inputs(layout: PgxLayout) -> Inputs:
    """Every local input, each paired with its pinned :class:`Dataset`."""
    datasets: list[Dataset] = []
    annotations: list[PGxAnnotation] = []
    annotation_datasets: dict[str, str] = {}
    for source in AnnotationSource:
        tsv = layout.pharmgkb / f"{source.value}.tsv"
        annotations.extend(iter_annotations(tsv, source))
        ds = pharmgkb_dataset(tsv, layout.pharmgkb)
        datasets.append(ds)
        annotation_datasets[source.value] = ds.id

    pharmvar: list[HaplotypeDefinition] = []
    for tsv in grch38_haplotype_files(layout.pharmvar):
        ds = pharmvar_dataset(tsv, _mtime(tsv))
        datasets.append(ds)
        pharmvar.extend(load_pharmvar(tsv, ds.id))

    tables, table_datasets, failed_tables = _load_tables(layout)
    datasets.extend(table_datasets)

    local_ids: dict[str, str] = {}
    for logical_id, path, name in (
        (NAME_CACHE_DATASET, layout.name_cache, "Drug name -> PubChem CID cache"),
        (ALIGNED_TSV_DATASET, layout.aligned_tsv, "Training data drug/CID alignment"),
    ):
        if path.exists():
            ds = local_file_dataset(logical_id, path, name)
            datasets.append(ds)
            local_ids[logical_id] = ds.id
    drug_cids = {
        drug: (cid, local_ids[logical])
        for drug, (cid, logical) in load_drug_cid_provenance(
            layout.name_cache, layout.aligned_tsv
        ).items()
    }
    return Inputs(
        annotations=annotations,
        pharmvar=pharmvar,
        tables=tables,
        datasets=datasets,
        drug_cids=drug_cids,
        annotation_datasets=annotation_datasets,
        failed_tables=failed_tables,
    )


__all__ = ["Inputs", "PgxLayout", "load_inputs"]
