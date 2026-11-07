"""PharmVar ``<GENE>.<NC>.haplotypes.tsv`` → :class:`HaplotypeDefinition`.

Each row is one variant of one haplotype. Coordinates are 1-based with ``-``
for an absent allele; insertions give the two flanking positions. Rows are
turned into *raw* SPDI (0-based interbase) without any normalization; NCBI
canonicalizes them in the fetch step.

Reference status comes only from PharmVar's ``REFERENCE`` marker: ``*1`` is
not always the reference sequence (GRCh38 carries CYP2C19*38, for example).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from modephere.core import DataError
from modephere.kg.provenance import Dataset
from modephere.pgx.provenance import sha256_file
from modephere.pgx.sources.definitions import DefinitionRow, HaplotypeDefinition
from modephere.pgx.variant import format_spdi, is_grch38_chromosome

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_NAME, _GENE, _RSID, _SEQ = "Haplotype Name", "Gene", "rsID", "ReferenceSequence"
_START, _STOP, _REF, _ALT, _TYPE = (
    "Variant Start",
    "Variant Stop",
    "Reference Allele",
    "Variant Allele",
    "Type",
)
_REFERENCE = "REFERENCE"
_ABSENT = {"-", ""}
_VERSION = re.compile(r"^#version=(\S+)")
LICENSE = "PharmVar terms of use (attribution required)"


def _row_spdi(row: dict[str, str]) -> str:
    seq = row[_SEQ].strip()
    start = int(row[_START])
    ref = row[_REF].strip()
    alt = row[_ALT].strip()
    deleted = "" if ref in _ABSENT else ref.upper()
    inserted = "" if alt in _ABSENT else alt.upper()
    # Insertion: between the flanking 1-based positions start and start+1,
    # i.e. 0-based interbase position `start`. Everything else starts at start-1.
    pos0 = start if not deleted else start - 1
    return format_spdi(seq, pos0, deleted, inserted)


def load_pharmvar(tsv: Path, dataset_id: str) -> list[HaplotypeDefinition]:
    frame = pl.read_csv(tsv, separator="\t", comment_prefix="#", infer_schema=False)
    if _NAME not in frame.columns:
        msg = f"PharmVar TSV missing {_NAME!r} column: {tsv}"
        raise DataError(msg)

    rows_of: dict[str, list[DefinitionRow]] = {}
    gene_of: dict[str, str] = {}
    reference: set[str] = set()
    broken: set[str] = set()
    off_assembly: set[str] = set()
    for row in frame.iter_rows(named=True):
        name = (row.get(_NAME) or "").strip()
        if not name:
            continue
        gene_of.setdefault(name, (row.get(_GENE) or name.split("*")[0]).strip())
        rows_of.setdefault(name, [])
        if (row.get(_SEQ) or "").strip() == _REFERENCE:
            reference.add(name)
            continue
        evidence = "\t".join(str(row.get(c) or "") for c in frame.columns)
        if not is_grch38_chromosome((row.get(_SEQ) or "").strip()):
            off_assembly.add(name)  # e.g. the legacy CYP2D6 M33388 coordinates
            continue
        try:
            spdi = _row_spdi(row)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("PharmVar %s: unusable row (%s): %s", name, e, evidence)
            broken.add(name)
            continue
        rows_of[name].append(
            DefinitionRow(
                spdi_input=spdi,
                rsid=(row.get(_RSID) or "").strip() or None,
                evidence=evidence,
            )
        )

    defs = [
        HaplotypeDefinition(
            name=name,
            gene=gene_of[name],
            core_name=name.split(".", 1)[0],
            is_reference=name in reference,
            rows=() if name in off_assembly else tuple(rows),
            dataset_id=dataset_id,
            unresolved_reason=(
                "no_grch38_placement"
                if name in off_assembly
                else "ambiguous_definition"
                if name in broken
                else None
            ),
        )
        for name, rows in rows_of.items()
    ]
    logger.info("PharmVar %s: %d haplotypes", tsv.name, len(defs))
    return defs


def grch38_haplotype_files(root: Path) -> list[Path]:
    """PharmVar ``<GENE>.<ACCESSION>.haplotypes.tsv`` files on GRCh38 chromosomes.

    PharmVar also ships some genes on legacy gene sequences (CYP2D6 on
    GenBank M33388); those coordinates are not genomic and are skipped.
    """
    selected = []
    for tsv in sorted(root.rglob("*.haplotypes.tsv")):
        parts = tsv.name.split(".")
        accession = ".".join(parts[1:-2])
        if is_grch38_chromosome(accession):
            selected.append(tsv)
        else:
            logger.info("PharmVar: skipping non-GRCh38 file %s (%s)", tsv, accession)
    return selected


def pharmvar_version(tsv: Path) -> str:
    with tsv.open() as f:
        for line in f:
            if not line.startswith("#"):
                break
            if match := _VERSION.match(line):
                return match.group(1)
    msg = f"PharmVar TSV has no '#version=' header: {tsv}"
    raise DataError(msg)


def pharmvar_dataset(tsv: Path, retrieved_at: datetime) -> Dataset:
    gene = tsv.name.split(".", 1)[0]
    version = pharmvar_version(tsv)
    return Dataset(
        id=f"pharmvar:{gene}@{version}#{sha256_file(tsv)[:12]}",
        name=f"PharmVar {gene} haplotypes",
        version=version,
        uri=f"https://www.pharmvar.org/gene/{gene}",
        retrieved_at=retrieved_at,
        sha256=sha256_file(tsv),
        license=LICENSE,
    )


__all__ = [
    "grch38_haplotype_files",
    "load_pharmvar",
    "pharmvar_dataset",
    "pharmvar_version",
]
