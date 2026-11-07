"""ClinPGx (PharmGKB) allele-definition tables → :class:`HaplotypeDefinition`.

These are the tables CPIC guidelines and PharmCAT use for genes PharmVar
does not cover (TPMT, UGT1A1, …). Layout, one column per defining position:

* a row whose first cell starts ``Position at NC_…`` holds that position's
  GRCh38 HGVS ``g.`` — several alternatives ``;``-separated;
* a row whose first cell ends with ``Allele`` heads the allele rows; the
  first allele row is the reference allele.

Allele cells are on the forward strand: a base (``C``), ``delAAGT``,
``insT``, or a repeat count ``TA(8)``. A cell selects the column alternative
it matches; anything else (IUPAC codes, no match) is never guessed — the
allele becomes ``unresolved`` with reason ``ambiguous_definition``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import openpyxl

from modephere.core import DataError
from modephere.pgx.sources.definitions import DefinitionRow, HaplotypeDefinition

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_POSITION_HEADER = re.compile(r"^Position at (NC_\d+\.\d+)")
_SNV_ALT = re.compile(r">([ACGT])$")
_REPEAT_CELL = re.compile(r"^([ACGT]+)\((\d+)\)$")
LICENSE = "CC BY-SA 4.0"


def match_alternative(cell: str, alternatives: list[str]) -> str | None:
    """The column alternative a cell denotes, or None when it is not explicit."""
    cell = cell.strip()
    if re.fullmatch(r"[ACGT]", cell):
        hits = [
            a for a in alternatives if (m := _SNV_ALT.search(a)) and m.group(1) == cell
        ]
    elif cell.startswith("del"):
        hits = [
            a for a in alternatives if a.endswith("del") or re.search(r"del[ACGT]*$", a)
        ]
    elif cell.startswith("ins"):
        hits = [a for a in alternatives if a.endswith(cell)]
    elif m := _REPEAT_CELL.match(cell):
        hits = [a for a in alternatives if a.endswith(f"{m.group(1)}[{m.group(2)}]")]
    else:
        hits = []
    return hits[0] if len(hits) == 1 else None


def _text(value: object) -> str:
    return "" if value is None else " ".join(str(value).split())


def load_allele_table(xlsx: Path, dataset_id: str) -> list[HaplotypeDefinition]:
    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        rows = [
            [_text(c) for c in row]
            for row in workbook.worksheets[0].iter_rows(values_only=True)
        ]
    finally:
        workbook.close()

    gene = next(
        (r[0].split(":", 1)[1].strip() for r in rows if r and r[0].startswith("GENE:")),
        None,
    )
    header = next((r for r in rows if r and _POSITION_HEADER.match(r[0])), None)
    start = next(
        (i for i, r in enumerate(rows) if r and r[0].endswith(" Allele")), None
    )
    if gene is None or header is None or start is None:
        msg = f"unrecognized allele-definition table layout: {xlsx}"
        raise DataError(msg)
    accession = _POSITION_HEADER.match(header[0]).group(1)  # type: ignore[union-attr]
    columns = {
        j: [a.strip() for a in cell.split(";") if a.strip()]
        for j, cell in enumerate(header)
        if j > 0 and cell
    }

    allele_rows = [r for r in rows[start + 1 :] if r and r[0]]
    reference = allele_rows[0]
    defs: list[HaplotypeDefinition] = []
    for k, row in enumerate(allele_rows):
        label = row[0]
        name = f"{gene}{label}"
        is_reference = k == 0
        def_rows: list[DefinitionRow] = []
        ambiguous = False
        if not is_reference:
            for j, alternatives in columns.items():
                cell = row[j] if j < len(row) else ""
                if not cell or cell == (reference[j] if j < len(reference) else ""):
                    continue
                chosen = match_alternative(cell, alternatives)
                if chosen is None:
                    logger.warning(
                        "%s: cell %r matches none of %s", name, cell, alternatives
                    )
                    ambiguous = True
                    continue
                def_rows.append(
                    DefinitionRow(
                        hgvs_input=f"{accession}:{chosen}",
                        evidence=f"{xlsx.name}\t{name}\tcolumn={header[j]}\tcell={cell}",
                    )
                )
        defs.append(
            HaplotypeDefinition(
                name=name,
                gene=gene,
                core_name=name,
                is_reference=is_reference,
                rows=tuple(def_rows),
                dataset_id=dataset_id,
                unresolved_reason="ambiguous_definition" if ambiguous else None,
            )
        )
    logger.info("Allele table %s: %d alleles (%s)", xlsx.name, len(defs), accession)
    return defs


__all__ = ["LICENSE", "load_allele_table", "match_alternative"]
