"""Readers for the raw inputs of the knowledge graph.

* PharmGKB variant-annotation TSVs → :class:`PGxAnnotation`.
* Drug name → PubChem CID map, so KG drugs join the training rows (which key
  drugs by CID) and inference inputs (which only carry CIDs).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import polars as pl
from pydantic import ValidationError

from modephere.core import DataError
from modephere.pgx.annotation import (
    AnnotationSource,
    PGxAnnotation,
    normalize_drug_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)


def iter_annotations(path: Path, source: AnnotationSource) -> Iterator[PGxAnnotation]:
    """Yield validated annotations from one PharmGKB ``var_*_ann.tsv`` file.

    Rows that fail validation are logged and skipped, never silently coerced.
    """
    if not path.exists():
        msg = f"PharmGKB annotation file not found: {path}"
        raise DataError(msg)
    # PharmGKB TSVs embed literal double quotes inside fields; disable quoting.
    df = pl.read_csv(path, separator="\t", infer_schema=False, quote_char=None)
    skipped = 0
    for row in df.iter_rows(named=True):
        try:
            yield PGxAnnotation.from_pharmgkb_row(row, source)
        except ValidationError as e:
            skipped += 1
            logger.warning(
                "Skipping annotation %s from %s: %s",
                row.get("Variant Annotation ID"),
                path.name,
                e.errors()[0]["msg"],
            )
    logger.info(
        "Read %d annotations from %s (%d skipped)", len(df) - skipped, path, skipped
    )


NAME_CACHE_DATASET = "local:drug_name_cache"
ALIGNED_TSV_DATASET = "local:train_data_drug_alignment"


def load_drug_cid_provenance(
    name_cache: Path | None = None,
    aligned_tsv: Path | None = None,
) -> dict[str, tuple[int, str]]:
    """Normalized drug name → ``(PubChem CID, dataset id that supplied it)``.

    Sources, later ones winning on conflict:

    * ``name_cache`` — JSON ``{name: cid}`` (``data/dicts/drug_name_cache.json``;
      keys use ``_`` for spaces) → ``NAME_CACHE_DATASET``.
    * ``aligned_tsv`` — a training TSV with ``|``-aligned ``drugs`` and
      ``drugs_cid`` columns; only rows where both lists have equal length count
      → ``ALIGNED_TSV_DATASET``.
    """
    mapping: dict[str, tuple[int, str]] = {}
    if name_cache is not None and name_cache.exists():
        for name, cid in json.loads(name_cache.read_text()).items():
            if cid:
                mapping[normalize_drug_name(name.replace("_", " "))] = (
                    int(cid),
                    NAME_CACHE_DATASET,
                )
    if aligned_tsv is not None and aligned_tsv.exists():
        df = pl.read_csv(
            aligned_tsv,
            separator="\t",
            infer_schema=False,
            quote_char=None,
            columns=["drugs", "drugs_cid"],
        )
        # `columns=` keeps file order; select to pin (name, cid) order.
        df = df.select("drugs", "drugs_cid").drop_nulls()
        for names, cids in df.iter_rows():
            name_list, cid_list = names.split("|"), cids.split("|")
            if len(name_list) != len(cid_list):
                continue
            for name, cid in zip(name_list, cid_list, strict=True):
                if cid.strip().isdigit():
                    mapping[normalize_drug_name(name)] = (int(cid), ALIGNED_TSV_DATASET)
    logger.info("Drug CID map: %d names", len(mapping))
    return mapping


def load_drug_cid_map(
    name_cache: Path | None = None,
    aligned_tsv: Path | None = None,
) -> dict[str, int]:
    """Normalized drug name → PubChem CID (see :func:`load_drug_cid_provenance`)."""
    return {
        n: cid
        for n, (cid, _src) in load_drug_cid_provenance(name_cache, aligned_tsv).items()
    }


__all__ = [
    "ALIGNED_TSV_DATASET",
    "NAME_CACHE_DATASET",
    "iter_annotations",
    "load_drug_cid_map",
    "load_drug_cid_provenance",
]
