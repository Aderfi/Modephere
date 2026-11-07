"""dbSNP ESummary record → :class:`RefSNPRecord`.

An rsID names a *locus*: dbSNP lists every alternative allele observed at
it, and all of them are kept (rs3892097 has C>A, C>G and C>T). Which one an
annotation is about is decided elsewhere, from the annotation's own alleles
(:mod:`modephere.pgx.allele_inference`).

dbSNP publishes SPDI already normalized (fully justified), so the record's
GRCh38 SPDIs are used as canonical keys. SNV HGVS is formatted locally;
indel HGVS needs NCBI's ``/spdi/{spdi}/hgvs`` and is left in
``pending_indel_spdis`` for the fetch step.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from modephere.library.ingest.raw_cache import CacheEntry
from modephere.pgx.variant import (
    CanonicalVariant,
    is_grch38_chromosome,
    parse_spdi,
    snv_hgvs,
)

logger = logging.getLogger(__name__)


class DbSnpGeneRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    entrez_id: int = Field(..., ge=1)


class RefSNPRecord(BaseModel):
    """One dbSNP RefSNP as reported by a pinned dbSNP build."""

    model_config = ConfigDict(frozen=True)

    requested_rsid: str
    current_rsid: str
    dbsnp_build: str
    grch38_spdis: tuple[str, ...] = Field(
        ..., description="Every non-reference GRCh38 allele, as SPDI."
    )
    genes: tuple[DbSnpGeneRef, ...] = ()
    snv_variants: tuple[CanonicalVariant, ...] = ()
    pending_indel_spdis: tuple[str, ...] = ()

    @property
    def merged(self) -> bool:
        return self.requested_rsid != self.current_rsid


class Unresolved(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str
    detail: str


def parse_esummary(
    entry: CacheEntry, requested_rsid: str, build: str
) -> RefSNPRecord | Unresolved:
    record = entry.response
    if entry.status != "ok" or not isinstance(record, dict) or "error" in record:
        return Unresolved(reason="rsid_not_found", detail=requested_rsid)

    spdis: list[str] = []
    for raw in str(record.get("spdi") or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            seq_id, _pos, deleted, inserted = parse_spdi(raw)
        except ValueError:
            logger.warning("%s: skipping unparseable SPDI %r", requested_rsid, raw)
            continue
        if is_grch38_chromosome(seq_id) and deleted != inserted:
            spdis.append(raw)
    if not spdis:
        return Unresolved(reason="no_grch38_placement", detail=requested_rsid)

    snvs: list[CanonicalVariant] = []
    pending: list[str] = []
    for spdi in dict.fromkeys(spdis):
        seq_id, pos0, deleted, inserted = parse_spdi(spdi)
        if len(deleted) == 1 and len(inserted) == 1:
            snvs.append(
                CanonicalVariant.from_spdi(
                    seq_id,
                    pos0,
                    deleted,
                    inserted,
                    snv_hgvs(seq_id, pos0 + 1, deleted, inserted),
                )
            )
        else:
            pending.append(spdi)

    genes = tuple(
        DbSnpGeneRef(symbol=g["name"], entrez_id=int(g["gene_id"]))
        for g in record.get("genes") or []
        if g.get("name") and str(g.get("gene_id", "")).isdigit()
    )
    return RefSNPRecord(
        requested_rsid=requested_rsid,
        current_rsid=f"rs{record['snp_id']}",
        dbsnp_build=build,
        grch38_spdis=tuple(dict.fromkeys(spdis)),
        genes=genes,
        snv_variants=tuple(snvs),
        pending_indel_spdis=tuple(pending),
    )


__all__ = ["DbSnpGeneRef", "RefSNPRecord", "Unresolved", "parse_esummary"]
