"""Where a gene sits, per assembly, from its NCBI Gene record.

Whole-gene deletion alleles (CYP2D6*5, GSTM1 null) are written in HGVS over
the gene span with uncertain breakpoints, ``NC_000022.11:g.(?_42126499)_
(42130810_?)del``: the deletion covers at least the gene, its exact ends are
not known. Coordinates come from the NCBI Gene record (``genomicinfo`` = the
current GRCh38 annotation; ``locationhist`` also lists T2T-CHM13). A gene
that GRCh38 places only on an alternate locus (GSTT1) has no GRCh38 span.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from modephere.pgx.variant import is_grch38_chromosome

if TYPE_CHECKING:
    from modephere.library.ingest.raw_cache import CacheEntry

GRCH38_ASSEMBLY = "GCF_000001405"  # any GRCh38 patch (.40 = p14)
T2T_ASSEMBLY = "GCF_009914755"  # T2T-CHM13v2.0


class Span(BaseModel):
    """A 1-based, inclusive span on a chromosome accession."""

    model_config = ConfigDict(frozen=True)

    seq_id: str
    start: int
    end: int

    @property
    def deletion(self) -> str:
        """HGVS change for a deletion of at least this span."""
        return f"(?_{self.start})_({self.end}_?)del"


class GeneLocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    gene_id: int
    grch38: Span | None = None
    t2t: Span | None = None


def _span(seq_id: str, start0: int, stop0: int) -> Span:
    lo, hi = sorted((start0, stop0))  # minus-strand genes list start > stop
    return Span(seq_id=seq_id, start=lo + 1, end=hi + 1)


def gene_location(entry: CacheEntry, symbol: str) -> GeneLocation | None:
    """Placements of ``symbol``; None when NCBI has no such human gene."""
    if entry.status != "ok" or not isinstance(entry.response, dict):
        return None
    record = entry.response["record"]
    grch38 = next(
        (
            _span(g["chraccver"], int(g["chrstart"]), int(g["chrstop"]))
            for g in record.get("genomicinfo", [])
            if is_grch38_chromosome(g.get("chraccver", ""))
        ),
        None,
    )
    t2t = next(
        (
            _span(h["chraccver"], int(h["chrstart"]), int(h["chrstop"]))
            for h in record.get("locationhist", [])
            if str(h.get("assemblyaccver", "")).startswith(T2T_ASSEMBLY)
            and str(h.get("chraccver", "")).startswith("NC_")
        ),
        None,
    )
    return GeneLocation(
        symbol=symbol, gene_id=int(entry.response["gene_id"]), grch38=grch38, t2t=t2t
    )


__all__ = ["GeneLocation", "Span", "gene_location"]
