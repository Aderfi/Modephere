"""Resolution steps shared by ``fetch`` (online client) and ``build`` (offline).

Running the *same* functions in both stages guarantees the fetch step caches
exactly the responses the offline build later reads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from modephere.pgx.cache import CacheMissError
from modephere.pgx.sources.dbsnp import RefSNPRecord, Unresolved, parse_esummary
from modephere.pgx.variant import (
    CanonicalVariant,
    format_spdi,
    is_grch38_chromosome,
    parse_spdi,
    snv_hgvs,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from modephere.pgx.ncbi import NCBIClient
    from modephere.pgx.sources.definitions import DefinitionRow

logger = logging.getLogger(__name__)


class Canonicalized(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: CanonicalVariant | None
    warnings: tuple[str, ...] = ()
    failure: str | None = None


class ResolvedRefSNP(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: RefSNPRecord
    variants: tuple[CanonicalVariant, ...]


def _spdi_from(data: dict) -> str:
    return format_spdi(
        data["seq_id"],
        int(data["position"]),
        data["deleted_sequence"],
        data["inserted_sequence"],
    )


def _hgvs_for(spdi: str, client: NCBIClient) -> str | None:
    seq_id, pos0, deleted, inserted = parse_spdi(spdi)
    if len(deleted) == 1 and len(inserted) == 1:
        return snv_hgvs(seq_id, pos0 + 1, deleted, inserted)
    entry = client.spdi_hgvs(spdi)
    if entry.status != "ok" or not isinstance(entry.response, dict):
        return None
    return entry.response.get("hgvs")


class _CanonicalizationFailed(Exception):
    pass


def _canonical_spdi(row: DefinitionRow, client: NCBIClient, warnings: list[str]) -> str:
    """Source row → NCBI canonical SPDI; raises _CanonicalizationFailed."""
    if row.hgvs_input is not None:
        ctx = client.hgvs_contextuals(row.hgvs_input)
        warnings += ctx.warnings
        spdis = (ctx.response or {}).get("spdis", []) if ctx.status == "ok" else []
        if len(spdis) != 1:
            raise _CanonicalizationFailed(f"hgvs_contextuals:{ctx.status}:{len(spdis)}")
        raw = _spdi_from(spdis[0])
    else:
        assert row.spdi_input is not None
        raw = row.spdi_input
    canon = client.spdi_canonical(raw)
    warnings += canon.warnings
    if canon.status != "ok" or not isinstance(canon.response, dict):
        raise _CanonicalizationFailed(f"spdi_canonical:{canon.status}")
    return _spdi_from(canon.response)


def canonicalize_row(row: DefinitionRow, client: NCBIClient) -> Canonicalized:
    """Source SPDI/HGVS → NCBI canonical SPDI + HGVS (raises CacheMissError offline)."""
    warnings: list[str] = []
    try:
        spdi = _canonical_spdi(row, client, warnings)
        seq_id, pos0, deleted, inserted = parse_spdi(spdi)
        if not is_grch38_chromosome(seq_id):
            raise _CanonicalizationFailed(f"not_grch38:{seq_id}")
        hgvs = _hgvs_for(spdi, client)
        if hgvs is None:
            raise _CanonicalizationFailed("spdi_hgvs:error")
    except _CanonicalizationFailed as e:
        return Canonicalized(variant=None, warnings=tuple(warnings), failure=str(e))
    return Canonicalized(
        variant=CanonicalVariant.from_spdi(seq_id, pos0, deleted, inserted, hgvs),
        warnings=tuple(warnings),
    )


def resolve_refsnps(
    rsids: Iterable[int],
    client: NCBIClient,
    on_miss: Callable[[CacheMissError], None] | None = None,
) -> dict[int, ResolvedRefSNP | Unresolved]:
    """dbSNP records for ``rsids``, with indel HGVS resolved through NCBI."""
    ids = sorted(set(rsids))
    if not ids:
        return {}
    entries = client.esummary_snp(ids)
    build = client.dbsnp_build()
    out: dict[int, ResolvedRefSNP | Unresolved] = {}
    for i in ids:
        # Each record keeps the dbSNP build it was actually fetched under.
        parsed = parse_esummary(
            entries[i], f"rs{i}", entries[i].source_version or build
        )
        if isinstance(parsed, Unresolved):
            out[i] = parsed
            continue
        variants = list(parsed.snv_variants)
        for spdi in parsed.pending_indel_spdis:
            try:
                hgvs = _hgvs_for(spdi, client)
            except CacheMissError as e:
                if on_miss is None:
                    raise
                on_miss(e)
                continue
            if hgvs is None:
                logger.warning(
                    "rs%d: no HGVS for indel allele %s; allele skipped", i, spdi
                )
                continue
            seq_id, pos0, deleted, inserted = parse_spdi(spdi)
            variants.append(
                CanonicalVariant.from_spdi(seq_id, pos0, deleted, inserted, hgvs)
            )
        out[i] = ResolvedRefSNP(record=parsed, variants=tuple(variants))
    return out


__all__ = ["Canonicalized", "ResolvedRefSNP", "canonicalize_row", "resolve_refsnps"]
