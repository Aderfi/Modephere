"""Tests for parsing dbSNP ESummary records into RefSNP records."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from modephere.library.ingest.raw_cache import CacheEntry
from modephere.pgx.sources.dbsnp import (
    DbSnpGeneRef,
    RefSNPRecord,
    Unresolved,
    parse_esummary,
    protein_by_spdi,
)

from .conftest import FIXTURES

_RESULT = json.loads((FIXTURES / "esummary_snp.json").read_text())["result"]
BUILD = "Build250306-1408.1"


def _entry(record: dict | None, status: str = "ok") -> CacheEntry:
    return CacheEntry(
        key="dbsnp:rsX",
        request="u",
        status=status,
        retrieved_at=datetime.now(UTC),
        response=record,
        source_version=BUILD,
    )


def entry_for(rsid: int) -> CacheEntry:
    return _entry(_RESULT[str(rsid)])


def test_multiallelic_site_keeps_every_alt():
    r = parse_esummary(entry_for(3892097), "rs3892097", BUILD)
    assert isinstance(r, RefSNPRecord)
    assert {v.spdi for v in r.snv_variants} == {
        "NC_000022.11:42128944:C:A",
        "NC_000022.11:42128944:C:G",
        "NC_000022.11:42128944:C:T",
    }
    assert {v.hgvs_g for v in r.snv_variants} >= {"NC_000022.11:g.42128945C>T"}
    assert r.genes == (DbSnpGeneRef(symbol="CYP2D6", entrez_id=1565),)
    assert r.dbsnp_build == BUILD
    assert not r.merged


def _ok(r: RefSNPRecord | Unresolved) -> RefSNPRecord:
    assert isinstance(r, RefSNPRecord)
    return r


def test_merge_followed():
    r = _ok(parse_esummary(entry_for(17879456), "rs17879456", BUILD))
    assert r.requested_rsid == "rs17879456"
    assert r.current_rsid == "rs4244285"
    assert r.merged


def test_grch37_and_alt_contigs_ignored():
    r = _ok(parse_esummary(entry_for(4244285), "rs4244285", BUILD))
    assert r.snv_variants
    assert all(v.seq_id == "NC_000010.11" for v in r.snv_variants)


def test_not_found():
    out = parse_esummary(
        _entry({"error": "cannot get document summary"}, "not_found"),
        "rs999999999999",
        BUILD,
    )
    assert out == Unresolved(reason="rsid_not_found", detail="rs999999999999")


def test_only_grch37_placement():
    out = parse_esummary(
        _entry({"snp_id": 1, "spdi": "NC_000010.10:1:G:A", "genes": []}), "rs1", BUILD
    )
    assert isinstance(out, Unresolved)
    assert out.reason == "no_grch38_placement"


def test_indel_goes_to_pending_and_reference_allele_skipped():
    record = {
        "snp_id": 2,
        "spdi": "NC_000006.12:18149103:AAGT:,NC_000006.12:18149103:AAGT:AAGT",
        "genes": [{"name": "TPMT", "gene_id": "7172"}],
    }
    r = _ok(parse_esummary(_entry(record), "rs2", BUILD))
    assert r.snv_variants == ()
    assert r.pending_indel_spdis == ("NC_000006.12:18149103:AAGT:",)
    assert r.grch38_spdis == ("NC_000006.12:18149103:AAGT:",)


def test_unparseable_spdi_skipped_not_fatal():
    record = {"snp_id": 3, "spdi": "NC_000010.11:5:R:A,NC_000010.11:5:G:A", "genes": []}
    r = _ok(parse_esummary(_entry(record), "rs3", BUILD))
    assert r.grch38_spdis == ("NC_000010.11:5:G:A",)


def test_each_allele_gets_its_own_protein_change():
    # rs1065852: G>A is CYP2D6 P34S (*10), G>C is P34A
    r = _ok(parse_esummary(entry_for(1065852), "rs1065852", BUILD))
    assert r.protein_hgvs == {
        "NC_000022.11:42130691:G:A": "NP_000097.3:p.Pro34Ser",
        "NC_000022.11:42130691:G:C": "NP_000097.3:p.Pro34Ala",
    }
    assert r.so_terms == ("missense_variant", "coding_sequence_variant")


def test_non_coding_site_has_terms_but_no_protein_change():
    r = _ok(parse_esummary(entry_for(3892097), "rs3892097", BUILD))
    assert r.protein_hgvs == {}
    assert "splice_acceptor_variant" in r.so_terms


def test_protein_accession_with_another_allele_count_is_not_used():
    spdis = ["NC_1:9:A:C", "NC_1:9:A:G", "NC_1:9:A:T"]
    docsum = (
        "HGVS=NC_1:g.10A>C,NC_1:g.10A>G,NC_1:g.10A>T,"
        "NP_1.1:p.Lys4Asn,NP_1.1:p.Lys4Glu,"  # 2 changes for 3 alleles: ambiguous
        "NP_2.1:p.Lys9Gln,NP_2.1:p.Lys9Arg,NP_2.1:p.Lys9Ter|SEQ=[A/C/G/T]"
    )
    assert protein_by_spdi(docsum, spdis) == {
        "NC_1:9:A:C": "NP_2.1:p.Lys9Gln",
        "NC_1:9:A:G": "NP_2.1:p.Lys9Arg",
        "NC_1:9:A:T": "NP_2.1:p.Lys9Ter",
    }
    assert protein_by_spdi("HGVS=NP_1.1:p.Lys4Asn|SEQ=[A/C]", spdis[:2]) == {}
    assert protein_by_spdi(None, spdis) == {}
