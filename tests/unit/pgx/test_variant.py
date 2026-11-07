"""Tests for canonical variant identity (SPDI on GRCh38)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modephere.pgx.variant import (
    CanonicalVariant,
    format_spdi,
    is_grch38_chromosome,
    parse_spdi,
    snv_hgvs,
)


def test_grch38_accessions():
    assert is_grch38_chromosome("NC_000010.11")
    assert is_grch38_chromosome("NC_000006.12")
    assert is_grch38_chromosome("NC_012920.1")  # chrM
    assert not is_grch38_chromosome("NC_000010.10")  # GRCh37
    assert not is_grch38_chromosome("NG_008384.3")
    assert not is_grch38_chromosome("NW_015148968.1")


def test_spdi_round_trip_and_one_based_pos():
    v = CanonicalVariant.from_spdi(
        "NC_000010.11", 94781858, "G", "A", "NC_000010.11:g.94781859G>A"
    )
    assert v.spdi == "NC_000010.11:94781858:G:A"
    assert v.pos == 94781859
    assert v.is_snv
    assert v.vrs_id is None
    assert v.assembly == "GRCh38"
    assert parse_spdi(v.spdi) == ("NC_000010.11", 94781858, "G", "A")


def test_deletion_spdi_has_empty_inserted():
    assert (
        format_spdi("NC_000006.12", 18149103, "AAGT", "")
        == "NC_000006.12:18149103:AAGT:"
    )
    assert parse_spdi("NC_000006.12:18149103:AAGT:") == (
        "NC_000006.12",
        18149103,
        "AAGT",
        "",
    )
    v = CanonicalVariant.from_spdi(
        "NC_000006.12", 18149103, "AAGT", "", "NC_000006.12:g.18149104_18149107del"
    )
    assert not v.is_snv


def test_snv_hgvs():
    assert snv_hgvs("NC_000022.11", 42128945, "C", "T") == "NC_000022.11:g.42128945C>T"


def test_non_grch38_rejected():
    with pytest.raises(ValidationError):
        CanonicalVariant.from_spdi("NC_000010.10", 1, "G", "A", "x")


def test_malformed_spdi_rejected():
    with pytest.raises(ValueError, match="SPDI"):
        parse_spdi("NC_000010.11:abc:G:A")


GRCH38_P14 = (
    "NC_000001.11 NC_000002.12 NC_000003.12 NC_000004.12 NC_000005.10 NC_000006.12 "
    "NC_000007.14 NC_000008.11 NC_000009.12 NC_000010.11 NC_000011.10 NC_000012.12 "
    "NC_000013.11 NC_000014.9 NC_000015.10 NC_000016.10 NC_000017.11 NC_000018.10 "
    "NC_000019.10 NC_000020.11 NC_000021.9 NC_000022.11 NC_000023.11 NC_000024.10 "
    "NC_012920.1"
).split()


@pytest.mark.parametrize("accession", GRCH38_P14)
def test_every_grch38_p14_chromosome_accepted(accession):
    # Verified 2026-10-18 against NCBI nuccore titles "GRCh38.p14 Primary Assembly".
    assert is_grch38_chromosome(accession)


@pytest.mark.parametrize(
    "accession",
    ["NC_000019.9", "NC_000014.8", "NC_000021.8", "NC_000005.9", "NC_000019.11"],
)
def test_grch37_or_unknown_versions_rejected(accession):
    assert not is_grch38_chromosome(accession)
