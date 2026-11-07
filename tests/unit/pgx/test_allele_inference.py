"""Tests for linking an annotation to one specific alt allele of an rsID."""

from __future__ import annotations

from modephere.pgx.allele_inference import METHOD, Inference, infer_variant
from modephere.pgx.variant import CanonicalVariant

SITE = [
    CanonicalVariant.from_spdi(
        "NC_000022.11", 42128944, "C", a, f"NC_000022.11:g.42128945C>{a}"
    )
    for a in "AGT"
]


def _alt(result: Inference) -> str:
    assert result.variant is not None
    return result.variant.alt


def test_method_name():
    assert METHOD == "annotation_alleles_forward_strand"


def test_rs3892097_genotypes_pick_t():
    r = infer_variant("CT", "CC", SITE)
    assert r.variant is not None
    assert r.variant.spdi == "NC_000022.11:42128944:C:T"
    assert r.reason is None
    assert r.evidence == "Alleles='CT'; Comparison='CC'"


def test_allele_form():
    assert _alt(infer_variant("T", "C", SITE)) == "T"


def test_combined_genotypes():
    assert _alt(infer_variant("CT + TT", "CC", SITE)) == "T"


BIALLELIC = [
    CanonicalVariant.from_spdi(
        "NC_000010.11", 94781858, "G", "A", "NC_000010.11:g.94781859G>A"
    )
]
PALINDROMIC = [
    CanonicalVariant.from_spdi("NC_000001.11", 99, "C", "G", "NC_000001.11:g.100C>G")
]


def test_reverse_strand_is_mismatch():
    # rs4244285 is G>A on GRCh38; "CT"/"CC" is the reverse-strand report.
    r = infer_variant("CT", "CC", BIALLELIC)
    assert r.variant is None
    assert r.reason == "allele_strand_mismatch"


def test_forward_strand_biallelic():
    assert _alt(infer_variant("GA", "GG", BIALLELIC)) == "A"


def test_palindromic_homozygous_is_strand_ambiguous():
    # "GG" read forward is hom-alt; reverse-complemented it is hom-ref.
    r = infer_variant("GG", None, PALINDROMIC)
    assert r.variant is None
    assert r.reason == "strand_ambiguous"


def test_palindromic_het_identity_is_strand_invariant():
    # {C, G} complements to itself: the implicated variant is C>G either way.
    assert _alt(infer_variant("CG", "CC", PALINDROMIC)) == "G"
    assert _alt(infer_variant("CG", None, PALINDROMIC)) == "G"


def test_multiallelic_site_where_complement_is_inconsistent():
    # CT/CC at C>A/G/T: the complement {G, A} would imply two alts -> unique answer T.
    assert _alt(infer_variant("CT", "CC", SITE)) == "T"


def test_text_alleles_not_inferred():
    assert infer_variant("del", None, SITE).reason == "not_nucleotide_alleles"
    assert infer_variant("*1/*2", None, SITE).reason == "not_nucleotide_alleles"


def test_missing_alleles_not_inferred():
    assert infer_variant(None, None, SITE).reason == "not_nucleotide_alleles"


def test_only_reference_observed():
    assert infer_variant("CC", None, SITE).reason == "no_alt_observed"


def test_two_alts():
    assert infer_variant("AT", "CC", SITE).reason == "multiple_alts_observed"


def test_indel_site_not_inferred():
    site = [
        CanonicalVariant.from_spdi(
            "NC_000006.12", 18149103, "AAGT", "", "NC_000006.12:g.18149104_18149107del"
        )
    ]
    assert infer_variant("A", None, site).reason == "not_snv_site"


def test_sites_at_different_positions_not_inferred():
    other = CanonicalVariant.from_spdi(
        "NC_000022.11", 42128999, "C", "T", "NC_000022.11:g.42129000C>T"
    )
    assert infer_variant("CT", None, [*SITE, other]).reason == "not_snv_site"
