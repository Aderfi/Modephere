"""Parsing and resolving the genotypes PharmGKB annotations compare."""

from __future__ import annotations

import pytest

from modephere.pgx.genotype import (
    Copy,
    Genotype,
    Locus,
    ParsedGenotype,
    annotation_locus,
    parse_field,
    read_indel_site,
    resolve_indel,
    resolve_null,
    resolve_snv,
    resolve_star,
    strand_ambiguous,
    unresolved_site,
)
from modephere.pgx.labels import classify_label
from modephere.pgx.variant import CanonicalVariant

CYP2C9 = Locus(kind="star", name="CYP2C9")
CYP2D6 = Locus(kind="star", name="CYP2D6")
HLA_B = Locus(kind="star", name="HLA-B")
GSTM1 = Locus(kind="null", name="GSTM1")
SITE = Locus(kind="rsid", name="rs1799853", rsid=1799853)
SEQ = "NC_000010.11"
DEL_2D6 = "(?_42126499)_(42130810_?)del"
HAPLOTYPES = {
    "CYP2C9*1": (SEQ, ()),
    "CYP2C9*2": (SEQ, ("94942290C>T",)),
    "CYP2C9*3": (SEQ, ("94981296A>C",)),
    "CYP2D6*1": ("NC_000022.11", ()),
    "CYP2D6*2": ("NC_000022.11", ("42127941G>A",)),
    "CYP2D6*4": ("NC_000022.11", ("42128945C>T",)),
    "CYP2D6*5": ("NC_000022.11", (DEL_2D6,)),
    "GSTM1*0": ("NC_000001.11", ("(?_109687817)_(109693745_?)del",)),
}
RESOLVED = {*HAPLOTYPES, "HLA-B*58:01"}


def _snv(pos: int, ref: str, alt: str) -> CanonicalVariant:
    return CanonicalVariant.from_spdi(
        SEQ, pos - 1, ref, alt, f"{SEQ}:g.{pos}{ref}>{alt}"
    )


def _indel(ref: str, alt: str, change: str) -> CanonicalVariant:
    return CanonicalVariant.from_spdi(SEQ, 99, ref, alt, f"{SEQ}:g.{change}")


C_T = (_snv(100, "C", "T"),)
C_G = (_snv(100, "C", "G"),)
C_AGT = tuple(_snv(100, "C", b) for b in "AGT")
DELETION = (_indel("ACAG", "A", "101_103del"),)
INSERTION = (_indel("A", "ATTA", "100_101insTTA"),)
REPEAT = (
    _indel("TCCTC", "TC", "103_105del"),
    _indel("TCCTC", "TCCTCCTC", "103_105dup"),
)
HOMOPOLYMER = tuple(
    _indel("TTTT", "T" * n, change)
    for n, change in ((3, "103del"), (2, "102_103del"), (5, "103dup"))
)


def _one(text: str, locus: Locus) -> ParsedGenotype:
    parsed = parse_field(text, locus)
    assert parsed is not None and len(parsed) == 1
    return parsed[0]


def _hap_reason(name: str) -> str | None:
    return None if name in RESOLVED else "not_in_pharmvar_or_tables"


def _star(text: str, locus: Locus = CYP2C9) -> Genotype:
    return resolve_star(
        _one(text, locus),
        locus.name,
        haplotype_reason=_hap_reason,
        haplotype_hgvs=HAPLOTYPES,
    )


def _null(text: str) -> Genotype:
    return resolve_null(
        _one(text, GSTM1),
        "GSTM1",
        haplotype_reason=_hap_reason,
        haplotype_hgvs=HAPLOTYPES,
    )


def _snv_gt(text: str, site=C_T) -> Genotype:
    return resolve_snv(_one(text, SITE), "rs1799853", site)


def _read(site, *fields: str):
    parsed = [p for f in fields for p in parse_field(f, SITE) or ()]
    return parsed, read_indel_site(parsed, site)


def _indels(site, *fields: str) -> list[Genotype]:
    """Every genotype of an annotation whose fields are ``fields``."""
    parsed, reading = _read(site, *fields)
    assert not isinstance(reading, str), reading
    tokens, mapping = reading
    return [
        resolve_indel(p, t, mapping, "rs1", site)
        for p, t in zip(parsed, tokens, strict=True)
    ]


def _indel_reason(site, *fields: str) -> str:
    _parsed, reading = _read(site, *fields)
    assert isinstance(reading, str)
    return reading


# --------------------------------------------------------------------------- #
# Locus and notation
# --------------------------------------------------------------------------- #


def test_locus_is_one_gene_or_one_site() -> None:
    def locus(*labels: str):
        return annotation_locus([classify_label(lab) for lab in labels])

    assert locus("CYP2C9*1", "CYP2C9*3") == CYP2C9
    assert locus("CYP2D6*1", "CYP2D6*2xN", "CYP2D6*5") == CYP2D6
    assert locus("HLA-B*58:01") == HLA_B
    assert locus("GSTM1 null", "GSTM1 non-null") == GSTM1
    assert locus("rs1799853") == SITE
    assert locus("rs1", "rs2") == "multi_locus"
    assert locus("CYP2C9*3", "VKORC1*2") == "multi_locus"
    assert locus("CYP2C9 poor metabolizer") == "not_a_genotype_locus"
    assert locus("rs1799853", "CYP2C9*3") == "multi_locus"


def test_parse_alternatives_and_sides() -> None:
    assert parse_field("*1/*22 + *22/*22", CYP2C9) == [
        ParsedGenotype(sides=("CYP2C9*1", "CYP2C9*22"), slashed=True),
        ParsedGenotype(sides=("CYP2C9*22", "CYP2C9*22"), slashed=True),
    ]
    assert parse_field("*3", CYP2C9) == [
        ParsedGenotype(sides=("CYP2C9*3",), slashed=False)
    ]
    assert parse_field("AA + AG", SITE) == [
        ParsedGenotype(sides=("AA",), slashed=False),
        ParsedGenotype(sides=("AG",), slashed=False),
    ]
    assert parse_field("Null", GSTM1) == [
        ParsedGenotype(sides=("null",), slashed=False)
    ]
    assert parse_field(None, CYP2C9) == []


@pytest.mark.parametrize(
    ("text", "locus"),
    [
        ("HTTLPR long form (L allele)", CYP2C9),
        ("*1/*2/*3", CYP2C9),
        ("VKORC1*2/*2", CYP2C9),  # another gene
        ("(CCGCGCCACTTCGCCTGCCTCCGTCCCG)2", SITE),
        ("present", GSTM1),
    ],
)
def test_unknown_notation_is_not_parsed(text: str, locus: Locus) -> None:
    assert parse_field(text, locus) is None


# --------------------------------------------------------------------------- #
# Star loci: the pair of haplotypes
# --------------------------------------------------------------------------- #


def test_heterozygous_and_homozygous_diplotypes_differ() -> None:
    het, hom = _star("*3/*1"), _star("*3/*3")
    assert het.name == "CYP2C9*1/*3"  # sides in natural order
    assert het.copies == (
        Copy(allele="CYP2C9*1", copies=1, min_copies=1),
        Copy(allele="CYP2C9*3", copies=1, min_copies=1),
    )
    assert hom.copies == (Copy(allele="CYP2C9*3", copies=2, min_copies=2),)
    assert het.hgvs_g38 == f"{SEQ}:g.[=];[94981296A>C]"
    assert hom.hgvs_g38 == f"{SEQ}:g.[94981296A>C];[94981296A>C]"
    assert (het.gene_copies, het.zygosity, het.status) == (2, "diploid", "resolved")


def test_compound_heterozygote() -> None:
    gt = _star("*2/*3")
    assert [c.allele for c in gt.copies] == ["CYP2C9*2", "CYP2C9*3"]
    assert gt.hgvs_g38 == f"{SEQ}:g.[94942290C>T];[94981296A>C]"


def test_carrier_has_a_lower_bound_only() -> None:
    gt = _star("*3")
    assert (gt.name, gt.zygosity) == ("CYP2C9*3", "carrier")
    assert gt.copies == (Copy(allele="CYP2C9*3", copies=None, min_copies=1),)
    assert gt.hgvs_g38 == f"{SEQ}:g.[94981296A>C];[?]"
    assert gt.gene_copies is None


def test_gene_deletion_is_an_allele_with_no_gene_copy() -> None:
    gt = _star("*4/*5", CYP2D6)
    assert gt.status == "resolved"
    assert [(c.allele, c.copies) for c in gt.copies] == [
        ("CYP2D6*4", 1),
        ("CYP2D6*5", 1),
    ]
    assert (gt.gene_copies, gt.gene_copies_min) == (1, 1)
    assert gt.hgvs_g38 == f"NC_000022.11:g.[42128945C>T];[{DEL_2D6}]"
    assert _star("*5/*5", CYP2D6).gene_copies == 0


def test_duplication_counts_the_copied_haplotype() -> None:
    gt = _star("*1/*2xN", CYP2D6)
    assert gt.copies == (
        Copy(allele="CYP2D6*1", copies=1, min_copies=1),
        Copy(allele="CYP2D6*2", copies=None, min_copies=2),
    )
    assert (gt.gene_copies, gt.gene_copies_min) == (None, 3)
    assert gt.hgvs_g38 is None  # several copies on one chromosome: no diploid HGVS
    exact = _star("*1/*2x2", CYP2D6)
    assert exact.copies[1] == Copy(allele="CYP2D6*2", copies=2, min_copies=2)
    assert exact.gene_copies == 3


def test_hla_allele_is_named_not_hgvs() -> None:
    gt = _star("*58:01", HLA_B)
    assert (gt.name, gt.status, gt.hgvs_g38) == ("HLA-B*58:01", "resolved", None)
    assert gt.copies == (Copy(allele="HLA-B*58:01", copies=None, min_copies=1),)


def test_hybrid_or_unknown_allele_leaves_the_whole_genotype_unresolved() -> None:
    gt = _star("*1/*36", CYP2D6)
    assert (gt.status, gt.reason) == ("unresolved", "hybrid_allele")
    assert gt.copies == () and gt.hgvs_g38 is None  # never a partial genotype
    assert _star("*1/*99").reason == "not_in_pharmvar_or_tables"


def test_null_genotypes() -> None:
    null = _null("null")
    assert null.name == "GSTM1*0/*0"
    assert null.copies == (Copy(allele="GSTM1*0", copies=2, min_copies=2),)
    assert null.gene_copies == 0
    present = _null("non-null")
    assert (present.zygosity, present.gene_copies_min, present.copies) == (
        "carrier",
        1,
        (),
    )
    het = _null("non-null/null")
    assert het.name == "GSTM1 null/non-null"
    assert het.copies == (Copy(allele="GSTM1*0", copies=1, min_copies=1),)
    assert het.gene_copies_min == 1


# --------------------------------------------------------------------------- #
# SNV sites
# --------------------------------------------------------------------------- #


def test_snv_genotypes_carry_dosage() -> None:
    het, hom, ref = _snv_gt("CT"), _snv_gt("T/T"), _snv_gt("CC")
    assert (het.name, het.ref_copies) == ("rs1799853:C/T", 1)
    assert het.copies == (Copy(allele=f"{SEQ}:g.100C>T", copies=1, min_copies=1),)
    assert het.hgvs_g38 == f"{SEQ}:g.[100C>T];[100=]"
    assert (hom.copies[0].copies, hom.ref_copies) == (2, 0)
    assert (ref.copies, ref.ref_copies) == ((), 2)
    assert ref.hgvs_g38 == f"{SEQ}:g.[100=];[100=]"


def test_single_base_is_a_carrier() -> None:
    gt = _snv_gt("T")
    assert (gt.name, gt.zygosity, gt.ref_copies) == ("rs1799853:T", "carrier", None)
    assert gt.copies == (Copy(allele=f"{SEQ}:g.100C>T", copies=None, min_copies=1),)
    assert gt.hgvs_g38 == f"{SEQ}:g.[100C>T];[?]"


def test_base_not_at_the_site_is_a_strand_mismatch() -> None:
    gt = _snv_gt("CA")
    assert (gt.status, gt.reason) == ("unresolved", "allele_strand_mismatch")


def test_strand_ambiguity_is_judged_per_annotation() -> None:
    def parsed(*fields: str) -> list[ParsedGenotype]:
        return [p for f in fields for p in parse_field(f, SITE) or ()]

    assert strand_ambiguous(parsed("CG", "CC"), C_G)  # palindromic site
    assert not strand_ambiguous(parsed("CT", "CC"), C_T)
    # four-allele site: 'CT vs CC' flipped is 'GA vs GG' (no reference) -> fine,
    # but a lone 'T' flipped is a plausible 'A'
    assert not strand_ambiguous(parsed("CT", "CC"), C_AGT)
    assert strand_ambiguous(parsed("T"), C_AGT)


# --------------------------------------------------------------------------- #
# Indel sites, read per annotation
# --------------------------------------------------------------------------- #


def test_present_absent_notation_at_a_deletion_site() -> None:
    het, hom, ref = _indels(DELETION, "CAG/del + del/del", "CAG/CAG")
    assert het.name == "rs1:101_103del/="
    assert (het.copies[0].copies, het.ref_copies) == (1, 1)
    assert het.hgvs_g38 == f"{SEQ}:g.[101_103del];[100_103=]"
    assert hom.copies == (Copy(allele=f"{SEQ}:g.101_103del", copies=2, min_copies=2),)
    assert (ref.name, ref.ref_copies) == ("rs1:=/=", 2)


def test_insertion_site_reads_del_as_the_reference() -> None:
    ins, ref = _indels(INSERTION, "TTA/del", "del/del")
    assert ins.copies[0].allele == f"{SEQ}:g.100_101insTTA"
    assert (ref.copies, ref.ref_copies) == ((), 2)


def test_site_alleles_spelled_out() -> None:
    het, hom = _indels(REPEAT, "TC/TCCTC", "TC/TC")
    assert het.name == "rs1:103_105del/="
    assert hom.copies == (Copy(allele=f"{SEQ}:g.103_105del", copies=2, min_copies=2),)


def test_compact_token_splits_into_the_annotations_own_tokens() -> None:
    # 'T/del + TT': T present = reference, del = the one-base deletion
    het, ref = _indels(HOMOPOLYMER, "T/del", "TT")
    assert het.copies[0].allele == f"{SEQ}:g.103del"
    assert (ref.zygosity, ref.ref_copies) == ("diploid", 2)


def test_ambiguous_indel_notation_stays_unresolved() -> None:
    # two deletions at the site: a lone 'del' could be either
    assert _indel_reason(HOMOPOLYMER, "del/del") == "indel_notation_unmatched"
    misaligned = (
        *DELETION,
        CanonicalVariant.from_spdi(SEQ, 98, "GA", "G", f"{SEQ}:g.100del"),
    )
    assert _indel_reason(misaligned, "del") == "site_alleles_not_aligned"


def test_unresolved_site_keeps_the_written_genotype() -> None:
    gt = unresolved_site(_one("TT", SITE), "rs1", "rsid_not_found")
    assert (gt.name, gt.status, gt.reason) == ("rs1:TT", "unresolved", "rsid_not_found")
