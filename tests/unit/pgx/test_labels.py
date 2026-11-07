"""Tests for classifying PharmGKB allele labels."""

from __future__ import annotations

import pytest

from modephere.pgx.labels import AlleleKind, Structure, classify_label


@pytest.mark.parametrize(
    ("label", "kind", "reason"),
    [
        ("rs4244285", "rsid", None),
        ("CYP2C19*2", "star", None),
        ("CYP2C19*2.001", "star", None),
        ("TPMT*3A", "star", None),
        ("UGT1A1*28", "star", None),
        ("NAT2*5", "star", None),  # SNV-defined: *5 is structural only for CYP2D6
        ("SLCO1B1*5", "star", None),
        ("HLA-B*58:01", "hla", None),  # named by IPD-IMGT/HLA
        ("HLA-B*5801x", "hla", "unrecognized_hla_name"),
        ("CYP2D6*5", "cnv", None),  # whole-gene deletion
        ("CYP2D6*1xN", "cnv", None),
        ("CYP2D6*10x2", "cnv", None),
        ("CYP2D6*36", "cnv", "hybrid_allele"),  # CYP2D6-2D7 hybrid
        ("CYP2D6*13", "cnv", "hybrid_allele"),
        ("CYP2D6*36xN", "cnv", "duplicated_structural_allele"),
        ("CYP2A6*4", "cnv", None),  # whole-gene deletion
        ("CYP2A6*12", "cnv", "hybrid_allele"),  # CYP2A6-2A7 hybrid
        ("CYP2A6*2", "star", None),
        ("GSTM1 null", "cnv", None),
        ("GSTT1 non-null", "cnv", "genotype_class"),
        ("CYP2C19 poor metabolizer", "phenotype", "not_a_variant"),
        (
            "CYP2D6 poor metabolizer and intermediate metabolizer genotypes",
            "phenotype",
            "not_a_variant",
        ),
        ("NAT2 slow acetylator", "phenotype", "not_a_variant"),
        ("G6PD deficiency", "phenotype", "not_a_variant"),
        ("SLC6A4 HTTLPR long form (L allele)", "other", "not_in_pharmvar_or_tables"),
        ("G6PD Canton", "other", "not_in_pharmvar_or_tables"),
    ],
)
def test_classify(label, kind, reason):
    c = classify_label(label)
    assert (c.kind, c.reason, c.label) == (AlleleKind(kind), reason, label)


def test_rsid_number():
    c = classify_label("rs3892097")
    assert c.rsid == 3892097
    assert c.haplotype_name is None


def test_star_keeps_full_name():
    assert classify_label("CYP2C19*2").haplotype_name == "CYP2C19*2"


def test_label_kept_verbatim():
    assert classify_label("  CYP2C19*2 ").label == "  CYP2C19*2 "
    assert classify_label("  CYP2C19*2 ").haplotype_name == "CYP2C19*2"


@pytest.mark.parametrize(
    ("label", "structure", "haplotype", "copies"),
    [
        ("CYP2D6*5", Structure.GENE_DELETION, "CYP2D6*5", 0),
        ("GSTM1 null", Structure.GENE_DELETION, "GSTM1*0", 0),
        ("GSTT1*0", Structure.GENE_DELETION, "GSTT1*0", 0),
        ("CYP2D6*2xN", Structure.DUPLICATION, "CYP2D6*2", None),
        ("CYP2D6*1x2", Structure.DUPLICATION, "CYP2D6*1", 2),
        ("HLA-B*58:01", Structure.HLA_TYPED, "HLA-B*58:01", 1),
        ("CYP2D6*36", Structure.HYBRID, None, 1),
    ],
)
def test_structure(label, structure, haplotype, copies):
    c = classify_label(label)
    assert (c.structure, c.haplotype_name, c.copies) == (structure, haplotype, copies)
