"""Tests for haplotype definitions from PharmVar and ClinPGx allele tables."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modephere.pgx.sources.allele_tables import load_allele_table, match_alternative
from modephere.pgx.sources.pharmvar import load_pharmvar, pharmvar_dataset

from .conftest import FIXTURES

CYP2C19_TSV = FIXTURES / "pharmvar" / "CYP2C19.NC_000010.11.haplotypes.tsv"
NOW = datetime(2026, 10, 18, tzinfo=UTC)


@pytest.fixture(scope="module")
def cyp2c19():
    return {
        d.name: d
        for d in load_pharmvar(CYP2C19_TSV, "pharmvar:CYP2C19@pharmvar-6.2.18")
    }


@pytest.fixture(scope="module")
def tpmt():
    return {
        d.name: d
        for d in load_allele_table(
            FIXTURES / "allele_tables" / "TPMT.xlsx", "clinpgx:TPMT"
        )
    }


@pytest.fixture(scope="module")
def ugt1a1():
    return {
        d.name: d
        for d in load_allele_table(
            FIXTURES / "allele_tables" / "UGT1A1.xlsx", "clinpgx:UGT1A1"
        )
    }


# ----- PharmVar ------------------------------------------------------------ #


def test_pharmvar_core_definition(cyp2c19):
    core = cyp2c19["CYP2C19*2"]
    assert {r.spdi_input for r in core.rows} == {
        "NC_000010.11:94775366:A:G",
        "NC_000010.11:94781858:G:A",
        "NC_000010.11:94842865:A:G",
    }
    assert core.core_name == "CYP2C19*2"
    assert core.gene == "CYP2C19"
    assert core.dataset_id == "pharmvar:CYP2C19@pharmvar-6.2.18"
    assert {r.rsid for r in core.rows} == {"rs12769205", "rs4244285", "rs3758581"}


def test_pharmvar_suballele_links_to_core(cyp2c19):
    assert cyp2c19["CYP2C19*2.001"].core_name == "CYP2C19*2"


def test_reference_is_what_pharmvar_marks_not_star1(cyp2c19):
    # GRCh38 carries CYP2C19*38; *1 differs from it at rs3758581.
    assert cyp2c19["CYP2C19*38"].is_reference
    assert not cyp2c19["CYP2C19*38"].rows
    assert not cyp2c19["CYP2C19*1"].is_reference
    assert [r.spdi_input for r in cyp2c19["CYP2C19*1"].rows] == [
        "NC_000010.11:94842865:A:G"
    ]


def test_pharmvar_deletion_row(cyp2c19):
    spdis = {r.spdi_input for r in cyp2c19["CYP2C19*1.009"].rows}
    assert "NC_000010.11:94761533:T:" in spdis


def test_pharmvar_insertion_uses_interbase_between_flanks(tmp_path):
    tsv = tmp_path / "CYP2A13.NC_000019.10.haplotypes.tsv"
    tsv.write_text(
        "#version=pharmvar-6.2.18\n"
        "Haplotype Name\tGene\trsID\tReferenceSequence\tVariant Start\tVariant Stop\t"
        "Reference Allele\tVariant Allele\tType\n"
        "CYP2A13*3\tCYP2A13\trs57082577\tNC_000019.10\t41090105\t41090106\t-\tACC\tinsertion\n"
    )
    (d,) = load_pharmvar(tsv, "x")
    assert d.rows[0].spdi_input == "NC_000019.10:41090105::ACC"
    assert "rs57082577" in d.rows[0].evidence


def test_pharmvar_dataset_version():
    ds = pharmvar_dataset(CYP2C19_TSV, NOW)
    assert ds.version == "pharmvar-6.2.18"
    assert ds.id.startswith("pharmvar:CYP2C19@pharmvar-6.2.18#")
    assert ds.sha256 is not None and len(ds.sha256) == 64


# ----- ClinPGx allele-definition tables ------------------------------------ #


def test_tpmt_reference_row(tpmt):
    assert tpmt["TPMT*1"].is_reference
    assert not tpmt["TPMT*1"].rows


def test_tpmt_table_selects_alternative_by_cell(tpmt):
    # column AT lists "g.18130687T>C; g.18130687T>G"; the *3A cell "C" picks T>C
    assert {r.hgvs_input for r in tpmt["TPMT*3A"].rows} == {
        "NC_000006.12:g.18138997C>T",
        "NC_000006.12:g.18130687T>C",
    }


def test_tpmt_deletion_and_insertion_cells(tpmt):
    assert [r.hgvs_input for r in tpmt["TPMT*47"].rows] == [
        "NC_000006.12:g.18149104_18149107del"
    ]
    inserted = [
        d for d in tpmt.values() if any("insT" in (r.hgvs_input or "") for r in d.rows)
    ]
    assert inserted
    assert inserted[0].rows[0].hgvs_input == "NC_000006.12:g.18149032_18149033insT"


def test_ugt1a1_repeat_and_combination(ugt1a1):
    assert [r.hgvs_input for r in ugt1a1["UGT1A1*28"].rows] == [
        "NC_000002.12:g.233760235TA[8]"
    ]
    assert ugt1a1["UGT1A1*1"].is_reference
    combo = ugt1a1["UGT1A1*80+*28"]
    assert {r.hgvs_input for r in combo.rows} == {
        "NC_000002.12:g.233759924C>T",
        "NC_000002.12:g.233760235TA[8]",
    }


def test_table_rows_carry_cell_evidence(tpmt):
    row = next(r for r in tpmt["TPMT*3A"].rows if r.hgvs_input.endswith("T>C"))
    assert "TPMT*3A" in row.evidence and "C" in row.evidence


@pytest.mark.parametrize(
    "cell,alternatives,expected",
    [
        ("C", ["g.1T>C", "g.1T>G"], "g.1T>C"),
        ("delAAGT", ["g.5_8del"], "g.5_8del"),
        ("insT", ["g.3_4insT"], "g.3_4insT"),
        ("TA(8)", ["g.9TA[6]", "g.9TA[8]"], "g.9TA[8]"),
        ("R", ["g.1A>G"], None),  # IUPAC ambiguity code: never expanded
        ("A", ["g.1T>C"], None),
    ],
)
def test_match_alternative(cell, alternatives, expected):
    assert match_alternative(cell, alternatives) == expected


def test_non_grch38_row_makes_haplotype_unresolved(tmp_path):
    tsv = tmp_path / "CYP2D6.M33388.haplotypes.tsv"
    tsv.write_text(
        "#version=pharmvar-6.2.18\n"
        "Haplotype Name\tGene\trsID\tReferenceSequence\tVariant Start\tVariant Stop\t"
        "Reference Allele\tVariant Allele\tType\n"
        "CYP2D6*2\tCYP2D6\trs16947\tM33388\t5446\t5446\tG\tA\tsubstitution\n"
    )
    (d,) = load_pharmvar(tsv, "x")
    assert d.rows == ()
    assert d.unresolved_reason == "no_grch38_placement"


def test_only_grch38_pharmvar_files_selected(tmp_path):
    from modephere.pgx.sources.pharmvar import grch38_haplotype_files

    (tmp_path / "CYP2D6" / "M33388").mkdir(parents=True)
    keep = tmp_path / "CYP2D6" / "CYP2D6.NC_000022.11.haplotypes.tsv"
    keep.write_text("x")
    (tmp_path / "CYP2D6" / "M33388" / "CYP2D6.M33388.haplotypes.tsv").write_text("x")
    (tmp_path / "CYP2C19.NC_000010.10.haplotypes.tsv").write_text("x")  # GRCh37
    assert grch38_haplotype_files(tmp_path) == [keep]
