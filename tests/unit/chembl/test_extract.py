"""SQLite -> staging Parquet on the miniature release (see conftest)."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING

import polars as pl
import pytest

from modephere.chembl import (
    ChemblRelease,
    StagingManifest,
    aggregate_bioactivity,
    read_table,
)
from modephere.chembl.__main__ import main
from modephere.chembl.extract import (
    ACTIVITIES_SCHEMA,
    BIOACTIVITY_SCHEMA,
    COMPOUNDS_SCHEMA,
)
from modephere.chembl.source import read_frame
from modephere.core import DataError

if TYPE_CHECKING:
    from pathlib import Path


class TestRelease:
    def test_version_table(self, mini_db: Path) -> None:
        release = ChemblRelease.open(mini_db)
        assert release.id == "chembl_99"
        assert str(release.release_date) == "2026-05-01"
        assert ("EFO 3.74.0", "EFO version") in release.components

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="not found"):
            ChemblRelease.open(tmp_path / "nope.db")

    def test_not_chembl(self, tmp_path: Path) -> None:
        other = tmp_path / "other.db"
        with closing(sqlite3.connect(other)) as conn:
            conn.execute("CREATE TABLE version (name, creation_date, comments)")
        with pytest.raises(DataError, match="ChEMBL_<N>"):
            ChemblRelease.open(other)
        (tmp_path / "junk.db").write_text("not sqlite at all, just text" * 20)
        with pytest.raises(DataError, match="not a readable"):
            ChemblRelease.open(tmp_path / "junk.db")

    def test_read_only(self, mini_db: Path) -> None:
        with ChemblRelease.open(mini_db).connect() as conn:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("DELETE FROM version")

    def test_read_frame_checks_width(self, mini_db: Path) -> None:
        with ChemblRelease.open(mini_db).connect() as conn:
            with pytest.raises(DataError, match="2 columns"):
                read_frame(conn, "SELECT 1, 2", {"a": pl.Int64()})
            empty = read_frame(conn, "SELECT 1 WHERE 0", {"a": pl.Int64()})
            assert empty.schema == pl.Schema({"a": pl.Int64()})


class TestSelection:
    def test_drug_family_and_metabolites(self, staging: Path) -> None:
        compounds = read_table(staging, "compounds")
        assert compounds.schema == pl.Schema(COMPOUNDS_SCHEMA)
        ids = set(compounds["chembl_id"])
        # drugs, the salt of a drug's parent, the metabolite; not the research cpd
        assert ids == {"CHEMBL941", "CHEMBL1642", "CHEMBL25", "CHEMBL424", "CHEMBL6"}
        by_id = {r["chembl_id"]: r for r in compounds.iter_rows(named=True)}
        assert by_id["CHEMBL1642"]["parent_chembl_id"] == "CHEMBL941"
        assert by_id["CHEMBL6"]["is_seed"] is False
        assert by_id["CHEMBL424"]["is_seed"] is True  # named by metabolism
        assert by_id["CHEMBL941"]["max_phase"] == 4.0
        assert by_id["CHEMBL25"]["mw_freebase"] == 180.0  # integer-stored NUMERIC
        assert by_id["CHEMBL941"]["prodrug"] == -1

    def test_synonyms_trimmed(self, staging: Path) -> None:
        syn = read_table(staging, "synonyms")
        assert "Imatinib  Mesylate" in syn["synonym"].to_list()

    def test_list_columns(self, staging: Path) -> None:
        mech = (
            read_table(staging, "mechanisms")
            .filter(pl.col("mec_id") == 10)
            .row(0, named=True)
        )
        assert sorted(mech["refs"]) == ["FDA:label.pdf", "PubMed:12345"]
        assert mech["ref_urls"] == ["http://europepmc.org/abstract/MED/12345"]
        comps = read_table(staging, "components")
        assert comps.filter(pl.col("accession") == "P00519")[
            "gene_symbols"
        ].to_list() == [["ABL1"]]
        assert comps["accession"].null_count() == 0

    def test_atc_tree(self, staging: Path) -> None:
        classes = read_table(staging, "atc_classes")
        row = classes.filter(pl.col("code") == "L01EA").row(0, named=True)
        assert (row["level"], row["parent_code"]) == (4, "L01E")
        assert classes.filter(pl.col("level") == 1)["parent_code"].null_count() == 2


class TestActivities:
    def test_raw_keeps_everything_selected(self, staging: Path) -> None:
        raw = read_table(staging, "activities")
        assert raw.schema == pl.Schema(ACTIVITIES_SCHEMA)
        assert raw["activity_id"].to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert set(raw["parent_chembl_id"]) == {"CHEMBL941"}

    def test_summary_filters(self, staging: Path) -> None:
        summary = read_table(staging, "bioactivity")
        assert summary.schema == pl.Schema(BIOACTIVITY_SCHEMA)
        # organism-level (confidence 1) dropped at the default threshold
        assert summary["target_chembl_id"].to_list() == ["CHEMBL1862", "CHEMBL2366"]
        abl = summary.row(0, named=True)
        assert abl["n"] == 3  # '>' relation, duplicate, invalid value excluded
        assert abl["pchembl_median"] == pytest.approx(8.0)
        assert abl["n_assays"] == 2
        assert abl["n_documents"] == 2
        assert abl["assay_types"] == ["B"]
        assert abl["standard_types"] == ["IC50", "Ki"]
        assert (abl["first_year"], abl["last_year"]) == (2004, 2010)
        mouse = summary.row(1, named=True)
        assert mouse["n_documents"] == 0 and mouse["pchembl_std"] is None

    def test_threshold(self, staging: Path) -> None:
        raw = read_table(staging, "activities")
        assert aggregate_bioactivity(raw, 9)["target_chembl_id"].to_list() == [
            "CHEMBL1862"
        ]
        assert aggregate_bioactivity(raw, 0).height == 3


class TestManifestAndCli:
    def test_manifest(self, staging: Path) -> None:
        m = StagingManifest.read(staging)
        assert m.release == "chembl_99" and m.release_number == 99
        assert m.license == "CC BY-SA 3.0"
        assert m.sqlite_sha256 is not None and len(m.sqlite_sha256) == 64
        assert m.rows["bioactivity"] == 2
        assert m.rows["compounds"] == 5

    def test_cli_extract_then_aggregate(self, mini_db: Path, tmp_path: Path) -> None:
        root = tmp_path / "staging"
        argv = ["--staging", str(root)]
        assert main(["extract", "--sqlite", str(mini_db), "--no-checksum", *argv]) == 0
        assert StagingManifest.read(root / "chembl_99").sqlite_sha256 is None
        assert main(["aggregate", "--min-confidence", "0", *argv]) == 0
        m = StagingManifest.read(root / "chembl_99")
        assert (m.min_confidence, m.rows["bioactivity"]) == (0, 3)

    def test_cli_errors(self, tmp_path: Path) -> None:
        assert main(["aggregate", "--staging", str(tmp_path)]) == 1
        assert main(["extract", "--sqlite", str(tmp_path / "x.db")]) == 1
