"""Polars table schemas: rows from validated models, schema checks on read-back."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from modephere.types import (
    COMPOUNDS_SCHEMA,
    CONFORMERS_SCHEMA,
    FAILURES_SCHEMA,
    BuildStage,
    FailureCategory,
    FailureRecord,
    FrameSchemaError,
    check_frame,
    compounds_frame,
    conformers_frame,
    failures_frame,
)

from .conftest import ASPIRIN_KEY, NOW

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.types import ConformerEnsemble, StandardizedCompound


class TestFrames:
    def test_compounds(self, standardized: StandardizedCompound) -> None:
        df = compounds_frame([standardized])
        check_frame(df, COMPOUNDS_SCHEMA, table="compounds")
        row = df.row(0, named=True)
        assert row["inchikey"] == ASPIRIN_KEY
        assert row["query_key"] == "cid:2244"
        assert row["label"] == "row-1"
        assert row["source"] == "pubchem"
        assert row["is_ambiguous"] is False

    def test_conformers(self, ensemble: ConformerEnsemble) -> None:
        df = conformers_frame([ensemble, ensemble])
        check_frame(df, CONFORMERS_SCHEMA, table="conformers")
        assert df.height == 6
        assert df["relative_energy"].to_list()[:3] == [0.0, 1.5, 4.0]

    def test_failures(self) -> None:
        record = FailureRecord(
            key="name:unobtainium",
            stage=BuildStage.RESOLVE,
            category=FailureCategory.UNRESOLVED_IDENTIFIER,
            reason="PubChem returned 404",
            occurred_at=NOW,
        )
        df = failures_frame([record])
        check_frame(df, FAILURES_SCHEMA, table="failures")
        assert df["occurred_at"].to_list() == [NOW]

    def test_empty_inputs_keep_the_schema(self) -> None:
        assert compounds_frame([]).schema == COMPOUNDS_SCHEMA
        assert failures_frame([]).height == 0

    def test_parquet_round_trip(
        self, tmp_path: Path, standardized: StandardizedCompound
    ) -> None:
        path = tmp_path / "compounds.parquet"
        compounds_frame([standardized]).write_parquet(path)
        check_frame(pl.read_parquet(path), COMPOUNDS_SCHEMA, table="compounds")


class TestCheckFrame:
    def test_reports_every_problem(self, standardized: StandardizedCompound) -> None:
        df = (
            compounds_frame([standardized])
            .drop("title")
            .with_columns(pl.col("cid").cast(pl.String), pl.lit(1).alias("extra"))
        )
        with pytest.raises(FrameSchemaError) as exc:
            check_frame(df, COMPOUNDS_SCHEMA, table="compounds")
        text = str(exc.value)
        assert "missing 'title'" in text
        assert "unexpected 'extra'" in text
        assert "'cid' is String" in text

    def test_column_order_matters(self, standardized: StandardizedCompound) -> None:
        df = compounds_frame([standardized])
        with pytest.raises(FrameSchemaError, match="column order"):
            check_frame(df.select(reversed(df.columns)), COMPOUNDS_SCHEMA, table="c")
