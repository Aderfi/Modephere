"""Offline lookup and LibraryDataset labels from the staging tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from modephere.chembl import ChemblLookup, bioactivity_labels
from modephere.chembl.lookup import normalise_name

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def lookup(staging: Path) -> ChemblLookup:
    return ChemblLookup(staging)


class TestLookup:
    def test_size(self, lookup: ChemblLookup) -> None:
        assert len(lookup) == 5

    def test_name_normalisation(self) -> None:
        assert normalise_name("  Imatinib\tMESYLATE ") == "imatinib mesylate"

    def test_trade_name_finds_the_salt(self, lookup: ChemblLookup) -> None:
        (hit,) = lookup.by_name("GLEEVEC")
        assert (hit.chembl_id, hit.parent_chembl_id) == ("CHEMBL1642", "CHEMBL941")
        assert hit.matched == "TRADE_NAME" and not hit.is_parent

    def test_ranking(self, lookup: ChemblLookup) -> None:
        hits = lookup.by_name("aspirin")
        # pref_name match first, then the salt that lists it as a synonym
        assert [h.chembl_id for h in hits] == ["CHEMBL25", "CHEMBL6"]
        assert hits[0].matched == "pref_name"
        assert hits[0].smiles == "CC(=O)Oc1ccccc1C(=O)O"

    def test_unknown(self, lookup: ChemblLookup) -> None:
        assert lookup.by_name("unobtainium") == []
        assert lookup.by_chembl_id("CHEMBL1") is None
        assert lookup.by_inchikey("AAAAAAAAAAAAAA-UHFFFAOYSA-N") == []

    def test_ids(self, lookup: ChemblLookup) -> None:
        hit = lookup.by_chembl_id(" chembl941 ")
        assert hit is not None and hit.pref_name == "IMATINIB" and hit.max_phase == 4.0
        (by_key,) = lookup.by_inchikey("bsynrymutxbxsq-uhfffaoysa-n")
        assert by_key.chembl_id == "CHEMBL25"


class TestLabels:
    def test_wide(self, staging: Path) -> None:
        labels = bioactivity_labels(staging)
        assert labels.columns == ["inchikey", "chembl_id", "CHEMBL1862", "CHEMBL2366"]
        assert labels["CHEMBL1862"].to_list() == [pytest.approx(8.0)]
        assert labels["inchikey"].to_list() == ["KTUFNOKKBVMGRW-UHFFFAOYSA-N"]

    def test_target_order_and_filters(self, staging: Path) -> None:
        labels = bioactivity_labels(
            staging, ["CHEMBL2366", "CHEMBL1862", "CHEMBL0"], value="pchembl_max"
        )
        assert labels.columns == ["inchikey", "chembl_id", "CHEMBL2366", "CHEMBL1862"]
        assert labels["CHEMBL1862"].to_list() == [pytest.approx(9.0)]
        assert bioactivity_labels(staging, min_measurements=2).columns[2:] == [
            "CHEMBL1862"
        ]

    def test_long(self, staging: Path) -> None:
        long = bioactivity_labels(staging, wide=False)
        assert long.columns == ["inchikey", "chembl_id", "target", "value", "n"]
        assert long.height == 2 and long.schema["value"] == pl.Float64
