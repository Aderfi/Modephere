"""Staging tables -> graph rows (pure: no database)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modephere.chembl import StagingManifest
from modephere.kg.chembl import chembl_dataset, chembl_rows
from modephere.kg.schema import NODE_KEYS, REL_TYPES

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.kg import GraphRows


@pytest.fixture(scope="module")
def rows(staging: Path) -> GraphRows:
    return chembl_rows(staging)


def _rels(rows: GraphRows, rel_type: str) -> list[dict[str, object]]:
    return next(r for spec, r in rows.rels if spec.type == rel_type)


def _node(rows: GraphRows, label: str, key: str) -> dict[str, object]:
    return next(r["props"] for r in rows.nodes[label] if r["key"] == key)


def test_schema_covers_everything(rows: GraphRows) -> None:
    assert set(rows.nodes) <= set(NODE_KEYS)
    assert {spec.type for spec, _ in rows.rels} <= REL_TYPES


def test_counts(rows: GraphRows) -> None:
    assert rows.counts() == {
        "Compound": 5,
        "Target": 3,  # ABL1, CYP2C9, mouse Abl1; not the organism nor the unused one
        "Protein": 3,
        "Gene": 2,  # human only: ABL1, CYP2C9 (not mouse Abl1)
        "Disease": 2,
        "DrugWarning": 1,
        "ATC": 10,  # two level-5 codes and their ancestors only
        "HAS_PARENT": 2,
        "HAS_MECHANISM": 1,  # the untargeted mechanism has no edge
        "HAS_BIOACTIVITY": 2,
        "INDICATED_FOR": 2,
        "HAS_WARNING": 1,
        "IN_ATC": 2,
        "SUBCLASS_OF": 8,
        "METABOLIZED_TO": 1,
        "METABOLIZED_BY": 1,
        "HAS_COMPONENT": 3,
        "ENCODED_BY": 2,
    }


def test_compound_props(rows: GraphRows) -> None:
    imatinib = _node(rows, "Compound", "CHEMBL941")
    assert imatinib["oral"] is True and imatinib["prodrug"] is None  # -1 = unknown
    assert imatinib["is_parent"] is True
    assert str(imatinib["smiles"]).startswith("Cc1ccc")
    assert "standard_inchi" not in imatinib
    assert imatinib["names_text"] == "IMATINIB | Imatinib | STI-571"
    # lifted from the mesylate record
    assert imatinib["mechanisms_of_action"] == ["Bcr/Abl fusion protein inhibitor"]
    aspirin = _node(rows, "Compound", "CHEMBL25")
    assert aspirin["mechanisms_of_action"] == ["Unknown"]
    assert _node(rows, "Compound", "CHEMBL424")["synonyms"] == []


def test_annotations_hang_off_the_parent(rows: GraphRows) -> None:
    (mech,) = _rels(rows, "HAS_MECHANISM")
    assert (mech["start"], mech["end"], mech["key"]) == ("CHEMBL941", "CHEMBL1862", 10)
    props = mech["props"]
    assert props["form"] == "CHEMBL1642"  # type: ignore[index]
    assert props["direct_interaction"] is True  # type: ignore[index]
    cml = next(r for r in _rels(rows, "INDICATED_FOR") if r["end"] == "D015464")
    assert cml["start"] == "CHEMBL941"
    assert cml["props"] == {
        "max_phase": 4.0,
        "efo_ids": ["EFO:0000339"],
        "forms": ["CHEMBL1642", "CHEMBL941"],
    }
    (atc,) = [r for r in _rels(rows, "IN_ATC") if r["end"] == "L01EA01"]
    assert atc["props"] == {"forms": ["CHEMBL1642", "CHEMBL941"]}


def test_bioactivity_edge(rows: GraphRows) -> None:
    abl = next(r for r in _rels(rows, "HAS_BIOACTIVITY") if r["end"] == "CHEMBL1862")
    assert abl["start"] == "CHEMBL941"
    props = abl["props"]
    assert props["n"] == 3 and props["min_confidence"] == 7  # type: ignore[index]


def test_metabolism_and_genes(rows: GraphRows) -> None:
    (to,) = _rels(rows, "METABOLIZED_TO")
    assert (to["start"], to["end"], to["key"]) == ("CHEMBL25", "CHEMBL424", 50)
    (by,) = _rels(rows, "METABOLIZED_BY")
    assert by["end"] == "CHEMBL3397"
    pairs = {(r["start"], r["end"]) for r in _rels(rows, "ENCODED_BY")}
    assert pairs == {("P00519", "ABL1"), ("P11712", "CYP2C9")}
    assert _node(rows, "Protein", "P00520")["gene_symbols"] == ["Abl1"]


def test_atc_tree(rows: GraphRows) -> None:
    edges = {(r["start"], r["end"]) for r in _rels(rows, "SUBCLASS_OF")}
    assert ("L01EA01", "L01EA") in edges and ("L01", "L") in edges
    assert not any(e[0] == "B01AA03" for e in edges)  # unused code


def test_dataset(staging: Path) -> None:
    ds = chembl_dataset(StagingManifest.read(staging))
    assert (ds.id, ds.name, ds.version) == ("chembl_99", "ChEMBL", "99")
    assert ds.uri.endswith("/chembl_99/") and ds.license == "CC BY-SA 3.0"
