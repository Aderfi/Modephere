"""Tests for the knowledge-graph input readers."""

from __future__ import annotations

import json

from modephere.pgx.sources.pharmgkb import load_drug_cid_map


def test_aligned_tsv_maps_names_regardless_of_column_order(tmp_path):
    # train_data.tsv lists drugs_cid *before* drugs.
    tsv = tmp_path / "train.tsv"
    tsv.write_text(
        "drugs_cid\tdrugs\tgene\n"
        "3385\tfluorouracil\tDPYD\n"
        "5460033|2907\tcisplatin|cyclophosphamide\tXRCC1\n"
        "1|2\tonly_one_name\tX\n"
    )
    mapping = load_drug_cid_map(aligned_tsv=tsv)
    assert mapping == {
        "fluorouracil": 3385,
        "cisplatin": 5460033,
        "cyclophosphamide": 2907,
    }


def test_name_cache_underscores_become_spaces(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"phosphoric_acid": 1004}))
    assert load_drug_cid_map(name_cache=cache) == {"phosphoric acid": 1004}


def test_cid_provenance_names_the_supplying_file(tmp_path):
    from modephere.pgx.sources.pharmgkb import (
        ALIGNED_TSV_DATASET,
        NAME_CACHE_DATASET,
        load_drug_cid_provenance,
    )

    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"codeine": 5284371, "fluorouracil": 1}))
    tsv = tmp_path / "train.tsv"
    tsv.write_text("drugs_cid\tdrugs\n3385\tfluorouracil\n")
    prov = load_drug_cid_provenance(cache, tsv)
    assert prov["codeine"] == (5284371, NAME_CACHE_DATASET)
    assert prov["fluorouracil"] == (3385, ALIGNED_TSV_DATASET)  # later source wins
