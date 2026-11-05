"""A miniature ChEMBL SQLite release with the columns the extraction reads.

Molecules (molregno: chembl_id)::

    1 CHEMBL941   IMATINIB            phase 4, parent
    2 CHEMBL1642  IMATINIB MESYLATE   phase 4, salt of 1; mechanisms on it
    3 CHEMBL25    ASPIRIN             phase 4
    4 CHEMBL424   SALICYLIC ACID      no phase; selected as aspirin's metabolite
    6 CHEMBL6     aspirin salt        no phase; selected as a form of 3
    5 CHEMBL999   research compound   no phase, unreferenced: excluded
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from modephere.chembl import ChemblRelease, extract

if TYPE_CHECKING:
    from pathlib import Path

_TABLES = {
    "version": "name, creation_date, comments",
    "molecule_dictionary": (
        "molregno, chembl_id, pref_name, max_phase, molecule_type, structure_type, "
        "first_approval, usan_year, usan_stem, chirality, availability_type, oral, "
        "parenteral, topical, black_box_warning, withdrawn_flag, prodrug, "
        "natural_product, first_in_class, therapeutic_flag, inorganic_flag, "
        "polymer_flag, orphan, veterinary, chemical_probe"
    ),
    "molecule_hierarchy": "molregno, parent_molregno",
    "compound_structures": "molregno, canonical_smiles, standard_inchi, standard_inchi_key",
    "compound_properties": (
        "molregno, full_molformula, mw_freebase, full_mwt, alogp, hba, hbd, psa, rtb, "
        "aromatic_rings, heavy_atoms, num_ro5_violations, qed_weighted, np_likeness_score"
    ),
    "molecule_synonyms": "molregno, synonyms, syn_type",
    "drug_mechanism": (
        "mec_id, molregno, tid, mechanism_of_action, action_type, direct_interaction, "
        "molecular_mechanism, disease_efficacy, mechanism_comment, selectivity_comment, "
        "binding_site_comment"
    ),
    "mechanism_refs": "mec_id, ref_type, ref_id, ref_url",
    "drug_indication": (
        "drugind_id, molregno, max_phase_for_ind, mesh_id, mesh_heading, efo_id, efo_term"
    ),
    "drug_warning": (
        "warning_id, molregno, warning_type, warning_class, warning_description, "
        "warning_country, warning_year, efo_id, efo_term, efo_id_for_warning_class"
    ),
    "compound_records": "record_id, molregno",
    "metabolism": (
        "met_id, drug_record_id, substrate_record_id, metabolite_record_id, enzyme_tid, "
        "enzyme_name, met_conversion, organism, tax_id, pathway_id, pathway_key, met_comment"
    ),
    "molecule_atc_classification": "molregno, level5",
    "atc_classification": (
        "who_name, level1, level2, level3, level4, level5, level1_description, "
        "level2_description, level3_description, level4_description"
    ),
    "target_dictionary": (
        "tid, chembl_id, pref_name, target_type, organism, tax_id, species_group_flag"
    ),
    "component_sequences": (
        "component_id, accession, component_type, description, organism, tax_id"
    ),
    "component_synonyms": "component_id, component_synonym, syn_type",
    "target_components": "tid, component_id, homologue",
    "assays": (
        "assay_id, chembl_id, assay_type, confidence_score, relationship_type, "
        "assay_organism, bao_format, tid"
    ),
    "docs": "doc_id, chembl_id, year",
    "activities": (
        "activity_id, molregno, assay_id, doc_id, standard_type, standard_relation, "
        "standard_value, standard_units, pchembl_value, activity_comment, "
        "data_validity_comment, potential_duplicate, bao_endpoint, action_type, "
        "modality, src_id"
    ),
}

_N = None
_FLAGS = (0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0)  # oral..chemical_probe (14)
_ORAL = (1, *_FLAGS[1:])

_ROWS: dict[str, list[tuple[object, ...]]] = {
    "version": [
        ("ChEMBL_99", "2026-05-01 00:00:00.000000", "ChEMBL Release 99"),
        ("EFO 3.74.0", _N, "EFO version"),
    ],
    "molecule_dictionary": [
        (
            1,
            "CHEMBL941",
            "IMATINIB",
            4,
            "Small molecule",
            "MOL",
            2001,
            1997,
            "-tinib",
            0,
            1,
            *_ORAL,
        ),
        (
            2,
            "CHEMBL1642",
            "IMATINIB MESYLATE",
            4.0,
            "Small molecule",
            "MOL",
            2001,
            _N,
            _N,
            0,
            1,
            *_ORAL,
        ),
        (
            3,
            "CHEMBL25",
            "ASPIRIN",
            4,
            "Small molecule",
            "MOL",
            1950,
            _N,
            _N,
            2,
            2,
            *_ORAL,
        ),
        (
            4,
            "CHEMBL424",
            "SALICYLIC ACID",
            _N,
            "Small molecule",
            "MOL",
            _N,
            _N,
            _N,
            2,
            -1,
            *_FLAGS,
        ),
        (5, "CHEMBL999", _N, _N, "Small molecule", "MOL", _N, _N, _N, 2, -1, *_FLAGS),
        (
            6,
            "CHEMBL6",
            "ASPIRIN SODIUM",
            _N,
            "Small molecule",
            "MOL",
            _N,
            _N,
            _N,
            2,
            -1,
            *_FLAGS,
        ),
    ],
    "molecule_hierarchy": [(1, 1), (2, 1), (3, 3), (4, 4), (5, 5), (6, 3)],
    "compound_structures": [
        (
            1,
            "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
            "InChI=1S/imatinib",
            "KTUFNOKKBVMGRW-UHFFFAOYSA-N",
        ),
        (
            2,
            "CS(=O)(=O)O.Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
            "InChI=1S/mesylate",
            "YLMAHDNUQAMNNX-UHFFFAOYSA-N",
        ),
        (3, "CC(=O)Oc1ccccc1C(=O)O", "InChI=1S/aspirin", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"),
        (4, "O=C(O)c1ccccc1O", "InChI=1S/salicylic", "YGSDEFSMJLZEOE-UHFFFAOYSA-N"),
        (5, "CCO", "InChI=1S/ethanol", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
        (
            6,
            "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",
            "InChI=1S/na",
            "XXXXXXXXXXXXXX-UHFFFAOYSA-M",
        ),
    ],
    "compound_properties": [
        (1, "C29H31N7O", 493.62, 493.62, 4.59, 7, 2, 86.28, 7, 4, 37, 0, 0.39, -0.9),
        (3, "C9H8O4", 180, 180.16, 1.31, 3, 1, 63.6, 2, 1, 13, 0, 0.55, 0.12),
    ],
    "molecule_synonyms": [
        (1, "Imatinib", "INN"),
        (1, "STI-571", "RESEARCH_CODE"),
        (2, "Gleevec", "TRADE_NAME"),
        (2, " Imatinib  Mesylate ", "USAN"),
        (3, "Aspirin", "BAN"),
        (3, "Acetylsalicylic acid", "OTHER"),
        (6, "Aspirin", "OTHER"),
    ],
    "drug_mechanism": [
        (
            10,
            2,
            100,
            "Bcr/Abl fusion protein inhibitor",
            "INHIBITOR",
            1,
            1,
            1,
            _N,
            _N,
            _N,
        ),
        (11, 3, _N, "Unknown", "INHIBITOR", 0, 0, 0, _N, _N, _N),
    ],
    "mechanism_refs": [
        (10, "PubMed", "12345", "http://europepmc.org/abstract/MED/12345"),
        (10, "FDA", "label.pdf", _N),
    ],
    "drug_indication": [
        (
            1,
            2,
            4,
            "D015464",
            "Leukemia, Myelogenous, Chronic, BCR-ABL Positive",
            "EFO:0000339",
            "CML",
        ),
        (
            2,
            1,
            3,
            "D015464",
            "Leukemia, Myelogenous, Chronic, BCR-ABL Positive",
            _N,
            _N,
        ),
        (3, 3, 4, "D010146", "Pain", "EFO:0003843", "pain"),
    ],
    "drug_warning": [
        (
            7,
            3,
            "Black Box Warning",
            "Gastrointestinal toxicity",
            "Bleeding",
            "USA",
            2000,
            _N,
            _N,
            _N,
        ),
    ],
    "compound_records": [(1000, 3), (1001, 4), (1002, 5)],
    "metabolism": [
        (
            50,
            1000,
            1000,
            1001,
            200,
            "CYP2C9",
            "deacetylation",
            "Homo sapiens",
            9606,
            1,
            "p1",
            _N,
        ),
    ],
    "molecule_atc_classification": [(1, "L01EA01"), (2, "L01EA01"), (3, "B01AC06")],
    "atc_classification": [
        (
            "imatinib",
            "L",
            "L01",
            "L01E",
            "L01EA",
            "L01EA01",
            "ANTINEOPLASTIC",
            "ANTINEOPLASTIC AGENTS",
            "PROTEIN KINASE INHIBITORS",
            "BCR-ABL tyrosine kinase inhibitors",
        ),
        (
            "acetylsalicylic acid",
            "B",
            "B01",
            "B01A",
            "B01AC",
            "B01AC06",
            "BLOOD",
            "ANTITHROMBOTIC AGENTS",
            "ANTITHROMBOTIC AGENTS",
            "Platelet aggregation inhibitors",
        ),
        (
            "other",
            "B",
            "B01",
            "B01A",
            "B01AA",
            "B01AA03",
            "BLOOD",
            "ANTITHROMBOTIC AGENTS",
            "ANTITHROMBOTIC AGENTS",
            "Vitamin K antagonists",
        ),
    ],
    "target_dictionary": [
        (
            100,
            "CHEMBL1862",
            "Tyrosine-protein kinase ABL1",
            "SINGLE PROTEIN",
            "Homo sapiens",
            9606,
            0,
        ),
        (
            200,
            "CHEMBL3397",
            "Cytochrome P450 2C9",
            "SINGLE PROTEIN",
            "Homo sapiens",
            9606,
            0,
        ),
        (300, "CHEMBL2366", "Abl1 (mouse)", "SINGLE PROTEIN", "Mus musculus", 10090, 0),
        (400, "CHEMBL612", "Homo sapiens", "ORGANISM", "Homo sapiens", 9606, 0),
        (500, "CHEMBL5555", "Unused target", "SINGLE PROTEIN", "Homo sapiens", 9606, 0),
    ],
    "component_sequences": [
        (1, "P00519", "PROTEIN", "ABL1 human", "Homo sapiens", 9606),
        (2, "P11712", "PROTEIN", "CYP2C9 human", "Homo sapiens", 9606),
        (3, "P00520", "PROTEIN", "Abl1 mouse", "Mus musculus", 10090),
        (4, _N, "PROTEIN", "no accession", "Homo sapiens", 9606),
    ],
    "component_synonyms": [
        (1, "ABL1", "GENE_SYMBOL"),
        (1, "ABL", "GENE_SYMBOL_OTHER"),
        (2, "CYP2C9", "GENE_SYMBOL"),
        (3, "Abl1", "GENE_SYMBOL"),
    ],
    "target_components": [(100, 1, 0), (200, 2, 0), (300, 3, 0), (500, 4, 0)],
    "assays": [
        (1, "CHEMBL_A1", "B", 9, "D", "Homo sapiens", "BAO_0000357", 100),
        (2, "CHEMBL_A2", "B", 9, "D", "Homo sapiens", "BAO_0000357", 100),
        (3, "CHEMBL_A3", "F", 1, "N", "Homo sapiens", "BAO_0000218", 400),
        (4, "CHEMBL_A4", "B", 8, "H", "Mus musculus", "BAO_0000357", 300),
    ],
    "docs": [(1, "CHEMBL_D1", 2004), (2, "CHEMBL_D2", 2010)],
    "activities": [
        # imatinib mesylate on ABL1: counts 8.0, 7.0, 9.0 (manually validated)
        (1, 2, 1, 1, "IC50", "=", 10, "nM", 8.0, _N, _N, 0, "BAO_0000190", _N, _N, 1),
        (2, 1, 2, 2, "Ki", "=", 100, "nM", 7, _N, _N, 0, "BAO_0000192", _N, _N, 1),
        (
            3,
            1,
            1,
            1,
            "IC50",
            "=",
            1,
            "nM",
            9.0,
            _N,
            "Manually validated",
            0,
            _N,
            _N,
            _N,
            1,
        ),
        # excluded from the summary
        (4, 1, 1, 1, "IC50", ">", 10000, "nM", 5.0, _N, _N, 0, _N, _N, _N, 1),
        (5, 1, 1, 1, "IC50", "=", 10, "nM", 8.0, _N, _N, 1, _N, _N, _N, 1),
        (
            6,
            1,
            1,
            1,
            "IC50",
            "=",
            10,
            "nM",
            4.0,
            _N,
            "Outside typical range",
            0,
            _N,
            _N,
            _N,
            1,
        ),
        (7, 1, 3, 1, "GI50", "=", 1000, "nM", 6.0, _N, _N, 0, _N, _N, _N, 1),
        (8, 1, 1, 1, "Inhibition", "=", 50, "%", _N, "Active", _N, 0, _N, _N, _N, 1),
        # mouse homologue, confidence 8
        (9, 1, 4, _N, "Ki", "=", 3, "nM", 8.52, _N, _N, 0, _N, _N, _N, 1),
        # research compound: not selected at all
        (10, 5, 1, 1, "IC50", "=", 10, "nM", 8.0, _N, _N, 0, _N, _N, _N, 1),
    ],
}


def build_mini_chembl(path: Path) -> Path:
    conn = sqlite3.connect(path)
    for table, columns in _TABLES.items():
        conn.execute(f"CREATE TABLE {table} ({columns})")
        rows = _ROWS.get(table, [])
        if rows:
            marks = ", ".join("?" * len(columns.split(",")))
            conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="session")
def mini_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_mini_chembl(tmp_path_factory.mktemp("chembl") / "chembl_99.db")


@pytest.fixture(scope="session")
def staging(mini_db: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("staging")
    return extract(ChemblRelease.open(mini_db), root, min_confidence=7)
