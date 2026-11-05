"""Graph model of Modephere's knowledge graph (ChEMBL layer, model v1).

::

    (:Compound {chembl_id, pref_name, max_phase, inchikey, smiles, flags, ...})
    (:Compound)-[:HAS_PARENT]->(:Compound)                salt / hydrate -> parent
    (:Compound)-[:HAS_MECHANISM {mec_id, action_type, ...}]->(:Target)
    (:Compound)-[:HAS_BIOACTIVITY {n, pchembl_median, ...}]->(:Target)
                                                          parent level, aggregated
    (:Compound)-[:INDICATED_FOR {max_phase, efo_ids}]->(:Disease {mesh_id})
    (:Compound)-[:HAS_WARNING]->(:DrugWarning {id, warning_type, ...})
    (:Compound)-[:IN_ATC]->(:ATC {code, level})-[:SUBCLASS_OF]->(:ATC)
    (:Compound)-[:METABOLIZED_TO {met_id, enzyme_name, ...}]->(:Compound)
    (:Compound)-[:METABOLIZED_BY {met_id, ...}]->(:Target)
    (:Target {chembl_id, target_type, organism})-[:HAS_COMPONENT]->(:Protein {accession})
    (:Protein)-[:ENCODED_BY]->(:Gene {symbol})            human proteins only
    (:IngestRun)-[:USED]->(:Dataset)                      PROV-O, as in dev_Pgen

Links (:mod:`modephere.kg.links`)::

    (:LibraryCompound {inchikey, num_conformers, ...})-[:REPRESENTS]->(:Compound)
    (:Drug {name, cid})-[:SAME_AS {method, evidence}]->(:Compound)   dev_Pgen drugs

Drug annotations (mechanisms, indications, warnings, ATC, metabolism) hang
off the *parent* compound even when ChEMBL curated them on a salt record; the
record is kept as ``form`` / ``forms`` on the edge (``DrugWarning.form``).

``Gene {symbol}``, ``Drug {name}``, ``Dataset {id}`` and ``IngestRun {id}`` use dev_Pgen's keys
(HGNC symbols, human only), so a ChEMBL target meets a PharmGKB annotation at
the gene when both are loaded into one database. ``HAS_BIOACTIVITY`` is named
for a measurement, not an effect: a pChEMBL of 4 is an inactive compound.
"""

from __future__ import annotations

#: Label -> its unique key property.
NODE_KEYS: dict[str, str] = {
    "Compound": "chembl_id",
    "Target": "chembl_id",
    "Protein": "accession",
    "Gene": "symbol",
    "Disease": "mesh_id",
    "DrugWarning": "id",
    "ATC": "code",
    "LibraryCompound": "inchikey",
    "Drug": "name",
    "Dataset": "id",
    "IngestRun": "id",
}

REL_TYPES: frozenset[str] = frozenset(
    {
        "HAS_PARENT",
        "HAS_MECHANISM",
        "HAS_BIOACTIVITY",
        "INDICATED_FOR",
        "HAS_WARNING",
        "IN_ATC",
        "SUBCLASS_OF",
        "METABOLIZED_TO",
        "METABOLIZED_BY",
        "HAS_COMPONENT",
        "ENCODED_BY",
        "REPRESENTS",
        "SAME_AS",
        "USED",
    }
)

FULLTEXT_COMPOUND_NAMES = "compound_names"

SCHEMA_STATEMENTS: tuple[str, ...] = (
    *(
        f"CREATE CONSTRAINT {label.lower()}_{key} IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        for label, key in NODE_KEYS.items()
    ),
    "CREATE INDEX compound_inchikey IF NOT EXISTS FOR (n:Compound) ON (n.inchikey)",
    "CREATE INDEX compound_pref_name IF NOT EXISTS FOR (n:Compound) ON (n.pref_name)",
    "CREATE INDEX target_pref_name IF NOT EXISTS FOR (n:Target) ON (n.pref_name)",
    "CREATE INDEX drug_cid IF NOT EXISTS FOR (d:Drug) ON (d.cid)",  # as dev_Pgen
    f"CREATE FULLTEXT INDEX {FULLTEXT_COMPOUND_NAMES} IF NOT EXISTS "
    "FOR (n:Compound) ON EACH [n.pref_name, n.names_text]",
)

__all__ = ["FULLTEXT_COMPOUND_NAMES", "NODE_KEYS", "REL_TYPES", "SCHEMA_STATEMENTS"]
