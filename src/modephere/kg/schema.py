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

Pharmacogenomic layer (:mod:`modephere.pgx`, model v3)::

    (:Variant {hgvs_g38, spdi_g38, seq_id_g38, pos_g38, ref_g38, alt_g38,
               rsids, star_alleles})-[:IN_GENE]->(:Gene)
    (:RefSNP {rsid})-[:HAS_ALLELE]->(:Variant)            alternatives at one site
    (:RefSNP)-[:MERGED_INTO]->(:RefSNP)
    (:Haplotype {name, gene, is_reference, hgvs_g38, structure})
        -[:HAS_VARIANT {phase: "cis"}]->(:Variant)
    (:Haplotype)-[:SUBALLELE_OF]->(:Haplotype)
    (:Annotation {id, source, pmid, direction, ...})       a PharmGKB finding
    (:Annotation)-[:CONCERNS_ALLELE]->(:Allele {label, kind, status, reason})
    (:Allele)-[:REFERS_TO]->(:RefSNP) | (:Haplotype)
    (:Annotation)-[:CONCERNS_GENE]->(:Gene), (:Annotation)-[:FOR_DRUG]->(:Drug)
    (:Annotation)-[:IMPLICATES {method, evidence}]->(:Variant)   allele-specific
    (:Allele)-[:REFERS_TO {copies, min_copies}]->(:Haplotype)   '*2xN': the copied allele
    (:Annotation)-[:TESTED | COMPARED_TO {strand_assumed}]->(:Genotype {name,
               locus, zygosity, status, reason, ref_copies, gene_copies,
               gene_copies_min, hgvs_g38})
    (:Genotype)-[:HAS_COPY {copies, min_copies}]->(:Haplotype)  diplotype (star locus)
    (:Genotype)-[:HAS_COPY {copies, min_copies}]->(:Variant)    site genotype (rsID)

A variant is its GRCh38 genomic HGVS (``hgvs_g38``, derived from the canonical
NCBI SPDI); it stands on its own, rsIDs and star alleles are metadata on it.
Assembly-specific properties carry the assembly suffix, so another assembly
(``hgvs_g37``, ``hgvs_t2t``) can be added beside them. Haplotype and genotype
``hgvs_g38`` use HGVS allele syntax (``g.[a;b]``, ``g.[a];[=]``). ``Genotype``
keeps the outcomes that depend on two alleles (``*1/*22`` vs ``*22/*22``,
``CT`` vs ``TT``): ``copies`` is exact (absent for a carrier-only report or
``xN``), ``min_copies`` the lower bound. Haplotypes that are no set of small
variants carry ``structure``: ``gene_deletion`` (``CYP2D6*5``, ``GSTM1*0``,
HGVS over the gene span, ``hgvs_t2t`` too where NCBI places the gene on
T2T-CHM13) or ``hla_typed`` (identified by the IPD-IMGT/HLA name, no HGVS).
The layers meet at ``Gene`` (HGNC symbol) and at ``Drug`` (``SAME_AS`` a
ChEMBL ``Compound``, ``kg link-drugs``).

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
    # pharmacogenomic layer
    "Annotation": "id",
    "Allele": "label",
    "RefSNP": "rsid",
    "Haplotype": "name",
    "Variant": "hgvs_g38",
    "Genotype": "name",
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
        # pharmacogenomic layer
        "CONCERNS_ALLELE",
        "CONCERNS_GENE",
        "FOR_DRUG",
        "IMPLICATES",
        "REFERS_TO",
        "MERGED_INTO",
        "HAS_ALLELE",
        "HAS_VARIANT",
        "SUBALLELE_OF",
        "IN_GENE",
        "TESTED",
        "COMPARED_TO",
        "HAS_COPY",
    }
)

FULLTEXT_COMPOUND_NAMES = "compound_names"

#: Schema objects of earlier models, dropped so they cannot shadow the new keys.
OBSOLETE_SCHEMA: tuple[str, ...] = (
    "DROP CONSTRAINT variant_spdi IF EXISTS",  # model v2: Variant keyed by SPDI
    "DROP INDEX variant_hgvs_g IF EXISTS",
)

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
    # one uniqueness constraint per assembly-specific identity of a variant
    "CREATE CONSTRAINT variant_spdi_g38 IF NOT EXISTS "
    "FOR (v:Variant) REQUIRE v.spdi_g38 IS UNIQUE",
    "CREATE INDEX haplotype_hgvs_g38 IF NOT EXISTS FOR (h:Haplotype) ON (h.hgvs_g38)",
    "CREATE INDEX genotype_locus IF NOT EXISTS FOR (g:Genotype) ON (g.locus)",
    "CREATE INDEX ingest_run_layer IF NOT EXISTS FOR (r:IngestRun) ON (r.layer)",
    f"CREATE FULLTEXT INDEX {FULLTEXT_COMPOUND_NAMES} IF NOT EXISTS "
    "FOR (n:Compound) ON EACH [n.pref_name, n.names_text]",
    *OBSOLETE_SCHEMA,
)

__all__ = [
    "FULLTEXT_COMPOUND_NAMES",
    "NODE_KEYS",
    "OBSOLETE_SCHEMA",
    "REL_TYPES",
    "SCHEMA_STATEMENTS",
]
