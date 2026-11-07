"""Neo4j -> parquet tables of the pharmacogenomic layer.

* ``annotation_rows.parquet`` — one row per ``:Annotation`` with its drugs,
  genes, allele labels and implicated variants. Each drug is a struct
  ``{name, cid, chembl_id, inchikey}``: ``chembl_id`` through ``SAME_AS``
  (``kg link-drugs``), ``inchikey`` through a library compound that
  ``REPRESENTS`` it (``kg link-library``), so a consumer finds the drug's
  Modephere graphs (``compounds/<InChIKey>.safetensors``) without a lookup.
  Drug, gene and allele lists are sorted, so the output is deterministic.
* ``allele_variants.parquet`` — allele label -> canonical variants (the
  label standardization table: rsID sites with every alternative, star
  alleles with their *cis* members).
* ``annotation_variants.parquet`` — the allele-specific ``IMPLICATES`` links.
* ``annotation_genotypes.parquet`` — one row per genotype an annotation
  ``TESTED`` or was ``COMPARED_TO`` (``*1/*22``, ``rs1799853:C/T``,
  ``CYP2D6*1/*5``), with the dosage it implies: ``haplotypes`` (star locus,
  with each haplotype's ``structure``) and ``variants`` — every variant with
  its copy number, star alleles expanded into their cis members (a variant on
  both haplotypes counts twice). ``copies`` is exact or null (carrier-only
  report, ``xN``); ``min_copies`` is always the lower bound. ``gene_copies`` /
  ``gene_copies_min`` count gene copies (a deletion adds none);
  ``strand_assumed`` marks links read on PharmGKB's forward-strand convention
  where the report alone could not prove the strand.

Variants are identified by ``hgvs_g38`` (GRCh38 genomic HGVS); ``spdi_g38``
travels along. ``annotation_rows`` keeps every annotation and the full drug
identity (dev_Pgen's training-row snapshot is derived from it, consumer side).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

ANNOTATION_ROWS_FILE = "annotation_rows.parquet"
ALLELE_VARIANTS_FILE = "allele_variants.parquet"
ANNOTATION_VARIANTS_FILE = "annotation_variants.parquet"
ANNOTATION_GENOTYPES_FILE = "annotation_genotypes.parquet"

ANNOTATION_ROWS = """
MATCH (a:Annotation)
WHERE $sources IS NULL OR a.source IN $sources
CALL (a) {
  OPTIONAL MATCH (a)-[:FOR_DRUG]->(d:Drug)
  OPTIONAL MATCH (d)-[:SAME_AS]->(c:Compound)
  OPTIONAL MATCH (lc:LibraryCompound)-[:REPRESENTS]->(c)
  WITH d, c, lc ORDER BY lc.inchikey
  WITH d, c, collect(lc.inchikey)[0] AS inchikey
  ORDER BY d.name
  RETURN [x IN collect(CASE WHEN d IS NULL THEN null ELSE
           {name: d.name, cid: d.cid, chembl_id: c.chembl_id, inchikey: inchikey}
         END) WHERE x IS NOT NULL] AS drugs
}
CALL (a) {
  OPTIONAL MATCH (a)-[:CONCERNS_GENE]->(g:Gene)
  WITH g ORDER BY g.symbol
  RETURN collect(g.symbol) AS genes
}
CALL (a) {
  OPTIONAL MATCH (a)-[:CONCERNS_ALLELE]->(al:Allele)
  WITH al ORDER BY al.label
  RETURN collect(al.label) AS genotypes
}
CALL (a) {
  OPTIONAL MATCH (a)-[:IMPLICATES]->(v:Variant)
  WITH v ORDER BY v.hgvs_g38
  RETURN collect(DISTINCT v.hgvs_g38) AS implicated_hgvs
}
RETURN a.id AS annotation_id,
       a.source AS source,
       a.pmid AS pmid,
       drugs,
       genes,
       genotypes,
       implicated_hgvs,
       a.alleles AS alleles,
       a.comparison_alleles AS comparison_alleles,
       coalesce(a.phenotype_categories, []) AS phenotype_categories,
       a.direction AS direction,
       a.significance AS significance,
       a.associated AS associated
ORDER BY annotation_id
"""

ALLELE_VARIANTS = """
MATCH (al:Allele)
OPTIONAL MATCH (al)-[:REFERS_TO]->(t)
OPTIONAL MATCH (t)-[:MERGED_INTO]->(cur:RefSNP)
WITH al, t, coalesce(cur, t) AS site
OPTIONAL MATCH (site)-[hv:HAS_ALLELE|HAS_VARIANT]->(v:Variant)
RETURN al.label AS label, al.kind AS kind, al.status AS status, al.reason AS reason,
       v.hgvs_g38 AS hgvs_g38, v.spdi_g38 AS spdi_g38,
       CASE WHEN t:RefSNP THEN 'rsid' WHEN t:Haplotype THEN 'haplotype' END AS path,
       CASE WHEN hv IS NULL THEN null
            WHEN type(hv) = 'HAS_VARIANT' THEN 'cis_member'
            ELSE 'site_alternative' END AS relation,
       coalesce(hv.sources, al.sources) AS sources,
       coalesce(hv.run_id, al.run_id) AS run_id
ORDER BY label, hgvs_g38
"""

ANNOTATION_VARIANTS = """
MATCH (a:Annotation)-[r:IMPLICATES]->(v:Variant)
RETURN a.id AS annotation_id, v.hgvs_g38 AS hgvs_g38, v.spdi_g38 AS spdi_g38,
       r.method AS method, r.evidence AS evidence, r.sources AS sources, r.run_id AS run_id
ORDER BY annotation_id, hgvs_g38
"""

ANNOTATION_GENOTYPES = """
MATCH (a:Annotation)-[t:TESTED|COMPARED_TO]->(gt:Genotype)
CALL (gt) {
  OPTIONAL MATCH (gt)-[c:HAS_COPY]->(h:Haplotype)
  WITH c, h ORDER BY h.name
  RETURN [x IN collect(CASE WHEN h IS NULL THEN null ELSE
           {name: h.name, copies: c.copies, min_copies: c.min_copies,
            structure: h.structure} END) WHERE x IS NOT NULL] AS haplotypes
}
CALL (gt) {
  OPTIONAL MATCH (gt)-[c:HAS_COPY]->(x)
  OPTIONAL MATCH (x)-[:HAS_VARIANT]->(member:Variant)
  WITH c, coalesce(member, x) AS v
  WHERE v:Variant
  WITH v.hgvs_g38 AS hgvs, sum(c.copies) AS copies, sum(c.min_copies) AS lower,
       count(c.copies) AS known, count(*) AS n
  ORDER BY hgvs
  RETURN collect({hgvs_g38: hgvs, copies: CASE WHEN known = n THEN copies END,
                  min_copies: lower}) AS variants
}
RETURN a.id AS annotation_id, type(t) AS role, gt.name AS genotype,
       coalesce(t.strand_assumed, false) AS strand_assumed,
       gt.locus AS locus, gt.zygosity AS zygosity, gt.status AS status,
       gt.reason AS reason, gt.hgvs_g38 AS hgvs_g38, gt.ref_copies AS ref_copies,
       gt.gene_copies AS gene_copies, gt.gene_copies_min AS gene_copies_min,
       haplotypes, variants
ORDER BY annotation_id, role, genotype
"""

DRUG_STRUCT = pl.Struct(
    {"name": pl.String, "cid": pl.Int64, "chembl_id": pl.String, "inchikey": pl.String}
)
ANNOTATION_ROW_SCHEMA: dict[str, pl.DataType] = {
    "annotation_id": pl.String(),
    "source": pl.String(),
    "pmid": pl.String(),
    "drugs": pl.List(DRUG_STRUCT),
    "genes": pl.List(pl.String),
    "genotypes": pl.List(pl.String),
    "implicated_hgvs": pl.List(pl.String),
    "alleles": pl.String(),
    "comparison_alleles": pl.String(),
    "phenotype_categories": pl.List(pl.String),
    "direction": pl.String(),
    "significance": pl.String(),
    "associated": pl.Boolean(),
}
ALLELE_VARIANT_SCHEMA: dict[str, pl.DataType] = {
    "label": pl.String(),
    "kind": pl.String(),
    "status": pl.String(),
    "reason": pl.String(),
    "hgvs_g38": pl.String(),
    "spdi_g38": pl.String(),
    "path": pl.String(),
    "relation": pl.String(),
    "sources": pl.List(pl.String),
    "run_id": pl.String(),
}
ANNOTATION_VARIANT_SCHEMA: dict[str, pl.DataType] = {
    "annotation_id": pl.String(),
    "hgvs_g38": pl.String(),
    "spdi_g38": pl.String(),
    "method": pl.String(),
    "evidence": pl.String(),
    "sources": pl.List(pl.String),
    "run_id": pl.String(),
}
ANNOTATION_GENOTYPE_SCHEMA: dict[str, pl.DataType] = {
    "annotation_id": pl.String(),
    "role": pl.String(),
    "genotype": pl.String(),
    "strand_assumed": pl.Boolean(),
    "locus": pl.String(),
    "zygosity": pl.String(),
    "status": pl.String(),
    "reason": pl.String(),
    "hgvs_g38": pl.String(),
    "ref_copies": pl.Int64(),
    "gene_copies": pl.Int64(),
    "gene_copies_min": pl.Int64(),
    "haplotypes": pl.List(
        pl.Struct(
            {
                "name": pl.String,
                "copies": pl.Int64,
                "min_copies": pl.Int64,
                "structure": pl.String,
            }
        )
    ),
    "variants": pl.List(
        pl.Struct({"hgvs_g38": pl.String, "copies": pl.Int64, "min_copies": pl.Int64})
    ),
}


def annotation_rows(
    graph: KnowledgeGraph, sources: list[str] | None = None
) -> pl.DataFrame:
    """Annotation rows; ``sources`` restricts to PharmGKB files (``var_drug_ann``…)."""
    rows = graph.run(ANNOTATION_ROWS, sources=sources)
    return pl.DataFrame(rows, schema=ANNOTATION_ROW_SCHEMA)


def allele_variants(graph: KnowledgeGraph) -> pl.DataFrame:
    return pl.DataFrame(graph.run(ALLELE_VARIANTS), schema=ALLELE_VARIANT_SCHEMA)


def annotation_variants(graph: KnowledgeGraph) -> pl.DataFrame:
    return pl.DataFrame(
        graph.run(ANNOTATION_VARIANTS), schema=ANNOTATION_VARIANT_SCHEMA
    )


def annotation_genotypes(graph: KnowledgeGraph) -> pl.DataFrame:
    return pl.DataFrame(
        graph.run(ANNOTATION_GENOTYPES), schema=ANNOTATION_GENOTYPE_SCHEMA
    )


def export_tables(graph: KnowledgeGraph, out_dir: Path) -> dict[str, int]:
    """Write the four tables to ``out_dir``; returns rows per file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        ANNOTATION_ROWS_FILE: annotation_rows(graph),
        ALLELE_VARIANTS_FILE: allele_variants(graph),
        ANNOTATION_VARIANTS_FILE: annotation_variants(graph),
        ANNOTATION_GENOTYPES_FILE: annotation_genotypes(graph),
    }
    counts: dict[str, int] = {}
    for name, frame in tables.items():
        path = out_dir / name
        tmp = path.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp)
        tmp.replace(path)
        counts[name] = frame.height
        logger.info("Exported %d rows to %s", frame.height, path)
    return counts


__all__ = [
    "ALLELE_VARIANTS_FILE",
    "ANNOTATION_GENOTYPES_FILE",
    "ANNOTATION_GENOTYPE_SCHEMA",
    "ANNOTATION_ROWS_FILE",
    "ANNOTATION_ROW_SCHEMA",
    "ANNOTATION_VARIANTS_FILE",
    "allele_variants",
    "annotation_genotypes",
    "annotation_rows",
    "annotation_variants",
    "export_tables",
]
