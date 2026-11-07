"""Pharmacogenomic views for ``inspect_graph.py`` (not part of the package).

* :func:`lookup_attempts` -- which node an identifier names: an rsID, a GRCh38
  HGVS or SPDI, a star allele, a genotype (``CYP2D6*1/*4``, ``rs9923231:C/T``)
  or a PharmGKB annotation id.
* :data:`VIEWS` -- the multi-hop tables shown under ``node`` for each label of
  the pgx layer (Gene, Variant, RefSNP, Haplotype, Genotype, Annotation, Drug).
* :func:`overview` -- ``inspect_graph.py pgx``: how much of the layer resolved,
  and where it did not.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import polars as pl
from _common import kv, note, show, subsection

if TYPE_CHECKING:
    from collections.abc import Callable

    from modephere.kg.graph import KnowledgeGraph

#: ``(label, property, value)``; a list property (``rsids``) matches by membership.
Attempt = tuple[str, str, str]
LIST_KEYS = frozenset({"rsids", "star_alleles"})
PGX_LABELS = ("Gene", "Variant", "RefSNP", "Haplotype", "Allele", "Genotype")

_RSID = re.compile(r"^rs\d+$", re.IGNORECASE)
_SPDI = re.compile(r"^N[CTGW]_\d+\.\d+:\d+:[ACGTN]*:[ACGTN]*$", re.IGNORECASE)
_HGVS = re.compile(r"^N[CTGW]_\d+\.\d+:[gm]\.", re.IGNORECASE)
_SITE_GENOTYPE = re.compile(r"^rs\d+:", re.IGNORECASE)


def _is_genotype(v: str) -> bool:
    return bool(_SITE_GENOTYPE.match(v)) or ("/" in v and "*" in v)


#: (does the value look like it, the ``(label, property)`` pairs to try).
_SHAPES: tuple[tuple[Callable[[str], object], tuple[tuple[str, str], ...]], ...] = (
    (_RSID.match, (("RefSNP", "rsid"), ("Allele", "label"))),
    (_SPDI.match, (("Variant", "spdi_g38"),)),
    (
        _HGVS.match,
        (("Variant", "hgvs_g38"), ("Haplotype", "hgvs_g38"), ("Genotype", "hgvs_g38")),
    ),
    (_is_genotype, (("Genotype", "name"),)),
    (
        lambda v: "*" in v,
        (("Haplotype", "name"), ("Allele", "label"), ("Genotype", "name")),
    ),
    (str.isdigit, (("Annotation", "id"),)),
)


def lookup_attempts(value: str) -> list[Attempt] | None:
    """Nodes a pgx identifier may name, most likely first; None if it is not one."""
    v = value.strip()
    if _RSID.match(v):
        v = v.lower()
    for looks_like, targets in _SHAPES:
        if looks_like(v):
            return [(label, key, v) for label, key in targets]
    return None


def variant_attempts(value: str) -> list[Attempt]:
    """``--label Variant``: by rsID or star allele (metadata), SPDI, else HGVS."""
    v = value.strip()
    if _RSID.match(v):
        return [("Variant", "rsids", v.lower())]
    if "*" in v:
        return [("Variant", "star_alleles", v)]
    if _SPDI.match(v):
        return [("Variant", "spdi_g38", v)]
    return [("Variant", "hgvs_g38", v)]


# --------------------------------------------------------------------------- #
# Annotations reached from a node, and by which route
# --------------------------------------------------------------------------- #

#: ``{start}`` binds the node; each route returns ``a`` (Annotation), ``via``
#: (how it is reached) and ``through`` (the node in between).
_ROUTES = """
{start}
CALL (n) {{
  {routes}
}}
OPTIONAL MATCH (a)-[:FOR_DRUG]->(d:Drug)
WITH a, via, through, collect(DISTINCT d.name) AS drugs
RETURN a.id AS annotation, via, through, drugs, a.significance AS significant,
       a.direction AS direction, a.pmid AS pmid
ORDER BY via, significant DESC, annotation
"""
_LABEL_ROUTE = (
    "MATCH (a:Annotation)-[:CONCERNS_ALLELE]->(al:Allele)-[:REFERS_TO]->(n) "
    "RETURN a, 'allele label' AS via, al.label AS through"
)


def _annotation_routes(
    graph: KnowledgeGraph, start: str, routes: list[str], limit: int, **params: Any
) -> None:
    """Print the annotations reaching a node, counted by route, then listed."""
    query = _ROUTES.format(start=start, routes="\n  UNION\n  ".join(routes))
    rows = pl.DataFrame(graph.run(query, **params))
    if rows.height == 0:
        subsection("PharmGKB annotations")
        note("none")
        return
    significant = pl.col("annotation").filter(pl.col("significant") == "yes")
    show(
        rows.group_by("via")
        .agg(
            pl.col("annotation").n_unique().alias("annotations"),
            significant.n_unique().alias("significant"),
            pl.col("through").n_unique().alias("through_nodes"),
        )
        .sort("annotations", descending=True),
        20,
        title=f"PharmGKB annotations: {rows['annotation'].n_unique()}, by route",
    )
    per_annotation = (
        rows.group_by("annotation", maintain_order=True)
        .agg(
            pl.col("via").unique(maintain_order=True),
            pl.col("through").drop_nulls().unique(maintain_order=True),
            pl.col("drugs").first(),
            pl.col("significant", "direction", "pmid").first(),
        )
        .sort(pl.col("significant") == "yes", "annotation", descending=[True, False])
    )
    show(per_annotation, limit, title="Annotations, significant first", max_str=40)


# --------------------------------------------------------------------------- #
# Views per label
# --------------------------------------------------------------------------- #

GENE_COUNTS = """
MATCH (g:Gene {symbol: $symbol})
OPTIONAL MATCH (g)<-[:CONCERNS_GENE]-(a:Annotation)
OPTIONAL MATCH (a)-[:FOR_DRUG]->(d:Drug)
RETURN COUNT { (g)<-[:IN_GENE]-(:Variant) } AS variants,
       COUNT { MATCH (h:Haplotype) WHERE h.gene = $symbol } AS haplotypes,
       COUNT { MATCH (h:Haplotype) WHERE h.gene = $symbol
               AND h.hgvs_g38 IS NULL AND h.structure IS NULL } AS haplotypes_without_hgvs,
       count(DISTINCT a) AS annotations, count(DISTINCT d) AS drugs
"""
GENE_RSIDS = """
MATCH (:Gene {symbol: $symbol})<-[:IN_GENE]-(v:Variant)
UNWIND v.rsids AS rsid
RETURN collect(DISTINCT rsid) AS rsids
"""
GENE_GENOTYPES = """
MATCH (gt:Genotype) WHERE gt.locus = $symbol OR gt.locus IN $rsids
RETURN CASE WHEN gt.locus = $symbol THEN 'star' ELSE 'site' END AS locus,
       gt.status AS status, gt.reason AS reason, count(*) AS genotypes,
       sum(COUNT { (gt)<-[:TESTED|COMPARED_TO]-(:Annotation) }) AS mentions
ORDER BY locus, status, genotypes DESC
"""
GENE_HAPLOTYPES = """
MATCH (h:Haplotype) WHERE h.gene = $symbol
OPTIONAL MATCH (h)-[:SUBALLELE_OF]->(p:Haplotype)
RETURN h.name AS haplotype, h.is_reference AS reference, h.structure AS structure,
       COUNT { (h)-[:HAS_VARIANT]->() } AS variants, p.name AS suballele_of,
       COUNT { (h)<-[:REFERS_TO]-(:Allele) } AS labels,
       COUNT { (h)<-[:HAS_COPY]-(:Genotype) } AS genotypes,
       coalesce(h.hgvs_g38, h.hgvs_t2t) AS hgvs
ORDER BY genotypes + labels DESC, h.name
LIMIT $limit
"""
GENE_VARIANTS = """
MATCH (:Gene {symbol: $symbol})<-[:IN_GENE]-(v:Variant)
WITH v, COUNT { (v)<-[:IMPLICATES]-(:Annotation) } AS implicated,
     COUNT { (v)<-[:HAS_COPY]-(:Genotype) } AS genotypes,
     COUNT { (v)<-[:HAS_VARIANT]-(:Haplotype) } AS haplotypes
RETURN v.hgvs_g38 AS hgvs_g38, v.rsids AS rsids, implicated, genotypes, haplotypes
ORDER BY implicated + genotypes DESC, v.pos_g38
LIMIT $limit
"""
GENE_DRUGS = """
MATCH (:Gene {symbol: $symbol})<-[:CONCERNS_GENE]-(a:Annotation)-[:FOR_DRUG]->(d:Drug)
RETURN d.name AS drug, count(DISTINCT a) AS annotations,
       count(DISTINCT CASE WHEN a.significance = 'yes' THEN a END) AS significant
ORDER BY annotations DESC, drug
LIMIT $limit
"""
GENE_UNRESOLVED_LABELS = """
MATCH (:Gene {symbol: $symbol})<-[:CONCERNS_GENE]-(a:Annotation)
      -[:CONCERNS_ALLELE]->(al:Allele)
WHERE al.status <> 'resolved'
RETURN al.label AS label, al.kind AS kind, al.reason AS reason,
       count(DISTINCT a) AS annotations
ORDER BY annotations DESC, label
LIMIT $limit
"""


def gene_view(graph: KnowledgeGraph, props: dict[str, Any], limit: int) -> None:
    symbol = props["symbol"]
    counts = graph.run(GENE_COUNTS, symbol=symbol)
    if not counts or not any(counts[0].values()):
        return  # a ChEMBL-only gene: nothing pharmacogenomic to show
    subsection("Pharmacogenomic layer")
    kv(counts[0], skip_none=False)
    rsids = graph.run(GENE_RSIDS, symbol=symbol)
    show(
        pl.DataFrame(
            graph.run(
                GENE_GENOTYPES, symbol=symbol, rsids=rsids[0]["rsids"] if rsids else []
            )
        ),
        50,
        title="Genotypes at this gene (star = diplotypes, site = rsID genotypes)",
        max_str=40,
    )
    show(
        pl.DataFrame(graph.run(GENE_HAPLOTYPES, symbol=symbol, limit=limit)),
        limit,
        title="Haplotypes, most used first",
        max_str=30,
    )
    show(
        pl.DataFrame(graph.run(GENE_VARIANTS, symbol=symbol, limit=limit)),
        limit,
        title="Variants, most annotated first",
        max_str=44,
    )
    show(
        pl.DataFrame(graph.run(GENE_DRUGS, symbol=symbol, limit=limit)),
        limit,
        title="Drugs with annotations on this gene",
    )
    show(
        pl.DataFrame(graph.run(GENE_UNRESOLVED_LABELS, symbol=symbol, limit=limit)),
        limit,
        title="Unresolved allele labels in this gene's annotations",
        max_str=44,
    )


VARIANT_GENOTYPES = """
MATCH (v:Variant {hgvs_g38: $hgvs})<-[c:HAS_COPY]-(gt:Genotype)
RETURN gt.name AS genotype, c.copies AS copies, c.min_copies AS min_copies,
       gt.zygosity AS zygosity, gt.hgvs_g38 AS hgvs_g38,
       COUNT { (gt)<-[:TESTED|COMPARED_TO]-(:Annotation) } AS mentions
ORDER BY mentions DESC, genotype
"""


def variant_view(graph: KnowledgeGraph, props: dict[str, Any], limit: int) -> None:
    hgvs = props["hgvs_g38"]
    show(
        pl.DataFrame(graph.run(VARIANT_GENOTYPES, hgvs=hgvs)),
        limit,
        title="Genotypes carrying this variant",
        max_str=44,
    )
    _annotation_routes(
        graph,
        "MATCH (n:Variant {hgvs_g38: $hgvs})",
        [
            "MATCH (a:Annotation)-[:IMPLICATES]->(n) "
            "RETURN a, 'implicates' AS via, null AS through",
            "MATCH (a:Annotation)-[r:TESTED|COMPARED_TO]->(gt:Genotype)-[:HAS_COPY]->(n) "
            "RETURN a, toLower(type(r)) + ' genotype' AS via, gt.name AS through",
            "MATCH (a:Annotation)-[:CONCERNS_ALLELE]->(:Allele)-[:REFERS_TO]->"
            "(h:Haplotype)-[:HAS_VARIANT]->(n) "
            "RETURN a, 'star allele label' AS via, h.name AS through",
            "MATCH (a:Annotation)-[r:TESTED|COMPARED_TO]->(:Genotype)-[:HAS_COPY]->"
            "(h:Haplotype)-[:HAS_VARIANT]->(n) "
            "RETURN a, toLower(type(r)) + ' diplotype' AS via, h.name AS through",
        ],
        limit,
        hgvs=hgvs,
    )


REFSNP_VARIANTS = """
MATCH (:RefSNP {rsid: $rsid})-[:HAS_ALLELE]->(v:Variant)
OPTIONAL MATCH (v)-[:IN_GENE]->(g:Gene)
WITH v, collect(g.symbol) AS genes
RETURN v.hgvs_g38 AS hgvs_g38, genes,
       COUNT { (v)<-[:IMPLICATES]-(:Annotation) } AS implicated,
       COUNT { (v)<-[:HAS_COPY]-(:Genotype) } AS genotypes,
       v.star_alleles AS star_alleles
ORDER BY v.pos_g38, hgvs_g38
"""
SITE_GENOTYPES = """
MATCH (gt:Genotype {locus: $rsid})
RETURN gt.name AS genotype, gt.status AS status, gt.reason AS reason,
       gt.zygosity AS zygosity, gt.hgvs_g38 AS hgvs_g38,
       COUNT { (gt)<-[:TESTED|COMPARED_TO]-(:Annotation) } AS mentions
ORDER BY mentions DESC, genotype
"""


def refsnp_view(graph: KnowledgeGraph, props: dict[str, Any], limit: int) -> None:
    rsid = props["rsid"]
    show(
        pl.DataFrame(graph.run(REFSNP_VARIANTS, rsid=rsid)),
        limit,
        title="Variants (alleles) of this RefSNP, by HGVS",
        max_str=44,
    )
    show(
        pl.DataFrame(graph.run(SITE_GENOTYPES, rsid=rsid)),
        limit,
        title="Genotypes at this site",
        max_str=44,
    )
    _annotation_routes(
        graph,
        "MATCH (n:RefSNP {rsid: $rsid})",
        [
            _LABEL_ROUTE,
            "MATCH (a:Annotation)-[r:TESTED|COMPARED_TO]->(gt:Genotype {locus: n.rsid}) "
            "RETURN a, toLower(type(r)) + ' genotype' AS via, gt.name AS through",
        ],
        limit,
        rsid=rsid,
    )


HAPLOTYPE_VARIANTS = """
MATCH (:Haplotype {name: $name})-[:HAS_VARIANT]->(v:Variant)
OPTIONAL MATCH (v)-[:IN_GENE]->(g:Gene)
WITH v, collect(g.symbol) AS genes
RETURN v.hgvs_g38 AS hgvs_g38, v.rsids AS rsids, genes,
       size(v.star_alleles) AS in_star_alleles
ORDER BY v.pos_g38, hgvs_g38
"""
HAPLOTYPE_GENOTYPES = """
MATCH (:Haplotype {name: $name})<-[c:HAS_COPY]-(gt:Genotype)
RETURN gt.name AS genotype, c.copies AS copies, c.min_copies AS min_copies,
       gt.zygosity AS zygosity, gt.gene_copies AS gene_copies,
       COUNT { (gt)<-[:TESTED|COMPARED_TO]-(:Annotation) } AS mentions
ORDER BY mentions DESC, genotype
"""


def haplotype_view(graph: KnowledgeGraph, props: dict[str, Any], limit: int) -> None:
    name = props["name"]
    if props.get("structure"):
        note(f"structural allele ({props['structure']}): no small-variant definition")
    else:
        show(
            pl.DataFrame(graph.run(HAPLOTYPE_VARIANTS, name=name)),
            200,
            title="Defining variants, in genomic order",
            max_str=44,
        )
    show(
        pl.DataFrame(graph.run(HAPLOTYPE_GENOTYPES, name=name)),
        limit,
        title="Genotypes with copies of this haplotype",
        max_str=44,
    )
    _annotation_routes(
        graph,
        "MATCH (n:Haplotype {name: $name})",
        [
            _LABEL_ROUTE,
            "MATCH (a:Annotation)-[r:TESTED|COMPARED_TO]->(gt:Genotype)-[:HAS_COPY]->(n) "
            "RETURN a, toLower(type(r)) + ' genotype' AS via, gt.name AS through",
        ],
        limit,
        name=name,
    )


GENOTYPE_COPIES = """
MATCH (:Genotype {name: $name})-[c:HAS_COPY]->(x)
RETURN labels(x)[0] AS kind, coalesce(x.name, x.hgvs_g38) AS copy_of,
       c.copies AS copies, c.min_copies AS min_copies, x.structure AS structure,
       coalesce(x.hgvs_g38, x.hgvs_t2t) AS hgvs
ORDER BY kind, copy_of
"""
GENOTYPE_SIBLINGS = """
MATCH (o:Genotype {locus: $locus}) WHERE o.name <> $name
RETURN o.name AS genotype, o.status AS status, o.reason AS reason,
       o.zygosity AS zygosity,
       COUNT { (o)<-[:TESTED|COMPARED_TO]-(:Annotation) } AS mentions
ORDER BY mentions DESC, genotype
LIMIT $limit
"""


def genotype_view(graph: KnowledgeGraph, props: dict[str, Any], limit: int) -> None:
    name = props["name"]
    if props.get("status") != "resolved":
        note(f"unresolved: {props.get('reason')}")
    show(
        pl.DataFrame(graph.run(GENOTYPE_COPIES, name=name)),
        50,
        title="Copies (what the two chromosomes carry)",
        max_str=44,
    )
    _annotation_routes(
        graph,
        "MATCH (n:Genotype {name: $name})",
        [
            "MATCH (a:Annotation)-[r:TESTED|COMPARED_TO]->(n) "
            "RETURN a, toLower(type(r)) + CASE WHEN r.strand_assumed "
            "THEN ' (strand assumed)' ELSE '' END AS via, r.evidence AS through",
        ],
        limit,
        name=name,
    )
    if props.get("locus"):
        show(
            pl.DataFrame(
                graph.run(
                    GENOTYPE_SIBLINGS, locus=props["locus"], name=name, limit=limit
                )
            ),
            limit,
            title=f"Other genotypes at locus {props['locus']}",
            max_str=44,
        )


ANNOTATION_ALLELES = """
MATCH (:Annotation {id: $id})-[:CONCERNS_ALLELE]->(al:Allele)
OPTIONAL MATCH (al)-[r:REFERS_TO]->(t)
RETURN al.label AS label, al.kind AS kind, al.status AS status, al.reason AS reason,
       collect(coalesce(t.name, t.rsid)) AS refers_to,
       collect(DISTINCT r.method) AS method
ORDER BY label
"""
ANNOTATION_GENOTYPES = """
MATCH (:Annotation {id: $id})-[r:TESTED|COMPARED_TO]->(gt:Genotype)
OPTIONAL MATCH (gt)-[c:HAS_COPY]->(x)
WITH r, gt, collect(coalesce(x.name, x.hgvs_g38) + ' x'
     + coalesce(toString(c.copies), '>=' + toString(c.min_copies))) AS copies
RETURN toLower(type(r)) AS role, gt.name AS genotype, gt.status AS status,
       gt.reason AS reason, r.strand_assumed AS strand_assumed,
       gt.hgvs_g38 AS hgvs_g38, copies
ORDER BY role DESC, genotype
"""


def annotation_view(graph: KnowledgeGraph, props: dict[str, Any], limit: int) -> None:
    show(
        pl.DataFrame(graph.run(ANNOTATION_ALLELES, id=props["id"])),
        limit,
        title="Allele labels and what they resolve to",
        max_str=40,
    )
    show(
        pl.DataFrame(graph.run(ANNOTATION_GENOTYPES, id=props["id"])),
        limit,
        title="Genotypes tested / compared to",
        max_str=44,
    )


DRUG_GENES = """
MATCH (:Drug {name: $name})<-[:FOR_DRUG]-(a:Annotation)
OPTIONAL MATCH (a)-[:CONCERNS_GENE]->(g:Gene)
WITH coalesce(g.symbol, '-') AS gene, collect(DISTINCT a) AS anns
RETURN gene, size(anns) AS annotations,
       size([x IN anns WHERE x.significance = 'yes']) AS significant,
       size([x IN anns WHERE x.direction IS NOT NULL]) AS with_direction
ORDER BY annotations DESC, gene
LIMIT $limit
"""
DRUG_GENOTYPES = """
MATCH (:Drug {name: $name})<-[:FOR_DRUG]-(a:Annotation)-[r:TESTED|COMPARED_TO]->(gt:Genotype)
RETURN toLower(type(r)) AS role, gt.status AS status, gt.reason AS reason,
       count(*) AS mentions, count(DISTINCT gt) AS genotypes
ORDER BY role DESC, status, mentions DESC
"""


def drug_view(graph: KnowledgeGraph, props: dict[str, Any], limit: int) -> None:
    name = props["name"]
    show(
        pl.DataFrame(graph.run(DRUG_GENES, name=name, limit=limit)),
        limit,
        title="Pharmacogenes (annotations per gene)",
    )
    show(
        pl.DataFrame(graph.run(DRUG_GENOTYPES, name=name)),
        50,
        title="Genotype mentions in this drug's annotations",
        max_str=40,
    )


COMPOUND_DRUGS = """
MATCH (:Compound {chembl_id: $id})-[:SAME_AS]-(d:Drug)
RETURN d.name AS name
"""


def compound_view(graph: KnowledgeGraph, props: dict[str, Any], limit: int) -> None:
    """A ChEMBL compound with a PharmGKB drug twin gets that drug's pgx tables."""
    for row in graph.run(COMPOUND_DRUGS, id=props["chembl_id"]):
        subsection(f"PharmGKB drug {row['name']!r} (SAME_AS)")
        drug_view(graph, row, limit)


VIEWS: dict[str, Callable[[KnowledgeGraph, dict[str, Any], int], None]] = {
    "Gene": gene_view,
    "Variant": variant_view,
    "RefSNP": refsnp_view,
    "Haplotype": haplotype_view,
    "Genotype": genotype_view,
    "Annotation": annotation_view,
    "Drug": drug_view,
    "Compound": compound_view,
}


# --------------------------------------------------------------------------- #
# Layer overview (inspect_graph.py pgx)
# --------------------------------------------------------------------------- #

LAYER_RUN = """
MATCH (run:IngestRun {layer: 'pgx'})
OPTIONAL MATCH (run)-[:USED]->(d:Dataset)
RETURN left(run.id, 12) AS run_id, left(toString(run.started_at), 19) AS started_at,
       run.code_version AS code_version, count(d) AS datasets
ORDER BY started_at DESC
"""
LABEL_COUNT = "MATCH (n:{label}) RETURN count(n) AS n"
VARIANT_COVERAGE = """
MATCH (v:Variant)
RETURN count(v) AS variants,
       count(CASE WHEN size(v.rsids) > 0 THEN 1 END) AS with_rsid,
       count(CASE WHEN size(v.star_alleles) > 0 THEN 1 END) AS in_star_allele,
       count(CASE WHEN EXISTS { (v)-[:IN_GENE]->() } THEN 1 END) AS in_gene,
       count(CASE WHEN EXISTS { (v)<-[:IMPLICATES]-() } THEN 1 END) AS implicated,
       count(CASE WHEN EXISTS { (v)<-[:HAS_COPY]-() } THEN 1 END) AS in_genotype
"""
HAPLOTYPE_COVERAGE = """
MATCH (h:Haplotype)
RETURN coalesce(h.structure, 'small variants') AS structure, count(*) AS haplotypes,
       count(h.hgvs_g38) AS with_hgvs_g38, count(h.hgvs_t2t) AS with_hgvs_t2t,
       count(CASE WHEN h.is_reference THEN 1 END) AS reference
ORDER BY haplotypes DESC
"""
ALLELE_STATUS = """
MATCH (al:Allele)
RETURN al.kind AS kind, al.status AS status, al.reason AS reason, count(*) AS labels,
       sum(COUNT { (al)<-[:CONCERNS_ALLELE]-() }) AS mentions
ORDER BY kind, status, labels DESC
"""
GENOTYPE_STATUS = """
MATCH (gt:Genotype)
RETURN CASE WHEN gt.locus STARTS WITH 'rs' THEN 'site' ELSE 'star' END AS locus,
       gt.status AS status, gt.reason AS reason, count(*) AS genotypes,
       sum(COUNT { (gt)<-[:TESTED|COMPARED_TO]-() }) AS mentions
ORDER BY locus, status, genotypes DESC
"""
STRAND = """
MATCH ()-[r:TESTED|COMPARED_TO]->()
RETURN count(r) AS genotype_edges, count(r.strand_assumed) AS strand_assumed
"""
TOP_GENES = """
MATCH (g:Gene)<-[:CONCERNS_GENE]-(a:Annotation)
WITH g, count(a) AS annotations ORDER BY annotations DESC LIMIT $limit
RETURN g.symbol AS gene, annotations,
       COUNT { (g)<-[:IN_GENE]-(:Variant) } AS variants,
       COUNT { MATCH (h:Haplotype) WHERE h.gene = g.symbol } AS haplotypes,
       COUNT { MATCH (gt:Genotype {locus: g.symbol}) WHERE gt.status = 'resolved' }
         AS diplotypes_resolved,
       COUNT { MATCH (gt:Genotype {locus: g.symbol}) WHERE gt.status <> 'resolved' }
         AS diplotypes_unresolved
ORDER BY annotations DESC
"""


def overview(graph: KnowledgeGraph, limit: int) -> None:
    show(pl.DataFrame(graph.run(LAYER_RUN)), 10, title="pgx ingest runs, newest first")
    subsection("Nodes")
    counts = {
        label: graph.run(LABEL_COUNT.format(label=label))[0]["n"]
        for label in (*PGX_LABELS, "Annotation", "Drug")
    }
    kv(counts, skip_none=False)
    subsection("Variants (keyed by hgvs_g38)")
    kv(graph.run(VARIANT_COVERAGE)[0], skip_none=False)
    show(pl.DataFrame(graph.run(HAPLOTYPE_COVERAGE)), 20, title="Haplotypes")
    show(
        pl.DataFrame(graph.run(ALLELE_STATUS)),
        100,
        title="PharmGKB allele labels, by kind and status",
        max_str=40,
    )
    show(
        pl.DataFrame(graph.run(GENOTYPE_STATUS)),
        100,
        title="Genotypes (star = diplotypes, site = rsID genotypes)",
        max_str=40,
    )
    subsection("Genotype edges")
    kv(graph.run(STRAND)[0], skip_none=False)
    show(
        pl.DataFrame(graph.run(TOP_GENES, limit=limit)),
        limit,
        title=f"Top {limit} genes by annotations",
    )
    note("every unresolved case of the last ingest: data/pgx/reports/unresolved.log")


__all__ = [
    "LIST_KEYS",
    "VIEWS",
    "Attempt",
    "lookup_attempts",
    "overview",
    "variant_attempts",
]
