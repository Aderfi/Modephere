"""Inspect the Neo4j knowledge graphs: overall structure, the pgx layer, or one node.

Usage (from the project root)::

    # database structure: labels, relationship patterns, properties, indexes,
    # provenance (datasets / ingest runs) and the provenance audit
    uv run python scripts/inspect_graph.py schema                 # Modephere's graph
    uv run python scripts/inspect_graph.py schema --db pgen       # dev_Pgen's PharmGKB
    uv run python scripts/inspect_graph.py schema --db both

    # one node and everything attached to it
    uv run python scripts/inspect_graph.py node CHEMBL1464        # Compound by ChEMBL ID
    uv run python scripts/inspect_graph.py node warfarin          # Compound by name
    uv run python scripts/inspect_graph.py node PJVWKTKQMONHTI-UHFFFAOYSA-N
    uv run python scripts/inspect_graph.py node VKORC1 --label Gene

    # pharmacogenomic layer: the label is guessed from the identifier
    uv run python scripts/inspect_graph.py pgx                     # resolution overview
    uv run python scripts/inspect_graph.py node CYP2D6              # Gene (no Compound hit)
    uv run python scripts/inspect_graph.py node rs9923231           # RefSNP
    uv run python scripts/inspect_graph.py node NC_000016.10:g.31096368C>T   # Variant
    uv run python scripts/inspect_graph.py node NC_000016.10:31096367:C:T    # Variant (SPDI)
    uv run python scripts/inspect_graph.py node rs9923231 --label Variant    # by metadata
    uv run python scripts/inspect_graph.py node 'CYP2D6*4'          # Haplotype
    uv run python scripts/inspect_graph.py node 'CYP2D6*1/*4'       # Genotype (diplotype)
    uv run python scripts/inspect_graph.py node rs9923231:C/T       # Genotype (site)
    uv run python scripts/inspect_graph.py node 'GSTM1 null'        # Allele label
    uv run python scripts/inspect_graph.py node 1451834452          # Annotation

    uv run python scripts/inspect_graph.py node warfarin --db pgen --label Drug
    uv run python scripts/inspect_graph.py node rs9923231 --db pgen --label RefSNP

Databases: ``modephere`` (bolt://localhost:7688, ``MODEPHERE_KG__*``) and ``pgen``
(bolt://localhost:7687, ``PHARMAGEN_KG__URI`` / ``PHARMAGEN_KG__PASSWORD``).
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import TYPE_CHECKING, Any

import polars as pl
from _common import (
    GraphUnavailable,
    is_inchikey,
    iter_graphs,
    kv,
    note,
    open_graph,
    section,
    show,
    subsection,
)
from _pgx import (
    LIST_KEYS,
    VIEWS,
    Attempt,
    lookup_attempts,
    overview,
    variant_attempts,
)

from modephere.kg.schema import NODE_KEYS

if TYPE_CHECKING:
    from modephere.kg.graph import KnowledgeGraph

#: Node keys of dev_Pgen's graph (model v2), for ``node --db pgen``.
PGEN_KEYS: dict[str, str] = {
    "Drug": "name",
    "Gene": "symbol",
    "Annotation": "id",
    "RefSNP": "rsid",
    "Variant": "spdi",
    "Haplotype": "name",
    "Allele": "label",
    "Dataset": "id",
    "IngestRun": "id",
}
#: Properties tried, in order, to give a neighbour a readable name.
DISPLAY = (
    "pref_name",
    "name",
    "symbol",
    "label",
    "rsid",
    "hgvs_g38",
    "code",
    "mesh_heading",
    "warning_type",
    "chembl_id",
    "accession",
    "inchikey",
    "id",
)
#: Relationship properties shown next to each neighbour (when present).
REL_PROPS = (
    "action_type",
    "pchembl_median",
    "n",
    "max_phase",
    "method",
    "enzyme_name",
    "phase",
    "form",
    "copies",
    "min_copies",
    "strand_assumed",
    "evidence",
)

# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

PATTERNS = """
MATCH (a)-[r]->(b)
RETURN labels(a)[0] AS from, type(r) AS rel, labels(b)[0] AS to, count(*) AS n
ORDER BY from, rel, to
"""
LABELS = """
CALL db.labels() YIELD label
CALL (label) {
  MATCH (n) WHERE label IN labels(n) RETURN count(n) AS n
}
RETURN label, n ORDER BY n DESC
"""
NODE_PROPS = """
MATCH (n) WHERE $label IN labels(n)
WITH n LIMIT $sample
UNWIND keys(n) AS key
RETURN key, count(*) AS present, collect(DISTINCT valueType(n[key]))[0..3] AS types,
       head(collect(n[key])) AS example
ORDER BY present DESC, key
"""
REL_PROPS_Q = """
MATCH ()-[r]->() WHERE type(r) = $type
WITH r LIMIT $sample
UNWIND keys(r) AS key
RETURN key, count(*) AS present, collect(DISTINCT valueType(r[key]))[0..3] AS types,
       head(collect(r[key])) AS example
ORDER BY present DESC, key
"""
CONSTRAINTS = """
SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties
RETURN name, type, labelsOrTypes AS on, properties
ORDER BY name
"""
INDEXES = """
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state, owningConstraint
WHERE owningConstraint IS NULL
RETURN name, type, labelsOrTypes AS on, properties, state
ORDER BY name
"""
DATASETS = """
MATCH (d:Dataset)
OPTIONAL MATCH (run:IngestRun)-[:USED]->(d)
WITH d, count(run) AS runs
RETURN d.id AS id, d.name AS name, d.version AS version, runs
ORDER BY id
"""
RUNS = """
MATCH (run:IngestRun)
OPTIONAL MATCH (run)-[:USED]->(d:Dataset)
WITH run, collect(d.id) AS datasets
RETURN left(run.id, 12) AS run_id, left(toString(run.started_at), 19) AS started_at,
       run.tool AS tool, run.code_version AS code_version, run.dirty AS dirty, datasets
ORDER BY started_at DESC
"""


def _example(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 50 else text[:47] + "..."


def _types(raw: list[str]) -> str:
    """'LIST<STRING NOT NULL> NOT NULL' -> 'LIST<STRING>'; empty lists folded in."""
    types = sorted({t.replace(" NOT NULL", "") for t in raw})
    if len(types) > 1 and "LIST<NOTHING>" in types:
        types.remove("LIST<NOTHING>")
    return ", ".join(types)


def _props_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Property scan rows -> frame; examples of mixed types become strings."""
    return pl.DataFrame(
        [
            {
                "key": r["key"],
                "present": r["present"],
                "types": _types(r["types"]),
                "example": _example(r["example"]),
            }
            for r in rows
        ],
        schema={
            "key": pl.String,
            "present": pl.Int64,
            "types": pl.String,
            "example": pl.String,
        },
    )


def cmd_schema(args: argparse.Namespace) -> int:
    names = ["modephere", "pgen"] if args.db == "both" else [args.db]
    for name, graph in iter_graphs(names):
        section(f"Database '{name}'")
        if graph is None:
            continue
        _schema(graph, args)
    return 0


def _schema(graph: KnowledgeGraph, args: argparse.Namespace) -> None:
    version = graph.run(
        "CALL dbms.components() YIELD name, versions, edition "
        "RETURN name, versions[0] AS version, edition"
    )
    kv(version[0] if version else {})

    labels = pl.DataFrame(graph.run(LABELS))
    rels = pl.DataFrame(graph.run(PATTERNS))
    total_n = int(labels["n"].sum()) if labels.height else 0
    total_r = int(rels["n"].sum()) if rels.height else 0
    print(f"  {total_n:,} label memberships, {total_r:,} relationships")
    show(labels, 100, title="Node labels")
    show(
        rels.group_by("rel").agg(pl.col("n").sum()).sort("n", descending=True),
        100,
        title="Relationship types",
    )
    show(rels, 200, title="Patterns (from)-[rel]->(to)", max_str=40)

    subsection("Schema diagram (text)")
    for row in rels.iter_rows(named=True):
        print(f"  ({row['from']})-[:{row['rel']}]->({row['to']})  x{row['n']:,}")

    if not args.no_properties:
        for label in labels["label"].to_list():
            props = _props_frame(graph.run(NODE_PROPS, label=label, sample=args.sample))
            show(
                props,
                100,
                title=f"Properties of :{label} (sample of {args.sample} nodes)",
                max_str=50,
            )
        for rel in rels["rel"].unique().sort().to_list():
            props = _props_frame(graph.run(REL_PROPS_Q, type=rel, sample=args.sample))
            show(props, 100, title=f"Properties of [:{rel}]", max_str=50)

    show(pl.DataFrame(graph.run(CONSTRAINTS)), 100, title="Constraints", max_str=50)
    show(pl.DataFrame(graph.run(INDEXES)), 100, title="Other indexes", max_str=50)
    show(pl.DataFrame(graph.run(DATASETS)), 100, title="Datasets (provenance)")
    show(
        pl.DataFrame(graph.run(RUNS)),
        30,
        title="Ingest runs, newest first",
        max_str=40,
    )
    subsection("Provenance audit (elements without sources / run_id)")
    kv(graph.audit(), skip_none=False)


# --------------------------------------------------------------------------- #
# node
# --------------------------------------------------------------------------- #

NEIGHBOUR_SUMMARY = """
MATCH (n) WHERE elementId(n) = $id
MATCH (n)-[r]-(m)
RETURN CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS dir,
       type(r) AS rel, labels(m)[0] AS other, count(*) AS n
ORDER BY n DESC
"""
NEIGHBOURS = """
MATCH (n) WHERE elementId(n) = $id
MATCH (n)-[r]-(m) WHERE type(r) = $rel AND $other IN labels(m)
  AND (CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END) = $dir
RETURN properties(m) AS m, properties(r) AS r
ORDER BY coalesce(r.pchembl_median, r.max_phase, -1) DESC
LIMIT $limit
"""
COMPOUND_GENES = """
MATCH (c:Compound {chembl_id: $id})
MATCH (c)-[r:HAS_MECHANISM|HAS_BIOACTIVITY]->(t:Target)-[:HAS_COMPONENT]->(:Protein)
      -[:ENCODED_BY]->(g:Gene)
RETURN g.symbol AS gene, t.chembl_id AS target, t.pref_name AS target_name,
       collect(DISTINCT type(r)) AS via,
       max(r.pchembl_median) AS pchembl_median,
       collect(DISTINCT r.action_type)[0] AS action
ORDER BY ('HAS_MECHANISM' IN via) DESC, pchembl_median DESC
"""
COMPOUND_ATC = """
MATCH (c:Compound {chembl_id: $id})-[:IN_ATC]->(a:ATC)
MATCH path = (a)-[:SUBCLASS_OF*0..]->(top:ATC) WHERE NOT (top)-[:SUBCLASS_OF]->()
RETURN [x IN nodes(path) | x.code + ' ' + coalesce(x.name, '')] AS lineage
"""
GENE_TO_COMPOUNDS = """
MATCH (g:Gene {symbol: $symbol})<-[:ENCODED_BY]-(:Protein)<-[:HAS_COMPONENT]-(t:Target)
MATCH (c:Compound)-[r:HAS_MECHANISM|HAS_BIOACTIVITY]->(t)
RETURN c.chembl_id AS chembl_id, c.pref_name AS compound, c.max_phase AS phase,
       type(r) AS via, r.action_type AS action, r.pchembl_median AS pchembl
ORDER BY (via = 'HAS_MECHANISM') DESC, phase DESC, pchembl DESC
LIMIT $limit
"""
PROVENANCE = """
MATCH (run:IngestRun {id: $run_id})
OPTIONAL MATCH (run)-[:USED]->(d:Dataset)
RETURN properties(run) AS run, collect(d.id) AS datasets
"""


def _display(props: dict[str, Any]) -> str:
    if "code" in props and "name" in props:  # ATC
        return f"{props['code']} {props['name']}"
    if "warning_type" in props:
        return f"{props['warning_type']}: {props.get('warning_class') or '-'}"
    for key in DISPLAY:
        if props.get(key) not in (None, ""):
            return str(props[key])
    return "?"


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _compound_attempts(value: str) -> list[Attempt]:
    if value.upper().startswith("CHEMBL"):
        return [("Compound", "chembl_id", value.upper())]
    if is_inchikey(value):
        return [("Compound", "inchikey", value)]
    return [("Compound", "pref_name", value.upper())]


def _attempts(
    value: str, label: str | None, keys: dict[str, str], db: str
) -> list[Attempt]:
    """Candidate ``(label, property, value)`` lookups, most likely first."""
    if label == "Compound":
        return _compound_attempts(value)
    if label == "Variant" and db == "modephere":
        return variant_attempts(value)
    if label is not None:
        return [(label, keys[label], value)]
    if db != "modephere":
        return [("Drug", keys["Drug"], value)]
    pgx = lookup_attempts(value)
    if pgx is not None:
        return pgx
    return [
        *_compound_attempts(value),
        ("Gene", "symbol", value),
        ("Drug", "name", value),
        ("Allele", "label", value),
    ]


def _match(
    graph: KnowledgeGraph, attempt: Attempt, *, any_case: bool
) -> list[dict[str, Any]]:
    label, key, value = attempt
    if not (_IDENTIFIER.match(label) and _IDENTIFIER.match(key)):
        raise ValueError(f"not a label / property name: {label!r}, {key!r}")
    if key in LIST_KEYS:
        where = f"$value IN n.{key}"
    elif any_case:
        where = f"toLower(toString(n.{key})) = toLower($value)"
    else:
        where = f"n.{key} = $value"
    return graph.run(
        f"MATCH (n:{label}) WHERE {where} RETURN n, elementId(n) AS id LIMIT 6",
        value=value,
    )


def _find(
    graph: KnowledgeGraph,
    value: str,
    label: str | None,
    keys: dict[str, str],
    db: str,
) -> tuple[str, dict[str, Any], str] | None:
    """(label, properties, elementId) of the node, or None (candidates printed).

    Every candidate is tried with an exact match first; only then case-blind.
    """
    if label is not None and label not in keys:
        note(f"unknown label {label!r}; known: {', '.join(keys)}")
        return None
    attempts = _attempts(value, label, keys, db)
    for any_case in (False, True):
        for attempt in attempts:
            rows = _match(graph, attempt, any_case=any_case)
            if not rows:
                continue
            found, key, _ = attempt
            if len(rows) > 1:
                others = ", ".join(_display(dict(r["n"])) for r in rows[1:])
                note(
                    f"{len(rows)}{'+' if len(rows) == 6 else ''} :{found} nodes "
                    f"match {key} = {value!r}; showing the first (others: {others})"
                )
            return found, dict(rows[0]["n"]), rows[0]["id"]
    if any(a[0] == "Compound" for a in attempts):
        hits = graph.search_compounds(value, 10)
        if hits.height:
            show(hits, 10, title=f"No exact match; full-text candidates for {value!r}")
            note("re-run with one of these chembl_id values")
            return None
    tried = ", ".join(f":{lab}.{key}" for lab, key, _ in attempts)
    note(f"no node matches {value!r} (tried {tried})")
    return None


def cmd_node(args: argparse.Namespace) -> int:
    keys = NODE_KEYS if args.db == "modephere" else PGEN_KEYS
    try:
        graph = open_graph(args.db)
    except GraphUnavailable as e:
        note(str(e))
        return 1
    with graph:
        found = _find(graph, args.value, args.label, keys, args.db)
        if found is None:
            return 1
        label, props, element_id = found
        section(f":{label} {_display(props)}  ({args.db} graph)")
        props_shown = {
            k: v for k, v in sorted(props.items()) if k not in ("names_text",)
        }
        kv(props_shown)

        summary = pl.DataFrame(graph.run(NEIGHBOUR_SUMMARY, id=element_id))
        if summary.height == 0:
            note("no relationships")
            return 0
        summary = summary.with_columns(
            pl.when(pl.col("dir") == "out")
            .then(pl.format("(this)-[:{}]->(:{})", "rel", "other"))
            .otherwise(pl.format("(this)<-[:{}]-(:{})", "rel", "other"))
            .alias("pattern")
        )
        show(
            summary.select("pattern", "n"),
            100,
            title=f"Relationships: {int(summary['n'].sum())}",
            max_str=60,
        )

        for row in summary.iter_rows(named=True):
            rows = graph.run(
                NEIGHBOURS,
                id=element_id,
                rel=row["rel"],
                other=row["other"],
                dir=row["dir"],
                limit=args.limit,
            )
            table = pl.DataFrame(
                [
                    {
                        "neighbour": _display(r["m"]),
                        **{k: _cell(r["r"].get(k)) for k in REL_PROPS},
                    }
                    for r in rows
                ]
            )
            table = table.select(
                c for c in table.columns if table[c].null_count() < table.height
            )
            show(
                table,
                args.limit,
                title=f"{row['pattern']}: {row['n']}"
                + (f" (first {args.limit})" if row["n"] > args.limit else ""),
            )

        if label == "Compound":
            show(
                pl.DataFrame(graph.run(COMPOUND_GENES, id=props["chembl_id"])),
                args.limit,
                title="Two hops: Compound -> Target -> Protein -> Gene",
            )
            lineage = graph.run(COMPOUND_ATC, id=props["chembl_id"])
            subsection("ATC lineage")
            for r in lineage:
                print("  " + "  <  ".join(r["lineage"]))
            if not lineage:
                note("no ATC code")
        if label == "Gene" and args.db == "modephere":
            show(
                pl.DataFrame(
                    graph.run(
                        GENE_TO_COMPOUNDS, symbol=props["symbol"], limit=args.limit
                    )
                ),
                args.limit,
                title="Compounds acting on this gene's targets",
            )
        if args.db == "modephere" and label in VIEWS:
            VIEWS[label](graph, props, args.limit)

        if props.get("run_id"):
            prov = graph.run(PROVENANCE, run_id=props["run_id"])
            subsection("Provenance")
            kv({"sources": props.get("sources"), "run_id": props["run_id"]})
            if prov:
                kv({f"run.{k}": v for k, v in sorted(prov[0]["run"].items())})
                kv({"run used datasets": prov[0]["datasets"]})
    return 0


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return value if value is None else str(value)


# --------------------------------------------------------------------------- #
# pgx
# --------------------------------------------------------------------------- #


def cmd_pgx(args: argparse.Namespace) -> int:
    try:
        graph = open_graph("modephere")
    except GraphUnavailable as e:
        note(str(e))
        return 1
    with graph:
        section("Pharmacogenomic layer (modephere graph)")
        overview(graph, args.limit)
    return 0


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema", help="Structure of a whole database.")
    schema.add_argument(
        "--db", choices=["modephere", "pgen", "both"], default="modephere"
    )
    schema.add_argument(
        "--sample",
        type=int,
        default=2000,
        help="Nodes / rels sampled per property scan.",
    )
    schema.add_argument(
        "--no-properties", action="store_true", help="Skip the per-label property scan."
    )
    pgx = sub.add_parser(
        "pgx", help="Pharmacogenomic layer: coverage and what did not resolve."
    )
    pgx.add_argument("--limit", type=int, default=20, help="Genes in the top list.")
    node = sub.add_parser("node", help="One node and its neighbourhood.")
    node.add_argument(
        "value",
        help="ChEMBL ID, name, InChIKey, gene symbol, rsID, HGVS, SPDI, star allele, "
        "genotype or annotation id.",
    )
    node.add_argument("--db", choices=["modephere", "pgen"], default="modephere")
    node.add_argument(
        "--label",
        help="Node label (default: guessed from the value on modephere, Drug on pgen). "
        f"Modephere: {', '.join(NODE_KEYS)}. pgen: {', '.join(PGEN_KEYS)}.",
    )
    node.add_argument("--limit", type=int, default=10, help="Neighbours per type.")
    args = parser.parse_args(argv)
    commands = {"schema": cmd_schema, "pgx": cmd_pgx, "node": cmd_node}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
