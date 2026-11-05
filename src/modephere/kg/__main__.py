"""CLI for the knowledge graph.

Usage::

    docker compose -f docker-compose.kg.yml up -d
    python -m modephere.kg load-chembl            # latest staged release
    python -m modephere.kg load-chembl --release chembl_37
    python -m modephere.kg stats
    python -m modephere.kg audit                  # exit 2 if provenance is missing
    python -m modephere.kg search "gleevec"
    python -m modephere.kg remove-source chembl_37 --yes   # withdraw a release
    python -m modephere.kg link-library                     # data/library -> graph
    # dev_Pgen's PharmGKB drugs (its Neo4j, password PHARMAGEN_KG__PASSWORD)
    python -m modephere.kg link-drugs --drugs-uri bolt://localhost:7687
    python -m modephere.kg link-drugs --drugs-file ../dev_Pgen/data/drugs_cid.tsv
    python -m modephere.kg link-drugs                       # Drug nodes already here

Connection settings: ``[kg]`` in settings.toml, ``MODEPHERE_KG__*`` variables
(``MODEPHERE_KG__PASSWORD``, ``MODEPHERE_KG__URI``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from modephere.chembl.__main__ import _latest
from modephere.chembl.lookup import ChemblLookup
from modephere.config import PROJECT_ROOT, KGSettings, get_settings
from modephere.core import ModephereException, setup_logging
from modephere.kg.chembl import load_chembl
from modephere.kg.graph import KnowledgeGraph
from modephere.kg.links import (
    drugs_from_file,
    drugs_from_graph,
    link_drugs,
    link_library,
)
from modephere.library.ingest.pubchem import PubChemClient

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modephere.kg", description="Modephere knowledge graph."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    load = sub.add_parser("load-chembl", help="Merge a staged ChEMBL release.")
    load.add_argument("--release", help="Staging directory name, e.g. chembl_37.")
    load.add_argument("--staging", type=Path, help="Staging root (default: settings).")
    remove = sub.add_parser(
        "remove-source", help="Delete what only this dataset supports."
    )
    remove.add_argument("dataset", help="Dataset id, e.g. chembl_37.")
    remove.add_argument("--yes", action="store_true", help="Required: confirms.")
    lib = sub.add_parser("link-library", help="Library compounds -> ChEMBL.")
    lib.add_argument("--library", type=Path, help="Library root (default: settings).")
    lib.add_argument(
        "--dataset-id", help="Dataset id (default: modephere_library:<dir>)."
    )
    drugs = sub.add_parser("link-drugs", help="dev_Pgen Drug nodes -> ChEMBL.")
    src = drugs.add_mutually_exclusive_group()
    src.add_argument("--drugs-file", type=Path, help="tsv/csv/parquet: name[, cid].")
    src.add_argument("--drugs-uri", help="Neo4j holding the :Drug nodes (dev_Pgen).")
    drugs.add_argument("--drugs-user", default="neo4j")
    drugs.add_argument(
        "--drugs-password",
        default=os.environ.get("PHARMAGEN_KG__PASSWORD", "pharmagen-dev"),
        help="Default: $PHARMAGEN_KG__PASSWORD or dev_Pgen's dev password.",
    )
    drugs.add_argument("--release", help="Staged ChEMBL release (default: latest).")
    drugs.add_argument(
        "--offline",
        action="store_true",
        help="CID lookups from the PubChem cache only.",
    )
    sub.add_parser("stats", help="Node and relationship counts.")
    sub.add_parser("audit", help="Elements without provenance.")
    search = sub.add_parser("search", help="Full-text compound name search.")
    search.add_argument("text")
    search.add_argument("--limit", type=int, default=10)
    for p in sub.choices.values():
        p.add_argument("--verbose", "-v", action="store_true")
    return parser


def _load_chembl(graph: KnowledgeGraph, args: argparse.Namespace) -> int:
    root = args.staging or get_settings().paths.chembl
    staging = root / args.release if args.release else _latest(root)
    summary = load_chembl(graph, staging, repo=PROJECT_ROOT)
    logger.info("Loaded %s (run %s)", summary.dataset, summary.run_id)
    return 0


def _remove_source(graph: KnowledgeGraph, args: argparse.Namespace) -> int:
    if not args.yes:
        logger.error("remove-source deletes data: add --yes to confirm")
        return 1
    print(graph.remove_source(args.dataset))
    return 0


def _link_library(graph: KnowledgeGraph, args: argparse.Namespace) -> int:
    root = args.library or get_settings().paths.library
    s = link_library(graph, root, dataset_id=args.dataset_id, repo=PROJECT_ROOT)
    logger.info(
        "%s: %d compounds, %d linked to ChEMBL, %d without a ChEMBL match",
        s.dataset,
        s.nodes,
        s.links,
        s.unmatched,
    )
    return 0


def _link_drugs(graph: KnowledgeGraph, args: argparse.Namespace) -> int:
    settings = get_settings()
    root = settings.paths.chembl
    lookup = ChemblLookup(root / args.release if args.release else _latest(root))
    if args.drugs_file is not None:
        drugs, origin = (
            drugs_from_file(args.drugs_file),
            f"file:{args.drugs_file.resolve()}",
        )
    elif args.drugs_uri is not None:
        source = KGSettings(
            uri=args.drugs_uri, user=args.drugs_user, password=args.drugs_password
        )
        with KnowledgeGraph.connect(source) as other:
            drugs = drugs_from_graph(other)
        origin = args.drugs_uri
    else:
        drugs, origin = drugs_from_graph(graph), settings.kg.uri
    if drugs.height == 0:
        logger.error("no drugs found in %s", origin)
        return 1
    with PubChemClient(
        settings.pubchem, settings.paths.cache / "pubchem", offline=args.offline
    ) as client:
        s = link_drugs(
            graph,
            drugs,
            lookup,
            client,
            origin=origin,
            write_drugs=origin != settings.kg.uri,
            repo=PROJECT_ROOT,
        )
    logger.info("%d drugs: %d linked, %d unmatched", drugs.height, s.links, s.unmatched)
    return 0


def _stats(graph: KnowledgeGraph, _: argparse.Namespace) -> int:
    nodes, rels = graph.stats()
    with pl.Config(tbl_rows=-1):
        print(nodes)
        print(rels)
    return 0


def _audit(graph: KnowledgeGraph, _: argparse.Namespace) -> int:
    result = graph.audit()
    print(result)
    return 0 if not any(result.values()) else 2


def _search(graph: KnowledgeGraph, args: argparse.Namespace) -> int:
    print(graph.search_compounds(args.text, args.limit))
    return 0


_COMMANDS: dict[str, Callable[[KnowledgeGraph, argparse.Namespace], int]] = {
    "load-chembl": _load_chembl,
    "remove-source": _remove_source,
    "link-library": _link_library,
    "link-drugs": _link_drugs,
    "stats": _stats,
    "audit": _audit,
    "search": _search,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(name="kg", level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        with KnowledgeGraph.connect() as graph:
            return _COMMANDS[args.command](graph, args)
    except ModephereException as e:
        logger.error("%s", e)  # noqa: TRY400 -- user-facing message
        return 1


if __name__ == "__main__":
    sys.exit(main())
