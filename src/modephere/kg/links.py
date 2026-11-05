"""Links from the ChEMBL layer to the rest: the built library and dev_Pgen's drugs.

* ``(:LibraryCompound {inchikey})-[:REPRESENTS]->(:Compound)`` — every
  compound of a Modephere library (its registry), tied to its ChEMBL parent
  by the ``chembl_id`` the resolver recorded. Library compounds without a
  ChEMBL match are written too (no edge), so the graph knows what is built.
* ``(:Drug {name})-[:SAME_AS {method, evidence}]->(:Compound)`` — dev_Pgen's
  PharmGKB drugs (normalised name, PubChem CID), matched by the same
  :class:`~modephere.library.ingest.resolve.Resolver` the library uses: the
  CID's InChIKey (PubChem) against ChEMBL, and the name against ChEMBL's
  curated names and synonyms. When both give a parent and they differ, the
  name wins (the CID came from an automatic name lookup, ChEMBL's synonyms are
  curated) and the evidence records the conflict. Drug classes (ATC class
  names, e.g. "corticosteroids") and combinations ("a / b") are never linked
  to a single compound.

Drugs are read from a graph that holds them (dev_Pgen's, or this one when the
layers share a database) or from a table with ``name`` and ``cid`` columns.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from modephere.core import DataError
from modephere.kg.graph import RelSpec
from modephere.kg.provenance import Dataset, IngestRun, Stamp
from modephere.library.ingest.resolve import Resolver
from modephere.types import ChemblMatch, CidQuery, NameQuery, ResolvedCompound

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from modephere.chembl.lookup import ChemblLookup
    from modephere.kg.graph import KnowledgeGraph, Row
    from modephere.library.ingest.pubchem import PubChemClient
    from modephere.types import CompoundQuery

logger = logging.getLogger(__name__)

REPRESENTS = RelSpec("REPRESENTS", "LibraryCompound", "Compound")
SAME_AS = RelSpec("SAME_AS", "Drug", "Compound")

_REGISTRY_PROPS = (
    "title",
    "query_key",
    "source",
    "cid",
    "chembl_id",
    "working_smiles",
    "microspecies_smiles",
    "num_heavy_atoms",
    "formal_charge",
    "toolkit_version",
)

# --------------------------------------------------------------------------- #
# Library -> graph
# --------------------------------------------------------------------------- #


def library_dataset(root: Path, dataset_id: str | None = None) -> Dataset:
    """The ``:Dataset`` of a library: its registry, as last written."""
    registry = root / "compounds.parquet"
    if not registry.exists():
        msg = f"no library registry at {registry}"
        raise DataError(msg)
    modified = datetime.fromtimestamp(registry.stat().st_mtime, tz=UTC)
    return Dataset(
        id=dataset_id or f"modephere_library:{root.resolve().name}",
        name="Modephere library",
        version=modified.isoformat(timespec="seconds"),
        uri=str(root.resolve()),
        retrieved_at=modified,
    )


def library_rows(root: Path) -> tuple[list[Row], list[tuple[str, list[Row]]]]:
    """``LibraryCompound`` rows, and ``REPRESENTS`` rows per ChEMBL release."""
    registry = pl.read_parquet(root / "compounds.parquet")
    conformers_path = root / "conformers.parquet"
    counts = (
        pl.read_parquet(conformers_path, columns=["inchikey"])
        .group_by("inchikey")
        .agg(num_conformers=pl.len().cast(pl.Int64))
        if conformers_path.exists()
        else pl.DataFrame(schema={"inchikey": pl.String, "num_conformers": pl.Int64})
    )
    frame = registry.join(counts, on="inchikey", how="left").with_columns(
        pl.col("num_conformers").fill_null(0),
        pl.col("source").cast(pl.String),
        pl.col("chembl_match").cast(pl.String),
    )
    nodes = frame.select(
        key="inchikey",
        props=pl.struct(*_REGISTRY_PROPS, "num_conformers"),
    ).to_dicts()
    edges: list[tuple[str, list[Row]]] = []
    linked = frame.drop_nulls("chembl_id")
    for (release,), group in linked.group_by("chembl_release", maintain_order=True):
        rows = group.select(
            start="inchikey",
            end="chembl_id",
            props=pl.struct(match="chembl_match", chembl_record_id="chembl_record_id"),
        ).to_dicts()
        edges.append((str(release), rows))
    return nodes, edges


@dataclass(frozen=True)
class LinkSummary:
    run_id: str
    dataset: str
    nodes: int
    links: int
    unmatched: int


def link_library(
    graph: KnowledgeGraph,
    root: Path,
    *,
    dataset_id: str | None = None,
    repo: Path | None = None,
) -> LinkSummary:
    """Merge a library's compounds and their ``REPRESENTS`` edges."""
    dataset = library_dataset(root, dataset_id)
    run = IngestRun.start({"library": str(root)}, repo=repo)
    nodes, edges = library_rows(root)
    graph.ensure_schema()
    graph.record_run(dataset, run)
    written = graph.merge_nodes(
        "LibraryCompound", nodes, Stamp(sources=(dataset.id,), run_id=run.id)
    )
    links = 0
    for release, rows in edges:
        stamp = Stamp(sources=(dataset.id, release), run_id=run.id)
        links += graph.merge_relationships(REPRESENTS, rows, stamp)
    graph.record_run(dataset, run.model_copy(update={"finished_at": datetime.now(UTC)}))
    return LinkSummary(run.id, dataset.id, written, links, written - links)


# --------------------------------------------------------------------------- #
# dev_Pgen drugs -> graph
# --------------------------------------------------------------------------- #

DRUGS_FROM_GRAPH = "MATCH (d:Drug) RETURN d.name AS name, d.cid AS cid ORDER BY name"


@dataclass(frozen=True)
class DrugLink:
    name: str
    chembl_id: str
    method: str
    evidence: str
    chembl_record_id: str


def drugs_from_graph(graph: KnowledgeGraph) -> pl.DataFrame:
    """``(name, cid)`` of every ``:Drug`` node in ``graph``."""
    return pl.DataFrame(
        graph.run(DRUGS_FROM_GRAPH), schema={"name": pl.String, "cid": pl.Int64}
    )


def drugs_from_file(path: Path) -> pl.DataFrame:
    """``(name, cid)`` from a tsv / csv / parquet with ``name`` (or dev_Pgen's
    ``cmpd_name_cleaned``) and optional ``cid`` columns."""
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
    else:
        sep = "," if path.suffix == ".csv" else "\t"
        frame = pl.read_csv(path, separator=sep, infer_schema=False, quote_char=None)
    name = next((c for c in ("name", "cmpd_name_cleaned") if c in frame.columns), None)
    if name is None:
        msg = f"{path}: needs a 'name' (or 'cmpd_name_cleaned') column"
        raise DataError(msg)
    cid = pl.col("cid").cast(pl.Int64, strict=False) if "cid" in frame.columns else None
    return (
        frame.select(
            name=pl.col(name).str.strip_chars(),
            cid=cid if cid is not None else pl.lit(None, pl.Int64),
        )
        .filter(pl.col("name").is_not_null() & (pl.col("name") != ""))
        .unique("name", keep="first", maintain_order=True)
    )


def match_drugs(
    drugs: pl.DataFrame,
    chembl: ChemblLookup,
    client: PubChemClient | None,
) -> tuple[list[DrugLink], list[str]]:
    """ChEMBL parent of each drug: by its CID's structure, else by its name."""
    resolver = Resolver(client, chembl=chembl)
    rows = list(drugs.iter_rows(named=True))
    by_cid: dict[int, ResolvedCompound | Exception] = {}
    with_cid = [r["cid"] for r in rows if r["cid"] is not None]
    if with_cid and client is not None:
        queries: Sequence[CompoundQuery] = [CidQuery(cid=c) for c in with_cid]
        for cid, result in zip(with_cid, resolver.resolve_many(queries), strict=True):
            by_cid[cid] = result
    links: list[DrugLink] = []
    unmatched: list[str] = []
    for row in rows:
        link = _link(row, by_cid.get(row["cid"]), chembl)
        if link is None:
            unmatched.append(row["name"])
        else:
            links.append(link)
    return links, unmatched


def is_class_or_mixture(name: str, chembl: ChemblLookup) -> bool:
    """A drug class or a combination: not one compound."""
    return (
        " / " in name
        or "+" in name.split("(", maxsplit=1)[0]
        or chembl.is_atc_class(name)
    )


def _link(
    row: dict[str, Any], cid_result: object, chembl: ChemblLookup
) -> DrugLink | None:
    if is_class_or_mixture(row["name"], chembl):
        return None
    by_cid = _cid_link(row, cid_result)
    by_name = _name_link(row, chembl)
    if by_cid is None or by_name is None or by_cid.chembl_id == by_name.chembl_id:
        return by_cid or by_name
    return dataclasses.replace(
        by_name,
        evidence=f"{by_name.evidence}; overrides {by_cid.evidence} ({by_cid.chembl_id})",
    )


def _cid_link(row: dict[str, Any], result: object) -> DrugLink | None:
    if not isinstance(result, ResolvedCompound) or result.chembl_id is None:
        return None
    assert result.chembl_record_id is not None and result.chembl_match is not None
    via = (
        f"PubChem InChIKey {result.inchikey}"
        if result.chembl_match is ChemblMatch.UPSTREAM_INCHIKEY
        else "structure InChIKey"
    )
    return DrugLink(
        name=row["name"],
        chembl_id=result.chembl_id,
        method="cid_inchikey",
        evidence=f"CID {row['cid']} -> {via} -> {result.chembl_record_id}",
        chembl_record_id=result.chembl_record_id,
    )


def _name_link(row: dict[str, Any], chembl: ChemblLookup) -> DrugLink | None:
    hits = chembl.by_name(row["name"])
    parents = list(dict.fromkeys(h.parent_chembl_id for h in hits))
    if len(parents) != 1:  # unknown, or ambiguous: no guessed link
        return None
    return DrugLink(
        name=row["name"],
        chembl_id=parents[0],
        method="chembl_name",
        evidence=f"name matches ChEMBL {hits[0].matched} of {hits[0].chembl_id}",
        chembl_record_id=hits[0].chembl_id,
    )


def drugs_dataset(origin: str, count: int) -> Dataset:
    now = datetime.now(UTC)
    return Dataset(
        id=f"drugs:{origin}",
        name="Drug list (name, PubChem CID)",
        version=f"{count} drugs at {now.isoformat(timespec='seconds')}",
        uri=origin,
        retrieved_at=now,
    )


def link_drugs(
    graph: KnowledgeGraph,
    drugs: pl.DataFrame,
    chembl: ChemblLookup,
    client: PubChemClient | None,
    *,
    origin: str,
    write_drugs: bool,
    repo: Path | None = None,
) -> LinkSummary:
    """Merge ``SAME_AS`` edges (and, from outside, the ``:Drug`` nodes)."""
    dataset = drugs_dataset(origin, drugs.height)
    run = IngestRun.start({"drugs": origin, "chembl": chembl.release}, repo=repo)
    links, unmatched = match_drugs(drugs, chembl, client)
    graph.ensure_schema()
    graph.record_run(dataset, run)
    nodes = 0
    if write_drugs:
        rows = drugs.select(key="name", props=pl.struct("cid")).to_dicts()
        nodes = graph.merge_nodes(
            "Drug", rows, Stamp(sources=(dataset.id,), run_id=run.id)
        )
    edges = _link_rows(links)
    stamp = Stamp(sources=(dataset.id, chembl.release), run_id=run.id)
    written = graph.merge_relationships(SAME_AS, edges, stamp)
    graph.record_run(dataset, run.model_copy(update={"finished_at": datetime.now(UTC)}))
    logger.info(
        "Drugs: %d linked to ChEMBL, %d unmatched (e.g. %s)",
        written,
        len(unmatched),
        ", ".join(unmatched[:5]),
    )
    return LinkSummary(run.id, dataset.id, nodes, written, len(unmatched))


def _link_rows(links: Iterable[DrugLink]) -> list[Row]:
    return [
        {
            "start": link.name,
            "end": link.chembl_id,
            "props": {
                "method": link.method,
                "evidence": link.evidence,
                "chembl_record_id": link.chembl_record_id,
            },
        }
        for link in links
    ]


__all__ = [
    "REPRESENTS",
    "SAME_AS",
    "DrugLink",
    "LinkSummary",
    "drugs_from_file",
    "drugs_from_graph",
    "is_class_or_mixture",
    "library_dataset",
    "library_rows",
    "link_drugs",
    "link_library",
    "match_drugs",
]
