"""Neo4j access for Modephere's knowledge graph.

Writes are idempotent ``UNWIND $rows ... MERGE`` batches keyed by each label's
unique property (:data:`~modephere.kg.schema.NODE_KEYS`). Every node and
relationship receives ``run_id`` and unions ``sources`` (dev_Pgen's
convention), so reloading a release, or loading another source into the same
graph, never drops provenance. Labels and relationship types reach Cypher only
after an allow-list check: they cannot be query parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import polars as pl

from modephere.config import get_settings
from modephere.core import ConfigurationError, ResourceError
from modephere.kg.schema import (
    FULLTEXT_COMPOUND_NAMES,
    NODE_KEYS,
    REL_TYPES,
    SCHEMA_STATEMENTS,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from types import TracebackType

    from neo4j import Driver

    from modephere.config import KGSettings
    from modephere.kg.provenance import Dataset, IngestRun, Stamp

logger = logging.getLogger(__name__)

type Row = dict[str, Any]

_UNION_SOURCES = (
    "{v}.sources = CASE WHEN {v}.sources IS NULL THEN $sources "
    "ELSE {v}.sources + [s IN $sources WHERE NOT s IN {v}.sources] END"
)

AUDIT = """
CALL () { MATCH (n) WHERE n.sources IS NULL OR size(n.sources) = 0 OR n.run_id IS NULL
          RETURN count(n) AS nodes_missing }
CALL () { MATCH ()-[r]->() WHERE r.sources IS NULL OR size(r.sources) = 0 OR r.run_id IS NULL
          RETURN count(r) AS rels_missing }
CALL () { CALL () { MATCH (n) UNWIND n.sources AS s RETURN s
                    UNION
                    MATCH ()-[r]->() UNWIND r.sources AS s RETURN s }
          WITH DISTINCT s WHERE NOT EXISTS { MATCH (:Dataset {id: s}) }
          RETURN count(s) AS unknown_sources }
RETURN nodes_missing, rels_missing, unknown_sources
"""

NODE_STATS = (
    "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n ORDER BY label"
)
REL_STATS = "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type"

SEARCH_COMPOUNDS = f"""
CALL db.index.fulltext.queryNodes('{FULLTEXT_COMPOUND_NAMES}', $query) YIELD node, score
RETURN node.chembl_id AS chembl_id, node.pref_name AS pref_name,
       node.max_phase AS max_phase, node.inchikey AS inchikey, score
ORDER BY score DESC LIMIT $limit
"""


def _checked_label(label: str) -> str:
    if label not in NODE_KEYS:
        msg = f"label {label!r} is not in the graph schema"
        raise ConfigurationError(msg)
    return label


def _checked_type(rel_type: str) -> str:
    if rel_type not in REL_TYPES:
        msg = f"relationship type {rel_type!r} is not in the graph schema"
        raise ConfigurationError(msg)
    return rel_type


def node_query(label: str) -> str:
    """``MERGE`` statement for rows ``{key, props}`` of ``label``."""
    key = NODE_KEYS[_checked_label(label)]
    return (
        "UNWIND $rows AS row "
        f"MERGE (n:{label} {{{key}: row.key}}) "
        "SET n += row.props, n += $stamp, "
        f"{_UNION_SOURCES.format(v='n')} "
        "RETURN count(n) AS written"
    )


@dataclass(frozen=True)
class RelSpec:
    """A relationship type between two labels, optionally keyed.

    Unkeyed: at most one relationship of the type per node pair. Keyed: one
    per distinct ``key`` value (e.g. one ``HAS_MECHANISM`` per ChEMBL
    mechanism record).
    """

    type: str
    start: str
    end: str
    key: str | None = None

    def query(self) -> str:
        """``MERGE`` statement for rows ``{start, end, [key], props}``."""
        rel = _checked_type(self.type)
        start, end = _checked_label(self.start), _checked_label(self.end)
        ident = f" {{{self.key}: row.key}}" if self.key else ""
        return (
            "UNWIND $rows AS row "
            f"MATCH (a:{start} {{{NODE_KEYS[start]}: row.start}}) "
            f"MATCH (b:{end} {{{NODE_KEYS[end]}: row.end}}) "
            f"MERGE (a)-[r:{rel}{ident}]->(b) "
            "SET r += row.props, r += $stamp, "
            f"{_UNION_SOURCES.format(v='r')} "
            "RETURN count(r) AS written"
        )


def batched(rows: Sequence[Row], size: int) -> Iterator[Sequence[Row]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


class KnowledgeGraph:
    """A Neo4j database holding the knowledge graph."""

    def __init__(
        self, driver: Driver, *, database: str = "neo4j", batch_size: int = 1000
    ) -> None:
        self._driver = driver
        self._database = database
        self._batch_size = batch_size

    @classmethod
    def connect(cls, settings: KGSettings | None = None) -> Self:
        """Open a driver from :class:`~modephere.config.KGSettings` and verify it."""
        try:
            from neo4j import GraphDatabase
            from neo4j.exceptions import Neo4jError, ServiceUnavailable
        except ImportError as e:  # pragma: no cover
            msg = "the knowledge graph needs the neo4j driver: uv sync --extra kg"
            raise ConfigurationError(msg) from e
        cfg = settings or get_settings().kg
        driver = GraphDatabase.driver(
            cfg.uri, auth=(cfg.user, cfg.password.get_secret_value())
        )
        try:
            driver.verify_connectivity()
        except (ServiceUnavailable, Neo4jError, OSError) as e:
            driver.close()
            msg = f"cannot reach Neo4j at {cfg.uri}: {e}"
            raise ResourceError(msg) from e
        return cls(driver, database=cfg.database, batch_size=cfg.batch_size)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #

    def run(self, cypher: str, /, **params: Any) -> list[Row]:
        """Run one auto-commit query and return its records as dicts."""
        records, _, _ = self._driver.execute_query(
            cypher, params, database_=self._database
        )
        return [r.data() for r in records]

    def _write_batches(self, query: str, rows: Sequence[Row], stamp: Stamp) -> int:
        written = 0
        for batch in batched(rows, self._batch_size):
            out = self.run(
                query,
                rows=list(batch),
                stamp=stamp.props(),
                sources=list(stamp.sources),
            )
            written += int(out[0]["written"]) if out else 0
        return written

    def ensure_schema(self, statements: Iterable[str] = SCHEMA_STATEMENTS) -> None:
        """Create constraints and indexes (idempotent)."""
        for statement in statements:
            self.run(statement)

    def merge_nodes(self, label: str, rows: Sequence[Row], stamp: Stamp) -> int:
        """Merge rows ``{"key": ..., "props": {...}}`` as ``label`` nodes."""
        n = self._write_batches(node_query(label), rows, stamp)
        logger.info("%s: %d nodes", label, n)
        return n

    def merge_relationships(
        self, spec: RelSpec, rows: Sequence[Row], stamp: Stamp
    ) -> int:
        """Merge rows ``{"start", "end", ["key"], "props"}`` as ``spec`` edges.

        Rows whose endpoints are not in the graph are skipped; the returned
        count says how many were written.
        """
        n = self._write_batches(spec.query(), rows, stamp)
        if n != len(rows):
            logger.warning(
                "%s: %d of %d rows written (missing endpoints)", spec.type, n, len(rows)
            )
        else:
            logger.info("%s: %d relationships", spec.type, n)
        return n

    def record_run(self, dataset: Dataset, run: IngestRun) -> None:
        """Write the ``(:IngestRun)-[:USED]->(:Dataset)`` provenance nodes."""
        self.run(
            "MERGE (d:Dataset {id: $dataset}) SET d += $dprops, "
            "d.sources = [$dataset], d.run_id = $run "
            "MERGE (r:IngestRun {id: $run}) SET r += $rprops, "
            "r.sources = [$dataset], r.run_id = $run "
            "MERGE (r)-[u:USED]->(d) SET u.sources = [$dataset], u.run_id = $run",
            dataset=dataset.id,
            dprops=dataset.props(),
            run=run.id,
            rprops=run.props(),
        )

    def record_datasets(self, datasets: Sequence[Dataset], run: IngestRun) -> None:
        """Provenance of a run over several datasets: every ``:Dataset``, the
        ``:IngestRun`` (its ``sources`` are all of them) and one ``USED`` each.

        Unlike repeated :meth:`record_run` calls, the run keeps every dataset in
        ``sources``, so :meth:`remove_source` deletes it only with the last one.
        """
        ids = [d.id for d in datasets]
        self.run(
            "UNWIND $datasets AS ds "
            "MERGE (d:Dataset {id: ds.id}) SET d += ds.props, "
            "d.sources = [ds.id], d.run_id = $run",
            datasets=[{"id": d.id, "props": d.props()} for d in datasets],
            run=run.id,
        )
        self.run(
            "MERGE (r:IngestRun {id: $run}) SET r += $rprops, "
            "r.sources = $ids, r.run_id = $run "
            "WITH r UNWIND $ids AS id MATCH (d:Dataset {id: id}) "
            "MERGE (r)-[u:USED]->(d) SET u.sources = [id], u.run_id = $run",
            run=run.id,
            rprops=run.props(),
            ids=ids,
        )

    def layer_runs(self, layer: str) -> list[Row]:
        """``{id, datasets}`` of every ``:IngestRun`` of ``layer``."""
        return self.run(
            "MATCH (r:IngestRun {layer: $layer}) "
            "OPTIONAL MATCH (r)-[:USED]->(d:Dataset) "
            "RETURN r.id AS id, collect(d.id) AS datasets ORDER BY id",
            layer=layer,
        )

    def remove_source(self, dataset_id: str) -> Row:
        """Withdraw one dataset from the graph.

        Elements supported only by ``dataset_id`` are deleted (with its
        ``:Dataset`` and ``:IngestRun`` nodes); elements other sources also
        support just lose it from ``sources``. Data written by other projects
        into the same database is left alone.
        """
        steps = {
            "relationships_deleted": (
                "MATCH ()-[r]->() WHERE r.sources = [$id] "
                "CALL (r) { DELETE r } IN TRANSACTIONS OF 10000 ROWS"
            ),
            "relationships_updated": (
                "MATCH ()-[r]->() WHERE $id IN r.sources "
                "SET r.sources = [s IN r.sources WHERE s <> $id]"
            ),
            "nodes_deleted": (
                "MATCH (n) WHERE n.sources = [$id] "
                "CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
            ),
            "nodes_updated": (
                "MATCH (n) WHERE $id IN n.sources "
                "SET n.sources = [s IN n.sources WHERE s <> $id]"
            ),
        }
        out: Row = {}
        # CALL ... IN TRANSACTIONS needs an implicit (auto-commit) transaction.
        with self._driver.session(database=self._database) as session:
            for name, cypher in steps.items():
                counters = session.run(cypher, id=dataset_id).consume().counters
                out[name] = (
                    counters.relationships_deleted
                    if name == "relationships_deleted"
                    else counters.nodes_deleted
                    if name == "nodes_deleted"
                    else counters.properties_set
                )
        logger.info("Removed %s: %s", dataset_id, out)
        return out

    # ------------------------------------------------------------------ #

    def stats(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Node counts per label and relationship counts per type."""
        nodes = pl.DataFrame(
            self.run(NODE_STATS), schema={"label": pl.String, "n": pl.Int64}
        )
        rels = pl.DataFrame(
            self.run(REL_STATS), schema={"type": pl.String, "n": pl.Int64}
        )
        return nodes, rels

    def audit(self) -> Row:
        """Elements without provenance, and sources with no ``:Dataset`` node."""
        return self.run(AUDIT)[0]

    def search_compounds(self, text: str, limit: int = 10) -> pl.DataFrame:
        """Full-text search over compound names and synonyms."""
        rows = self.run(SEARCH_COMPOUNDS, query=_lucene_escape(text), limit=limit)
        return pl.DataFrame(
            rows,
            schema={
                "chembl_id": pl.String,
                "pref_name": pl.String,
                "max_phase": pl.Float64,
                "inchikey": pl.String,
                "score": pl.Float64,
            },
        )


_LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/')


def _lucene_escape(text: str) -> str:
    return "".join(f"\\{c}" if c in _LUCENE_SPECIAL else c for c in text)


__all__ = ["KnowledgeGraph", "RelSpec", "batched", "node_query"]
