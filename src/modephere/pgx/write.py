"""Write a :class:`~modephere.pgx.build.KGBuild` into Modephere's graph.

The pharmacogenomic layer shares the database with the ChEMBL layer, so a
rebuild can never wipe the graph. Instead every pgx run is an
``:IngestRun {layer: "pgx"}``; ``replace=True`` withdraws every dataset those
runs used (:meth:`~modephere.kg.graph.KnowledgeGraph.remove_source`) before
writing. Elements only the pgx layer supports disappear, shared ones
(``Gene``, ``Drug``) just lose the pgx sources, ChEMBL data is untouched.

Rows are written through the graph's own ``MERGE`` batches, grouped by
``(sources, method)`` so each batch carries one stamp; per-row ``evidence``
travels as a property. Unlike the ChEMBL loader, a relationship whose endpoint
is missing is an error here: the build is self-contained, so it means a bug.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modephere.core import DataError
from modephere.kg.graph import RelSpec
from modephere.kg.provenance import Stamp
from modephere.kg.schema import NODE_KEYS

if TYPE_CHECKING:
    from modephere.kg.graph import KnowledgeGraph, Row
    from modephere.pgx.build import KGBuild
    from modephere.pgx.provenance import ElementStamp

logger = logging.getLogger(__name__)

LAYER = "pgx"


class LayerExistsError(DataError):
    """The graph already holds pgx runs and ``replace`` was not requested."""


@dataclass(frozen=True)
class WriteSummary:
    run_id: str
    datasets: int
    nodes: int
    relationships: int
    replaced_runs: int


def _cypher_props(props: dict[str, object]) -> dict[str, object]:
    """Neo4j properties: datetimes as ISO strings, dicts as ``k=v`` lists."""
    out: dict[str, object] = {}
    for k, v in props.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, dict):
            out[k] = [f"{a}={b}" for a, b in sorted(v.items())]
        else:
            out[k] = v
    return out


def _with_evidence(props: dict[str, object], stamp: ElementStamp) -> dict[str, object]:
    out = _cypher_props(props)
    if stamp.evidence is not None:
        out["evidence"] = stamp.evidence
    return out


def _batch_stamp(stamp: ElementStamp) -> Stamp:
    return Stamp(sources=stamp.sources, run_id=stamp.run_id, method=stamp.method)


def remove_layer(graph: KnowledgeGraph, layer: str = LAYER) -> int:
    """Withdraw every dataset of every ``layer`` run; returns the run count."""
    runs = graph.layer_runs(layer)
    datasets = sorted({d for r in runs for d in r["datasets"]})
    for dataset in datasets:
        graph.remove_source(dataset)
    # A run whose datasets vanished elsewhere would otherwise linger.
    graph.run(
        "MATCH (r:IngestRun {layer: $layer}) WHERE r.id IN $ids DETACH DELETE r",
        layer=layer,
        ids=[r["id"] for r in runs],
    )
    if runs:
        logger.info(
            "Removed %d %s run(s) and %d datasets", len(runs), layer, len(datasets)
        )
    return len(runs)


def write_build(
    graph: KnowledgeGraph, build: KGBuild, *, replace: bool = False
) -> WriteSummary:
    """Merge ``build`` into ``graph`` (see the module docstring)."""
    existing = graph.layer_runs(LAYER)
    if existing and not replace:
        ids = [r["id"] for r in existing]
        msg = (
            f"the graph already holds pgx run(s) {ids}; rerun with --replace to "
            "rebuild the layer (stale elements of an older run would survive a merge)"
        )
        raise LayerExistsError(msg)
    graph.ensure_schema()
    replaced = remove_layer(graph) if existing else 0
    run = build.run.model_copy(update={"layer": LAYER})
    graph.record_datasets(build.datasets, run)

    node_groups: dict[tuple[str, ElementStamp], list[Row]] = defaultdict(list)
    for n in build.nodes:
        key = n.key[NODE_KEYS[n.label]]
        stamp = n.stamp.model_copy(update={"evidence": None})
        node_groups[n.label, stamp].append(
            {"key": key, "props": _with_evidence(n.props, n.stamp)}
        )
    nodes = 0
    for (label, stamp), rows in node_groups.items():
        nodes += graph.merge_nodes(label, rows, _batch_stamp(stamp))

    rel_groups: dict[tuple[RelSpec, ElementStamp], list[Row]] = defaultdict(list)
    for r in build.rels:
        spec = RelSpec(r.type, r.start_label, r.end_label)
        stamp = r.stamp.model_copy(update={"evidence": None})
        rel_groups[spec, stamp].append(
            {
                "start": r.start_key[NODE_KEYS[r.start_label]],
                "end": r.end_key[NODE_KEYS[r.end_label]],
                "props": _with_evidence(r.props, r.stamp),
            }
        )
    rels = 0
    for (spec, stamp), rows in rel_groups.items():
        written = graph.merge_relationships(spec, rows, _batch_stamp(stamp))
        if written != len(rows):
            msg = (
                f"{spec.type} ({spec.start}->{spec.end}): wrote {written} of "
                f"{len(rows)} relationships; an endpoint node is missing"
            )
            raise DataError(msg)
        rels += written
    logger.info(
        "pgx run %s: %d datasets, %d nodes, %d relationships",
        run.id,
        len(build.datasets),
        nodes,
        rels,
    )
    return WriteSummary(run.id, len(build.datasets), nodes, rels, replaced)


__all__ = ["LAYER", "LayerExistsError", "WriteSummary", "remove_layer", "write_build"]
