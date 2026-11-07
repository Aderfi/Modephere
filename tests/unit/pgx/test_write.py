"""The pgx writer against a recording fake driver: grouping, provenance, replace."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from modephere.core import DataError
from modephere.kg import KnowledgeGraph
from modephere.kg.provenance import Dataset, IngestRun
from modephere.kg.schema import NODE_KEYS, REL_TYPES
from modephere.pgx.build import KGBuild, NodeRow, RelRow
from modephere.pgx.provenance import ElementStamp
from modephere.pgx.write import LayerExistsError, write_build

NOW = datetime(2026, 10, 18, tzinfo=UTC)


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class _Session:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params: Any) -> SimpleNamespace:
        self.driver.session_calls.append((query, params))
        counters = SimpleNamespace(
            relationships_deleted=0, nodes_deleted=0, properties_set=0
        )
        return SimpleNamespace(consume=lambda: SimpleNamespace(counters=counters))

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeDriver:
    """Records queries; ``runs`` are the pgx runs already in the graph."""

    def __init__(
        self, runs: list[dict[str, Any]] | None = None, short_rel: str | None = None
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.session_calls: list[tuple[str, dict[str, Any]]] = []
        self.runs = runs or []
        self.short_rel = short_rel

    def execute_query(
        self, query: str, params: dict[str, Any], database_: str
    ) -> tuple[list[_Record], None, None]:
        self.calls.append((query, params))
        if "MATCH (r:IngestRun {layer: $layer})" in query and "RETURN r.id" in query:
            return [_Record(r) for r in self.runs], None, None
        if "RETURN count" in query and "rows" in params:
            n = len(params["rows"])
            if self.short_rel and f":{self.short_rel}]" in query:
                n -= 1
            return [_Record({"written": n})], None, None
        return [], None, None

    def session(self, database: str) -> _Session:
        return _Session(self)

    def close(self) -> None:
        pass


def _stamp(*sources: str, method: str | None = None, evidence: str | None = None):
    return ElementStamp(
        sources=sources or ("pharmgkb:var_drug_ann",),
        run_id="run1",
        method=method,
        evidence=evidence,
    )


def small_build() -> KGBuild:
    ds = Dataset(
        id="pharmgkb:var_drug_ann",
        name="n",
        version="v",
        uri="u",
        retrieved_at=NOW,
        sha256="0" * 64,
    )
    dbsnp = ds.model_copy(update={"id": "dbsnp_esummary@b157#abc"})
    run = IngestRun(id="run1", started_at=NOW, code_version="abc", dirty=False)
    return KGBuild(
        datasets=[ds, dbsnp],
        run=run,
        nodes=[
            NodeRow(label="Annotation", key={"id": "a1"}, props={}, stamp=_stamp()),
            NodeRow(label="Annotation", key={"id": "a2"}, props={}, stamp=_stamp()),
            NodeRow(
                label="Variant",
                key={"hgvs_g38": "NC_000010.11:g.94781859G>A"},
                props={"spdi_g38": "NC_000010.11:94781858:G:A"},
                stamp=_stamp("dbsnp_esummary@b157#abc"),
            ),
        ],
        rels=[
            RelRow(
                type="IMPLICATES",
                start_label="Annotation",
                start_key={"id": key},
                end_label="Variant",
                end_key={"hgvs_g38": "NC_000010.11:g.94781859G>A"},
                stamp=_stamp(
                    "pharmgkb:var_drug_ann",
                    "dbsnp_esummary@b157#abc",
                    method="annotation_alleles_forward_strand",
                    evidence=f"Alleles={key!r}",
                ),
            )
            for key in ("a1", "a2")
        ],
        resolution=[],
        disagreements=[],
        inference_counts={},
    )


def _merges(driver: FakeDriver, needle: str) -> list[dict[str, Any]]:
    return [p for q, p in driver.calls if needle in q]


def test_schema_covers_the_pgx_model() -> None:
    assert {"Annotation", "Allele", "RefSNP", "Haplotype", "Variant"} <= set(NODE_KEYS)
    assert NODE_KEYS["Variant"] == "hgvs_g38"
    assert NODE_KEYS["Genotype"] == "name"
    assert {"IMPLICATES", "HAS_ALLELE", "HAS_VARIANT", "REFERS_TO"} <= REL_TYPES
    assert {"TESTED", "COMPARED_TO", "HAS_COPY"} <= REL_TYPES


def test_writes_provenance_then_nodes_then_relationships() -> None:
    driver = FakeDriver()
    summary = write_build(KnowledgeGraph(driver), small_build())  # type: ignore[arg-type]
    queries = [q for q, _ in driver.calls]
    first_dataset = next(i for i, q in enumerate(queries) if "MERGE (d:Dataset" in q)
    first_node = next(i for i, q in enumerate(queries) if "MERGE (n:" in q)
    first_rel = next(i for i, q in enumerate(queries) if "MERGE (a)-" in q)
    assert first_dataset < first_node < first_rel
    assert (summary.nodes, summary.relationships, summary.replaced_runs) == (3, 2, 0)


def test_run_is_tagged_with_its_layer_and_cites_every_dataset() -> None:
    driver = FakeDriver()
    write_build(KnowledgeGraph(driver), small_build())  # type: ignore[arg-type]
    (params,) = _merges(driver, "MERGE (r:IngestRun")
    assert params["rprops"]["layer"] == "pgx"
    assert params["ids"] == ["pharmgkb:var_drug_ann", "dbsnp_esummary@b157#abc"]


def test_rows_are_grouped_by_stamp_and_keep_their_evidence() -> None:
    driver = FakeDriver()
    write_build(KnowledgeGraph(driver), small_build())  # type: ignore[arg-type]
    annotations = _merges(driver, "MERGE (n:Annotation")
    assert len(annotations) == 1 and len(annotations[0]["rows"]) == 2
    (implicates,) = _merges(driver, ":IMPLICATES]")
    assert implicates["sources"] == ["pharmgkb:var_drug_ann", "dbsnp_esummary@b157#abc"]
    assert implicates["stamp"] == {
        "run_id": "run1",
        "method": "annotation_alleles_forward_strand",
    }
    assert [r["props"]["evidence"] for r in implicates["rows"]] == [
        "Alleles='a1'",
        "Alleles='a2'",
    ]


def test_missing_endpoint_is_an_error() -> None:
    driver = FakeDriver(short_rel="IMPLICATES")
    with pytest.raises(DataError, match="endpoint node is missing"):
        write_build(KnowledgeGraph(driver), small_build())  # type: ignore[arg-type]


def test_existing_layer_needs_replace() -> None:
    driver = FakeDriver(runs=[{"id": "old", "datasets": ["pharmgkb:old"]}])
    with pytest.raises(LayerExistsError, match="--replace"):
        write_build(KnowledgeGraph(driver), small_build())  # type: ignore[arg-type]
    assert not _merges(driver, "MERGE (n:")


def test_replace_withdraws_only_the_old_pgx_datasets() -> None:
    old = [
        {"id": "old1", "datasets": ["pharmgkb:old", "dbsnp:old"]},
        {"id": "old2", "datasets": ["dbsnp:old"]},
    ]
    driver = FakeDriver(runs=old)
    summary = write_build(KnowledgeGraph(driver), small_build(), replace=True)  # type: ignore[arg-type]
    removed = {p["id"] for _, p in driver.session_calls}
    assert removed == {"pharmgkb:old", "dbsnp:old"}  # never chembl_*
    assert summary.replaced_runs == 2
    assert _merges(driver, "MERGE (n:Annotation")
