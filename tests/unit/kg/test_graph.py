"""The Neo4j writer against a recording fake driver, and the load sequence."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import version
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from modephere.config import PROJECT_ROOT
from modephere.core import ConfigurationError, ResourceError
from modephere.kg import KnowledgeGraph, RelSpec, load_chembl, provenance
from modephere.kg.graph import _lucene_escape, batched, node_query
from modephere.kg.provenance import Dataset, IngestRun, Stamp, code_version

if TYPE_CHECKING:
    from pathlib import Path


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class FakeDriver:
    """Records queries; answers ``written`` with the batch size."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def execute_query(
        self, query: str, params: dict[str, Any], database_: str
    ) -> tuple[list[_Record], None, None]:
        self.calls.append((query, params))
        if "RETURN count" in query and "rows" in params:
            n = len(params["rows"])
            if "MATCH (a:" in query:  # relationships: pretend one endpoint is missing
                n -= params["rows"][0].get("missing", 0)
            return [_Record({"written": n})], None, None
        return [], None, None

    def close(self) -> None:
        self.closed = True


STAMP = Stamp(sources=("chembl_99",), run_id="run1")


def test_node_query_shape() -> None:
    q = node_query("Compound")
    assert "MERGE (n:Compound {chembl_id: row.key})" in q
    assert "n.sources + [s IN $sources WHERE NOT s IN n.sources]" in q


def test_allow_lists() -> None:
    with pytest.raises(ConfigurationError, match="label"):
        node_query("Compound) DETACH DELETE (n")
    with pytest.raises(ConfigurationError, match="relationship type"):
        RelSpec("DROP", "Compound", "Target").query()
    with pytest.raises(ConfigurationError, match="label"):
        RelSpec("HAS_PARENT", "Compound", "Nope").query()


def test_rel_query_keyed() -> None:
    q = RelSpec("HAS_MECHANISM", "Compound", "Target", key="mec_id").query()
    assert "MERGE (a)-[r:HAS_MECHANISM {mec_id: row.key}]->(b)" in q
    assert "MATCH (b:Target {chembl_id: row.end})" in q
    plain = RelSpec("HAS_PARENT", "Compound", "Compound").query()
    assert "MERGE (a)-[r:HAS_PARENT]->(b)" in plain


def test_batching() -> None:
    driver = FakeDriver()
    graph = KnowledgeGraph(driver, batch_size=2)  # type: ignore[arg-type]
    rows = [{"key": f"C{i}", "props": {}} for i in range(5)]
    assert graph.merge_nodes("Compound", rows, STAMP) == 5
    assert [len(p["rows"]) for _, p in driver.calls] == [2, 2, 1]
    assert driver.calls[0][1]["stamp"] == {"run_id": "run1"}
    assert driver.calls[0][1]["sources"] == ["chembl_99"]
    assert list(batched([], 3)) == []


def test_missing_endpoints_are_reported(caplog: pytest.LogCaptureFixture) -> None:
    graph = KnowledgeGraph(FakeDriver())  # type: ignore[arg-type]
    rows: list[dict[str, Any]] = [
        {"start": "a", "end": "b", "props": {}, "missing": 1},
        {"start": "c", "end": "d", "props": {}},
    ]
    assert (
        graph.merge_relationships(
            RelSpec("HAS_PARENT", "Compound", "Compound"), rows, STAMP
        )
        == 1
    )
    assert "1 of 2 rows written" in caplog.text


def test_record_run_and_close() -> None:
    driver = FakeDriver()
    with KnowledgeGraph(driver) as graph:  # type: ignore[arg-type]
        ds = Dataset(
            id="chembl_99",
            name="ChEMBL",
            version="99",
            uri="u",
            retrieved_at=datetime.now(UTC),
        )
        run = IngestRun.start({"a": "1"})
        graph.record_run(ds, run)
    query, params = driver.calls[0]
    assert "MERGE (r)-[u:USED]->(d)" in query
    assert params["rprops"]["params"] == '{"a": "1"}'
    assert params["rprops"]["code_version"] == "unknown"
    assert driver.closed


def test_lucene_escape() -> None:
    assert _lucene_escape("a+b (c)") == "a\\+b \\(c\\)"


def test_code_version_of_an_installed_build_is_its_release(tmp_path: Path) -> None:
    # modephere's code does not live under tmp_path, even if tmp_path were a
    # git repository: the run is the installed release, not that repo's commit
    sha, dirty = code_version(tmp_path)
    assert sha == f"modephere=={version('modephere')}"
    assert dirty is False


def test_code_version_of_the_checkout_is_its_commit() -> None:
    sha, _ = code_version(PROJECT_ROOT)
    assert len(sha) == 40 and int(sha, 16) >= 0


def test_code_version_when_git_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "_is_checkout", lambda _repo: True)
    assert code_version(tmp_path) == ("unknown", True)


def test_connect_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import neo4j
    from neo4j.exceptions import ServiceUnavailable

    closed: list[bool] = []

    class Unreachable:
        def verify_connectivity(self) -> None:
            raise ServiceUnavailable("connection refused")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: Unreachable())
    settings = SimpleNamespace(
        uri="bolt://127.0.0.1:1",
        user="neo4j",
        password=SimpleNamespace(get_secret_value=lambda: "x"),
        database="neo4j",
        batch_size=10,
    )
    with pytest.raises(ResourceError, match="cannot reach Neo4j"):
        KnowledgeGraph.connect(settings)  # type: ignore[arg-type]
    assert closed == [True]


def test_load_sequence(staging: Path) -> None:
    driver = FakeDriver()
    summary = load_chembl(KnowledgeGraph(driver), staging)  # type: ignore[arg-type]
    assert summary.dataset == "chembl_99"
    assert summary.nodes["Compound"] == 5
    assert summary.relationships["HAS_BIOACTIVITY"] == 2
    queries = [q for q, _ in driver.calls]
    assert queries[0].startswith("CREATE CONSTRAINT")
    used = [i for i, q in enumerate(queries) if "USED" in q]
    assert len(used) == 2 and used[-1] == len(queries) - 1  # finished_at last
    assert driver.calls[-1][1]["rprops"]["finished_at"] is not None
    # every node batch precedes every relationship batch
    first_rel = next(i for i, q in enumerate(queries) if "MATCH (a:" in q)
    assert all("MATCH (a:" not in q for q in queries[:first_rel])
    assert all("MERGE (n:" not in q for q in queries[first_rel:-1])
