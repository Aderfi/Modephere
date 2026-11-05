"""The kg CLI and remove_source, against a scripted fake driver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from modephere.config import get_settings
from modephere.core import ResourceError
from modephere.kg import KnowledgeGraph
from modephere.kg.__main__ import main

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class _Session:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run(self, query: str, **params: Any) -> SimpleNamespace:
        self.log.append(query)
        counters = SimpleNamespace(
            relationships_deleted=3, nodes_deleted=2, properties_set=1
        )
        return SimpleNamespace(consume=lambda: SimpleNamespace(counters=counters))


class ScriptedDriver:
    def __init__(self, audit: dict[str, int]) -> None:
        self.audit = audit
        self.queries: list[str] = []

    def execute_query(
        self, query: str, params: dict[str, Any], database_: str
    ) -> tuple[list[_Record], None, None]:
        self.queries.append(query)
        if "nodes_missing" in query:
            return [_Record(self.audit)], None, None
        if "fulltext" in query:
            assert params["query"] == "gleevec\\-x"
            row = {
                "chembl_id": "CHEMBL941",
                "pref_name": "IMATINIB",
                "max_phase": 4.0,
                "inchikey": "K",
                "score": 3.2,
            }
            return [_Record(row)], None, None
        if "labels(n)" in query:
            return [_Record({"label": "Compound", "n": 5})], None, None
        if "RETURN count" in query and "rows" in params:
            return [_Record({"written": len(params["rows"])})], None, None
        return [], None, None

    def session(self, database: str) -> _Session:
        return _Session(self.queries)

    def close(self) -> None:
        pass


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch) -> ScriptedDriver:
    fake = ScriptedDriver({"nodes_missing": 0, "rels_missing": 0, "unknown_sources": 0})
    monkeypatch.setattr(
        KnowledgeGraph,
        "connect",
        classmethod(lambda cls, settings=None: cls(fake)),
    )
    return fake


def test_remove_source_counts(driver: ScriptedDriver) -> None:
    out = KnowledgeGraph(driver).remove_source("chembl_99")  # type: ignore[arg-type]
    assert out == {
        "relationships_deleted": 3,
        "relationships_updated": 1,
        "nodes_deleted": 2,
        "nodes_updated": 1,
    }
    assert sum("IN TRANSACTIONS" in q for q in driver.queries) == 2


def test_cli_commands(
    driver: ScriptedDriver, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["stats"]) == 0
    assert "Compound" in capsys.readouterr().out
    assert main(["audit"]) == 0
    assert main(["search", "gleevec-x"]) == 0
    assert "CHEMBL941" in capsys.readouterr().out
    assert main(["remove-source", "chembl_99"]) == 1  # needs --yes
    assert main(["remove-source", "chembl_99", "--yes"]) == 0


def test_cli_audit_failure(driver: ScriptedDriver) -> None:
    driver.audit = {"nodes_missing": 1, "rels_missing": 0, "unknown_sources": 0}
    assert main(["audit"]) == 2


def test_cli_load(driver: ScriptedDriver, staging: Path) -> None:
    assert main(["load-chembl", "--staging", str(staging.parent)]) == 0
    assert (
        main(
            ["load-chembl", "--staging", str(staging.parent), "--release", "chembl_99"]
        )
        == 0
    )
    assert any("MERGE (n:Compound" in q for q in driver.queries)


def test_cli_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(cls: type, settings: object = None) -> KnowledgeGraph:
        raise ResourceError("cannot reach Neo4j")

    monkeypatch.setattr(KnowledgeGraph, "connect", classmethod(fail))
    assert main(["stats"]) == 1


@pytest.fixture
def kg_settings(
    tmp_path: Path, staging: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    monkeypatch.setenv("MODEPHERE_PATHS__CHEMBL", str(staging.parent))
    monkeypatch.setenv("MODEPHERE_PATHS__CACHE", str(tmp_path / "cache"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("kg_settings")
class TestLinkCommands:
    def test_link_drugs_from_file(self, driver: ScriptedDriver, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        tsv.write_text("name\tcid\naspirin\t2244\ngleevec\t\n")
        assert main(["link-drugs", "--drugs-file", str(tsv), "--offline"]) == 0
        assert any("MERGE (n:Drug" in q for q in driver.queries)
        assert any("SAME_AS" in q for q in driver.queries)

    def test_link_drugs_needs_drugs(self, driver: ScriptedDriver) -> None:
        # neither a file nor another graph: the Drug nodes of this one (none here)
        assert main(["link-drugs", "--offline"]) == 1
        assert main(["link-drugs", "--drugs-uri", "bolt://elsewhere:7687"]) == 1

    def test_link_library(self, driver: ScriptedDriver, tmp_path: Path) -> None:
        assert main(["link-library", "--library", str(tmp_path / "none")]) == 1
