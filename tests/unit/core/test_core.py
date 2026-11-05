"""Exceptions, settings priority and the dedicated file loggers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modephere.config import Settings
from modephere.core import (
    CompoundError,
    DataError,
    ModephereException,
    build_file_logger,
)
from modephere.types import BuildStage, FailureCategory

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestExceptions:
    def test_details_in_str(self) -> None:
        e = DataError("bad row", {"row": 3})
        assert str(e) == "bad row | Context: {row=3}"
        assert isinstance(e, ModephereException)

    def test_compound_error(self) -> None:
        e = CompoundError(BuildStage.RESOLVE, FailureCategory.CACHE_MISS, "not fetched")
        assert (e.stage, e.category, e.reason) == (
            BuildStage.RESOLVE,
            FailureCategory.CACHE_MISS,
            "not fetched",
        )
        assert isinstance(e, DataError)


class TestSettings:
    def test_toml_defaults(self) -> None:
        s = Settings()
        assert s.pubchem.max_requests_per_second == 5.0
        assert s.paths.library.is_absolute()

    def test_env_beats_toml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MODEPHERE_PUBCHEM__BATCH_SIZE", "7")
        monkeypatch.setenv("MODEPHERE_PATHS__LIBRARY", str(tmp_path))
        s = Settings()
        assert s.pubchem.batch_size == 7
        assert s.paths.library == tmp_path

    def test_arguments_beat_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEPHERE_SEED", "1")
        assert Settings(seed=2).seed == 2


class TestFileLogger:
    def test_writes_and_does_not_propagate(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.log"
        log = build_file_logger("modephere.test.audit", path)
        log.warning("one")
        log = build_file_logger("modephere.test.audit", path)  # idempotent
        log.warning("two")
        lines = path.read_text().splitlines()
        assert len(lines) == 2 and lines[1].endswith("two")
        assert not log.propagate

    def test_without_path_drops_records(self) -> None:
        log = build_file_logger("modephere.test.silent", None)
        assert log.handlers == []
