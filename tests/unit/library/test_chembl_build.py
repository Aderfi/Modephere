"""Library builds that resolve through the staged ChEMBL release."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import polars as pl
import pytest

from modephere.chembl import bioactivity_labels
from modephere.config import get_settings
from modephere.library.__main__ import main
from modephere.library.builder import LibraryBuilder
from modephere.library.config import LibraryBuildConfig
from modephere.library.pipeline import PipelineOptions
from modephere.library.storage import LibraryStore

from .conftest import ASPIRIN, ASPIRIN_KEY

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

FLAT = PipelineOptions(build_3d=False)
IMATINIB_KEY = "KTUFNOKKBVMGRW-UHFFFAOYSA-N"


def _no_network(request: httpx.Request) -> httpx.Response:
    msg = f"PubChem must not be asked: {request.url}"
    raise AssertionError(msg)


def _config(tmp_path: Path, staging: Path | None, **kw: object) -> LibraryBuildConfig:
    return LibraryBuildConfig.model_validate(
        {
            "library_root": tmp_path / "lib",
            "cache_dir": tmp_path / "cache",
            "failures_log": tmp_path / "logs" / "failures.log",
            "saturation_log": tmp_path / "logs" / "saturation.log",
            "pipeline": FLAT,
            "chembl_staging": staging,
        }
        | kw
    )


def test_offline_build_from_chembl(tmp_path: Path, staging: Path) -> None:
    cfg = _config(
        tmp_path,
        staging,
        names=("Gleevec", "imatinib mesylate"),
        chembl_ids=("CHEMBL25",),
        smiles=(ASPIRIN,),
    )
    builder = LibraryBuilder(cfg, transport=httpx.MockTransport(_no_network))
    assert builder.fetch().failed == 0
    summary = builder.run()
    assert (summary.built, summary.duplicates, summary.failed) == (2, 2, 0)

    registry = LibraryStore(cfg.library_root).read_table("compounds").sort("inchikey")
    rows = {r["inchikey"]: r for r in registry.iter_rows(named=True)}
    imatinib = rows[IMATINIB_KEY]
    assert (imatinib["chembl_id"], imatinib["chembl_record_id"]) == (
        "CHEMBL941",
        "CHEMBL1642",
    )
    assert (imatinib["source"], imatinib["chembl_match"]) == ("chembl", "name")
    assert imatinib["chembl_release"] == "chembl_99"
    assert rows[ASPIRIN_KEY]["chembl_id"] == "CHEMBL25"

    labels = bioactivity_labels(staging, library=cfg.library_root)
    assert labels["inchikey"].to_list() == [IMATINIB_KEY]
    assert labels["CHEMBL1862"].to_list() == [pytest.approx(8.0)]


def test_without_chembl_names_need_pubchem(tmp_path: Path) -> None:
    cfg = _config(tmp_path, None, names=("Gleevec",))
    summary = LibraryBuilder(cfg).run()  # offline, nothing cached
    assert summary.failed == 1 and "cache_miss" in summary.failures


@pytest.fixture
def chembl_settings(
    tmp_path: Path, staging: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    for name in ("LOGS", "LIBRARY", "CACHE"):
        monkeypatch.setenv(f"MODEPHERE_PATHS__{name}", str(tmp_path / name.lower()))
    monkeypatch.setenv("MODEPHERE_PATHS__CHEMBL", str(staging.parent))
    get_settings.cache_clear()
    yield tmp_path / "library"
    get_settings.cache_clear()


def test_cli_uses_the_latest_release(chembl_settings: Path) -> None:
    assert main(["build", "--chembl-id", "CHEMBL1642", "--2d-only"]) == 0
    registry = pl.read_parquet(chembl_settings / "compounds.parquet")
    assert registry["chembl_id"].to_list() == ["CHEMBL941"]


def test_cli_no_chembl(chembl_settings: Path) -> None:
    # without the release a ChEMBL ID cannot be resolved: a configuration error
    assert main(["build", "--chembl-id", "CHEMBL25", "--2d-only", "--no-chembl"]) == 1
