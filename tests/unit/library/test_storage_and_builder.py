"""Storage round-trips, the builder end to end (fake PubChem), and the CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from rdkit import Chem
from safetensors.numpy import load_file, save_file

from modephere.config import get_settings
from modephere.core import StorageError
from modephere.library.__main__ import main
from modephere.library.builder import LibraryBuilder
from modephere.library.config import LibraryBuildConfig
from modephere.library.manifest import BuildManifest
from modephere.library.pipeline import PipelineOptions, build_compound
from modephere.library.storage import LibraryStore
from modephere.types import COMPOUNDS_SCHEMA, FailureCategory, check_frame

from .conftest import ASPIRIN, ASPIRIN_KEY, FAST, FakePubChem

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.types import CompoundArtifacts


class TestStore:
    def test_round_trip_is_exact(
        self, tmp_path: Path, aspirin: CompoundArtifacts
    ) -> None:
        store = LibraryStore(tmp_path)
        store.save(aspirin)
        assert store.exists(ASPIRIN_KEY)
        assert store.load(ASPIRIN_KEY) == aspirin
        assert store.inchikeys() == [ASPIRIN_KEY] and len(store) == 1

    def test_stereoisomer_ensemble_round_trip(
        self, tmp_path: Path, butanol: CompoundArtifacts
    ) -> None:
        store = LibraryStore(tmp_path)
        store.save(butanol)
        assert store.load(butanol.inchikey) == butanol

    def test_2d_only_round_trip(self, tmp_path: Path) -> None:
        art = build_compound(ASPIRIN, PipelineOptions(build_3d=False))
        store = LibraryStore(tmp_path)
        store.save(art)
        assert store.load(ASPIRIN_KEY) == art

    def test_missing_and_corrupt_files(
        self, tmp_path: Path, aspirin: CompoundArtifacts
    ) -> None:
        store = LibraryStore(tmp_path)
        with pytest.raises(StorageError, match="not in the library"):
            store.load(ASPIRIN_KEY)
        path = store.save(aspirin)
        arrays = load_file(path)
        arrays["molecule/x"] = arrays["molecule/x"][:, :60]  # wrong width
        save_file(arrays, path, metadata=json.loads(_metadata(path)))
        with pytest.raises(StorageError, match="corrupt"):
            store.load(ASPIRIN_KEY)

    def test_tables_upsert(
        self, tmp_path: Path, aspirin: CompoundArtifacts, butanol: CompoundArtifacts
    ) -> None:
        store = LibraryStore(tmp_path)
        store.upsert([aspirin])
        store.upsert([butanol, aspirin])  # replaces aspirin's rows
        compounds = store.read_table("compounds")
        check_frame(compounds, COMPOUNDS_SCHEMA, table="compounds")
        assert compounds["inchikey"].to_list() == sorted(
            [ASPIRIN_KEY, butanol.inchikey]
        )
        conformers = store.read_table("conformers")
        assert aspirin.conformers is not None and butanol.conformers is not None
        assert (
            conformers.height
            == aspirin.conformers.num_conformers + butanol.conformers.num_conformers
        )

    def test_sdf_export_keeps_each_stereoisomer(
        self, tmp_path: Path, butanol: CompoundArtifacts
    ) -> None:
        store = LibraryStore(tmp_path)
        store.save(butanol)
        n = store.export_sdf(butanol.inchikey, tmp_path / "b.sdf")
        assert butanol.conformers is not None
        mols = list(Chem.SDMolSupplier(str(tmp_path / "b.sdf"), removeHs=False))
        assert len(mols) == n == butanol.conformers.num_conformers
        by_isomer = {
            m.GetIntProp("stereoisomer"): Chem.MolToSmiles(Chem.RemoveHs(m))
            for m in mols
        }
        assert len(set(by_isomer.values())) == 2  # R and S written distinctly


def _metadata(path: Path) -> str:
    with path.open("rb") as f:
        size = int.from_bytes(f.read(8), "little")
        return json.dumps(json.loads(f.read(size))["__metadata__"])


def _config(tmp_path: Path, **kw: object) -> LibraryBuildConfig:
    defaults: dict[str, object] = {
        "library_root": tmp_path / "lib",
        "cache_dir": tmp_path / "cache",
        "failures_log": tmp_path / "logs" / "failures.log",
        "saturation_log": tmp_path / "logs" / "saturation.log",
        "pipeline": FAST,
    }
    return LibraryBuildConfig.model_validate(defaults | kw)


class TestBuilder:
    def test_needs_some_input(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="catalog"):
            _config(tmp_path)

    def test_smiles_build_offline(self, tmp_path: Path) -> None:
        cfg = _config(
            tmp_path,
            smiles=(ASPIRIN, "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "C", "bad(", "CC(O)CC"),
        )
        summary = LibraryBuilder(cfg).run()
        assert (summary.built, summary.duplicates, summary.failed) == (2, 1, 2)
        assert summary.failures == {
            FailureCategory.NO_PHARMACOPHORE_FEATURES: 1,
            FailureCategory.INVALID_SMILES: 1,
        }
        failures = pl.read_parquet(cfg.library_root / "failures.parquet")
        assert sorted(failures["category"].to_list()) == [
            "invalid_smiles",
            "no_pharmacophore_features",
        ]
        log = (tmp_path / "logs" / "failures.log").read_text()
        assert "nature=invalid_smiles" in log
        manifest = BuildManifest.load_or_empty(cfg.manifest_path)
        assert ASPIRIN_KEY in manifest.completed
        assert manifest.versions["rdkit"]

    def test_resume_skips_and_force_rebuilds(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, smiles=(ASPIRIN,))
        assert LibraryBuilder(cfg).run().built == 1
        assert LibraryBuilder(cfg).run().skipped == 1
        forced = cfg.model_copy(update={"force": True})
        assert LibraryBuilder(forced).run().built == 1

    def test_changed_options_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        LibraryBuilder(_config(tmp_path, smiles=(ASPIRIN,))).run()
        two_d = _config(
            tmp_path, smiles=(ASPIRIN,), pipeline=PipelineOptions(build_3d=False)
        )
        LibraryBuilder(two_d).run()
        assert "other pipeline options" in caplog.text

    def test_fetch_then_offline_build(self, tmp_path: Path) -> None:
        catalog = tmp_path / "q.tsv"
        catalog.write_text("cid\tname\n2244\t\n\tibuprofen\n\tunobtainium\n")
        fake = FakePubChem()
        cfg = _config(tmp_path, catalog=catalog)

        offline_first = LibraryBuilder(cfg).run()
        assert offline_first.failures == {FailureCategory.CACHE_MISS: 3}
        assert not fake.requests

        fetched = LibraryBuilder(cfg, transport=fake.transport).fetch()
        assert (fetched.resolved, fetched.failed) == (2, 1)
        requests = len(fake.requests)

        built = LibraryBuilder(cfg).run()  # offline again, from the cache
        assert built.built == 2
        assert built.failures == {FailureCategory.UNRESOLVED_IDENTIFIER: 1}
        assert len(fake.requests) == requests  # no network during the build
        registry = LibraryStore(cfg.library_root).read_table("compounds")
        assert set(registry["source"].to_list()) == {"pubchem"}
        assert set(registry["cid"].to_list()) == {2244, 3672}

    def test_online_build(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, names=("aspirin",), offline=False)
        summary = LibraryBuilder(cfg, transport=FakePubChem().transport).run()
        assert summary.built == 1


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    monkeypatch.setenv("MODEPHERE_PATHS__LOGS", str(tmp_path / "logs"))
    monkeypatch.setenv("MODEPHERE_PATHS__LIBRARY", str(tmp_path / "lib"))
    monkeypatch.setenv("MODEPHERE_PATHS__CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MODEPHERE_PATHS__CHEMBL", str(tmp_path / "chembl"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("isolated_settings")
class TestCli:
    def test_build_and_export(self, tmp_path: Path) -> None:
        args = [
            "build",
            "--smiles",
            ASPIRIN,
            "--num-conformers",
            "3",
            "--max-conformers",
            "3",
        ]
        assert main(args) == 0
        assert LibraryStore(tmp_path / "lib").inchikeys() == [ASPIRIN_KEY]
        assert main(["export-sdf", ASPIRIN_KEY, str(tmp_path / "a.sdf")]) == 0
        assert (tmp_path / "a.sdf").exists()

    def test_failures_exit_2(self) -> None:
        assert main(["build", "--smiles", "C", "--2d-only"]) == 2

    def test_errors_exit_1(self, tmp_path: Path) -> None:
        assert main(["build", "--catalog", str(tmp_path / "missing.tsv")]) == 1
        assert main(["export-sdf", ASPIRIN_KEY, str(tmp_path / "x.sdf")]) == 1


def test_numpy_arrays_in_store_are_read_only(
    tmp_path: Path, aspirin: CompoundArtifacts
) -> None:
    store = LibraryStore(tmp_path)
    store.save(aspirin)
    loaded = store.load(ASPIRIN_KEY)
    with pytest.raises(ValueError, match="read-only"):
        loaded.molecule.x[0, 0] = np.float32(1.0)
