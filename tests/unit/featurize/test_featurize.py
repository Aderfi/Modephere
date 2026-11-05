"""PyG conversion, the library dataset, batching and geometric-model smoke tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import polars as pl  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.nn.models import SchNet, ViSNet  # noqa: E402

from modephere.core import ConfigurationError, StorageError  # noqa: E402
from modephere.featurize import (  # noqa: E402
    LibraryDataset,
    atom_positions,
    feature_positions,
    hetero_data,
    molecule_data,
    pharmacophore_data,
)
from modephere.library.pipeline import PipelineOptions, build_compound  # noqa: E402
from modephere.library.storage import LibraryStore  # noqa: E402

from ..library.conftest import ASPIRIN, FAST  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.types import CompoundArtifacts


@pytest.fixture(scope="module")
def arts() -> list[CompoundArtifacts]:
    return [build_compound(s, FAST) for s in (ASPIRIN, "CC(O)CC", "CN1CCNCC1")]


@pytest.fixture
def store(tmp_path: Path, arts: list[CompoundArtifacts]) -> LibraryStore:
    s = LibraryStore(tmp_path)
    for a in arts:
        s.save(a)
    return s


class TestConvert:
    def test_molecule_data(self, arts: list[CompoundArtifacts]) -> None:
        art = arts[0]
        flat = molecule_data(art)
        assert flat.x.shape == (13, 61) and flat.z.tolist()[0] == 6
        assert "pos" not in flat
        geo = molecule_data(art, conformer=0)
        assert geo.pos.shape == (13, 3)
        assert geo.inchikey == art.inchikey

    def test_atom_and_feature_positions_share_a_conformer(
        self, arts: list[CompoundArtifacts]
    ) -> None:
        art = arts[0]
        assert art.conformers is not None
        last = art.conformers.num_conformers - 1
        atoms = atom_positions(art, last)
        feats = feature_positions(art, last)
        for k, feature in enumerate(art.pharmacophore_2d.features):
            centroid = atoms[list(feature.atom_ids)].mean(dim=0)
            assert torch.allclose(centroid, feats[k], atol=1e-5)

    def test_bad_conformer(self, arts: list[CompoundArtifacts]) -> None:
        with pytest.raises(StorageError, match="no conformer"):
            molecule_data(arts[0], conformer=999)
        flat = build_compound(ASPIRIN, PipelineOptions(build_3d=False))
        with pytest.raises(StorageError, match="2D-only"):
            pharmacophore_data(flat, conformer=0)

    def test_hetero(self, arts: list[CompoundArtifacts]) -> None:
        data = hetero_data(arts[0], conformer=0)
        data.validate(raise_on_error=True)
        assert data["pharm"].pos.shape == (arts[0].pharmacophore.num_nodes, 3)


class TestDataset:
    @pytest.mark.parametrize("kind", ["molecule", "pharmacophore", "hetero"])
    def test_batches(self, store: LibraryStore, kind: str) -> None:
        ds = LibraryDataset(store, kind=kind, conformer="sample")  # type: ignore[arg-type]
        batch = next(iter(DataLoader(ds, batch_size=3)))
        assert batch.num_graphs == 3

    def test_sampling_is_seedable(self, store: LibraryStore) -> None:
        def draw() -> Any:
            ds = LibraryDataset(
                store, conformer="sample", generator=torch.Generator().manual_seed(1)
            )
            return torch.cat([ds[i].pos for i in range(len(ds))])

        assert torch.equal(draw(), draw())

    def test_lowest_and_none(self, store: LibraryStore) -> None:
        lowest = LibraryDataset(store, conformer="lowest")
        art = lowest.artifacts(0)
        assert torch.equal(lowest[0].pos, atom_positions(art, 0))
        assert "pos" not in LibraryDataset(store, conformer="none")[0]

    def test_targets(self, store: LibraryStore, arts: list[CompoundArtifacts]) -> None:
        targets = pl.DataFrame(
            {
                "inchikey": [arts[0].inchikey, arts[1].inchikey],
                "a": [1.0, None],
                "b": [2, 3],
            }
        )
        ds = LibraryDataset(store, targets=targets)
        assert len(ds) == 2 and ds.target_names == ["a", "b"]  # unlabelled dropped
        y = next(iter(DataLoader(ds, batch_size=2))).y
        assert y.shape == (2, 2) and torch.isnan(y).sum() == 1

    @pytest.mark.parametrize(
        "targets",
        [
            pl.DataFrame({"key": ["x"], "a": [1.0]}),
            pl.DataFrame({"inchikey": ["x"]}),
            pl.DataFrame({"inchikey": ["x", "x"], "a": [1.0, 2.0]}),
        ],
    )
    def test_bad_targets(self, store: LibraryStore, targets: pl.DataFrame) -> None:
        with pytest.raises(ConfigurationError):
            LibraryDataset(store, targets=targets)

    def test_bad_options(self, store: LibraryStore) -> None:
        with pytest.raises(ConfigurationError):
            LibraryDataset(store, kind="atoms")  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            LibraryDataset(store, conformer="all")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "model",
    [
        lambda: SchNet(hidden_channels=16, num_filters=16, num_interactions=1),
        lambda: ViSNet(hidden_channels=16, num_layers=1, num_heads=2, num_rbf=8),
    ],
    ids=["SchNet", "ViSNet"],
)
def test_geometric_models_run(store: LibraryStore, model) -> None:  # noqa: ANN001
    batch = next(
        iter(DataLoader(LibraryDataset(store, conformer="sample"), batch_size=3))
    )
    out = model()(batch.z, batch.pos, batch.batch)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == (3, 1) and torch.isfinite(out).all()
