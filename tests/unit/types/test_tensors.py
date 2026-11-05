"""Torch / PyG contracts: validation, conversion and batching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

from pydantic import ValidationError  # noqa: E402

from modephere.types import (  # noqa: E402
    MOLECULE_GRAPH_V1,
    PHARMACOPHORE_GRAPH_V1,
    ConformerTensors,
    HeteroTensors,
    MoleculeTensors,
    PharmacophoreData,
    PharmacophoreTensors,
)
from modephere.types.tensors import BOND, MEMBER_OF, PHARM_DIST  # noqa: E402

from .conftest import make_ensemble  # noqa: E402

if TYPE_CHECKING:
    from modephere.types import ConformerEnsemble

Batch = pyg_data.Batch


def molecule(n: int = 3, **overrides: object) -> MoleculeTensors:
    """A path graph 0-1-...-(n-1), both directions."""
    src = list(range(n - 1))
    dst = list(range(1, n))
    edge_index = torch.tensor([src + dst, dst + src])
    fields: dict[str, object] = {
        "x": torch.rand(n, MOLECULE_GRAPH_V1.node_dim),
        "edge_index": edge_index,
        "edge_attr": torch.rand(edge_index.shape[1], MOLECULE_GRAPH_V1.edge_dim),
        "global_feats": torch.rand(1, MOLECULE_GRAPH_V1.global_dim),
    }
    fields.update(overrides)
    return MoleculeTensors(**fields)


def pharmacophore(**overrides: object) -> PharmacophoreTensors:
    """Two features over a 3-atom molecule: {0, 1} and {2}."""
    fields: dict[str, object] = {
        "x": torch.rand(2, PHARMACOPHORE_GRAPH_V1.node_dim),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_attr": torch.rand(2, PHARMACOPHORE_GRAPH_V1.edge_dim),
        "global_feats": torch.rand(1, PHARMACOPHORE_GRAPH_V1.global_dim),
        "atom2feat": torch.tensor([[0, 0], [1, 0], [2, 1]]),
        "num_drug_atoms": 3,
    }
    fields.update(overrides)
    return PharmacophoreTensors(**fields)


class TestTensorFields:
    def test_dtypes_coerced(self) -> None:
        m = molecule(x=torch.ones(3, 61, dtype=torch.float64))
        assert m.x.dtype == torch.float32
        assert m.edge_index.dtype == torch.int64

    def test_lists_accepted(self) -> None:
        m = molecule(edge_index=[[0, 1], [1, 0]], edge_attr=[[0.0] * 18] * 2)
        assert m.num_edges == 2

    def test_float_indices_rejected(self) -> None:
        with pytest.raises(ValidationError, match="integer"):
            molecule(edge_index=torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    def test_nan_rejected(self) -> None:
        x = torch.rand(3, 61)
        x[0, 0] = float("nan")
        with pytest.raises(ValidationError, match="NaN"):
            molecule(x=x)


class TestGraphTensors:
    def test_widths_enforced(self) -> None:
        with pytest.raises(ValidationError, match=r"x has shape \(3, 60\)"):
            molecule(x=torch.rand(3, 60))
        with pytest.raises(ValidationError, match="global_feats"):
            molecule(global_feats=torch.rand(1, 1024))

    def test_edge_attr_rows_match_edges(self) -> None:
        with pytest.raises(ValidationError, match="edge_attr"):
            molecule(edge_attr=torch.rand(3, 18))

    def test_out_of_range_edges(self) -> None:
        bad = torch.tensor([[0, 3], [3, 0]])
        with pytest.raises(ValidationError, match="outside"):
            molecule(edge_index=bad, edge_attr=torch.rand(2, 18))

    def test_self_loops(self) -> None:
        with pytest.raises(ValidationError, match="self-loops"):
            molecule(edge_index=torch.tensor([[1], [1]]), edge_attr=torch.rand(1, 18))

    def test_edges_need_both_directions(self) -> None:
        with pytest.raises(ValidationError, match="both directions"):
            molecule(edge_index=torch.tensor([[0], [1]]), edge_attr=torch.rand(1, 18))

    def test_isolated_atom(self) -> None:
        m = molecule(
            n=1,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 18)),
        )
        assert (m.num_nodes, m.num_edges) == (1, 0)

    def test_pos_shape(self) -> None:
        assert molecule(pos=torch.rand(3, 3)).pos is not None
        with pytest.raises(ValidationError, match="pos"):
            molecule(pos=torch.rand(2, 3))

    def test_base_class_is_abstract(self) -> None:
        from modephere.types import GraphTensors

        with pytest.raises(TypeError, match="LAYOUT"):
            GraphTensors(
                x=torch.rand(1, 1),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1)),
                global_feats=torch.rand(1, 1),
            )

    def test_data_round_trip(self) -> None:
        m = molecule(pos=torch.rand(3, 3))
        data = m.to_data(inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        assert data.inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        assert MoleculeTensors.from_data(data) == m

    def test_equality_is_exact(self) -> None:
        m = molecule()
        assert m == m.model_copy()
        assert m != molecule()  # different random features


class TestPharmacophoreTensors:
    def test_membership_ranges(self) -> None:
        with pytest.raises(ValidationError, match="outside"):
            pharmacophore(atom2feat=torch.tensor([[0, 0], [3, 1]]))

    def test_every_feature_needs_atoms(self) -> None:
        with pytest.raises(ValidationError, match="member atoms"):
            pharmacophore(atom2feat=torch.tensor([[0, 0], [1, 0]]))

    def test_duplicate_membership(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            pharmacophore(atom2feat=torch.tensor([[0, 0], [0, 0], [2, 1]]))

    def test_featureless_graph(self) -> None:
        p = pharmacophore(
            x=torch.empty((0, 11)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 13)),
            atom2feat=torch.empty((0, 2), dtype=torch.long),
        )
        assert p.num_nodes == 0

    def test_batching_offsets_atom2feat(self) -> None:
        data = pharmacophore().to_data()
        assert isinstance(data, PharmacophoreData)
        batch = Batch.from_data_list([data, data])
        assert batch.atom2feat.tolist() == [
            [0, 0], [1, 0], [2, 1],  # first graph
            [3, 2], [4, 2], [5, 3],  # second: +3 atoms, +2 features
        ]  # fmt: skip
        assert batch.global_feats.shape == (2, PHARMACOPHORE_GRAPH_V1.global_dim)
        assert PharmacophoreTensors.from_data(batch.get_example(1)) is not None


class TestHeteroTensors:
    def test_atom_counts_must_agree(self) -> None:
        with pytest.raises(ValidationError, match="indexes 3 atoms"):
            HeteroTensors(molecule=molecule(n=4), pharmacophore=pharmacophore())

    def test_to_hetero_data(self) -> None:
        h = HeteroTensors(molecule=molecule(), pharmacophore=pharmacophore())
        data = h.to_hetero_data(inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        data.validate(raise_on_error=True)
        assert data["atom"].num_nodes == 3
        assert data["pharm"].num_nodes == 2
        assert data[MEMBER_OF].edge_index.tolist() == [[0, 1, 2], [0, 0, 1]]
        assert data[BOND].edge_attr.shape == (4, 18)
        assert data[PHARM_DIST].edge_attr.shape == (2, 13)
        assert data.inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"

    def test_hetero_batching(self) -> None:
        h = HeteroTensors(molecule=molecule(), pharmacophore=pharmacophore())
        batch = Batch.from_data_list([h.to_hetero_data(), h.to_hetero_data()])
        member = batch[MEMBER_OF].edge_index.tolist()
        assert member == [[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 2, 3]]
        assert batch.molecule_global_feats.shape == (2, MOLECULE_GRAPH_V1.global_dim)


class TestConformerTensors:
    def test_from_ensemble_slices_heavy_atoms(
        self, ensemble: ConformerEnsemble
    ) -> None:
        conf = ConformerTensors.from_ensemble(ensemble)
        assert conf.positions.shape == (3, ensemble.num_heavy_atoms, 3)
        assert conf.positions.dtype == torch.float32
        assert conf.relative_energies is not None
        assert conf.relative_energies.tolist() == [0.0, 1.5, 4.0]
        assert torch.equal(
            conf.lowest(),
            torch.from_numpy(ensemble.positions[0, : ensemble.num_heavy_atoms].copy()),
        )

    def test_lowest_uses_energy(self) -> None:
        conf = ConformerTensors.from_ensemble(make_ensemble((3.0, 0.0, 1.0)))
        assert torch.equal(conf.lowest(), conf.positions[1])

    def test_sampling_is_seedable(self, ensemble: ConformerEnsemble) -> None:
        conf = ConformerTensors.from_ensemble(ensemble)
        draws = [conf.sample(torch.Generator().manual_seed(7)) for _ in range(2)]
        assert torch.equal(draws[0], draws[1])
        assert draws[0].shape == (ensemble.num_heavy_atoms, 3)

    def test_fills_molecule_pos(self, ensemble: ConformerEnsemble) -> None:
        conf = ConformerTensors.from_ensemble(ensemble)
        m = molecule(n=13, pos=conf.sample())
        assert m.pos is not None and m.pos.shape == (13, 3)

    def test_shape_checks(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            ConformerTensors(positions=torch.empty((0, 3, 3)))
        with pytest.raises(ValidationError, match="relative_energies"):
            ConformerTensors(positions=torch.rand(2, 3, 3), relative_energies=[0.0])
