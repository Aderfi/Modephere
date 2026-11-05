"""Layout contracts: pinned explicitly, since changing them silently would
invalidate every library and model built on them (dev_Pgen parity, PLAN Q2)."""

from __future__ import annotations

from modephere.types import (
    MOLECULE_GRAPH_V1,
    PHARMACOPHORE_GRAPH_V1,
    PharmacophoreFamily,
)


class TestPharmacophoreFamily:
    def test_order_matches_dev_pgen(self) -> None:
        # dev_Pgen src/data/library/pharmacophore.py::_FAMILIES
        assert [f.value for f in PharmacophoreFamily] == [
            "Donor",
            "Acceptor",
            "PosIonizable",
            "NegIonizable",
            "Aromatic",
            "Hydrophobe",
        ]
        assert [f.one_hot_index for f in PharmacophoreFamily] == list(range(6))

    def test_directional_families(self) -> None:
        directional = {f for f in PharmacophoreFamily if f.is_directional}
        assert directional == {
            PharmacophoreFamily.DONOR,
            PharmacophoreFamily.ACCEPTOR,
            PharmacophoreFamily.AROMATIC,
        }


class TestLayouts:
    def test_molecule_v1_is_dev_pgen_drug_graph(self) -> None:
        assert (
            MOLECULE_GRAPH_V1.node_dim,
            MOLECULE_GRAPH_V1.edge_dim,
            MOLECULE_GRAPH_V1.global_dim,
        ) == (61, 18, 1038)
        assert MOLECULE_GRAPH_V1.key == "molecule/v1"

    def test_pharmacophore_v1_is_dev_pgen_pharm_graph(self) -> None:
        assert (
            PHARMACOPHORE_GRAPH_V1.node_dim,
            PHARMACOPHORE_GRAPH_V1.edge_dim,
            PHARMACOPHORE_GRAPH_V1.global_dim,
        ) == (11, 13, 258)
        # 6 families + "other" + 4 member stats; 6 counts + 21 pairs x 12 bins.
        assert PHARMACOPHORE_GRAPH_V1.node_dim == len(PharmacophoreFamily) + 1 + 4
        assert PHARMACOPHORE_GRAPH_V1.global_dim == 6 + 21 * 12
