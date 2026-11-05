"""Numpy graph types, fingerprint layouts and the CompoundArtifacts contracts."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from modephere.library.pipeline import PipelineOptions, build_compound
from modephere.types import (
    GOBBI_PHARM2D_V1,
    TRIPLET_PHARM3D_V1,
    CompoundArtifacts,
    MoleculeGraph,
    PharmacophoreGraph,
)
from modephere.types.tensors import MoleculeTensors, PharmacophoreTensors

from ..library.conftest import FAST

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture(scope="module")
def art() -> CompoundArtifacts:
    return build_compound(ASPIRIN, FAST)


def _replace(art: CompoundArtifacts, **update: object) -> CompoundArtifacts:
    fields = {name: getattr(art, name) for name in type(art).model_fields}
    return CompoundArtifacts(**(fields | update))


class TestLayouts:
    def test_fingerprint_widths_pinned(self) -> None:
        assert GOBBI_PHARM2D_V1.num_bits == 39972
        assert TRIPLET_PHARM3D_V1.num_bits == 4096


class TestGraphs:
    def test_molecule_checks(self, art: CompoundArtifacts) -> None:
        m = art.molecule
        with pytest.raises(ValidationError, match="atomic_numbers"):
            MoleculeGraph(**(_fields(m) | {"atomic_numbers": m.atomic_numbers[:-1]}))
        with pytest.raises(ValidationError, match="both directions"):
            MoleculeGraph(
                **(
                    _fields(m)
                    | {
                        "edge_index": m.edge_index[:, ::2],
                        "edge_attr": m.edge_attr[::2],
                    }
                )
            )

    def test_pharmacophore_checks(self, art: CompoundArtifacts) -> None:
        p = art.pharmacophore
        with pytest.raises(ValidationError, match="member atoms"):
            PharmacophoreGraph(**(_fields(p) | {"atom2feat": p.atom2feat[:-1]}))

    def test_tensor_conversion(self, art: CompoundArtifacts) -> None:
        m = MoleculeTensors.from_graph(art.molecule)
        assert m.z is not None and m.z.tolist() == art.molecule.atomic_numbers.tolist()
        p = PharmacophoreTensors.from_graph(art.pharmacophore)
        assert p.num_nodes == art.pharmacophore.num_nodes


def _fields(model: object) -> dict[str, object]:
    return {n: getattr(model, n) for n in type(model).model_fields}  # type: ignore[attr-defined]


class TestArtifacts:
    def test_2d_features_must_match_the_graph(self, art: CompoundArtifacts) -> None:
        p2d = art.pharmacophore_2d.model_copy(
            update={"features": art.pharmacophore_2d.features[:-1]}
        )
        with pytest.raises(ValidationError, match="do not match"):
            _replace(art, pharmacophore_2d=p2d)

    def test_one_3d_pharmacophore_per_conformer(self, art: CompoundArtifacts) -> None:
        with pytest.raises(ValidationError, match="one 3D pharmacophore"):
            _replace(art, pharmacophores_3d=art.pharmacophores_3d[:-1])

    def test_3d_needs_conformers(self, art: CompoundArtifacts) -> None:
        with pytest.raises(ValidationError, match="need a conformer"):
            _replace(art, conformers=None)

    def test_on_bits_sorted_and_in_range(self, art: CompoundArtifacts) -> None:
        with pytest.raises(ValidationError, match="increasing"):
            _replace(art, pharm2d_fingerprint=art.pharm2d_fingerprint[::-1].copy())
        with pytest.raises(ValidationError, match="outside"):
            _replace(art, pharm3d_fingerprint=np.array([5000]))

    def test_2d_only_is_valid(self) -> None:
        flat = build_compound(ASPIRIN, PipelineOptions(build_3d=False))
        assert not flat.has_3d
