"""Feature layouts: the widths every graph of a kind must have.

A layout pins the node / edge / graph-level feature widths of one graph kind.
Changing a width silently would invalidate every stored library and trained
model, so a change is a new ``version``, never an edit.

The v1 layouts are dev_Pgen's (PLAN Q2: schema-compatible 2D outputs):

* molecule: ``DRUG_NODE_DIM`` 61, ``DRUG_EDGE_DIM`` 18, ``DRUG_GLOBAL_DIM``
  1038 (14 physchem + 1024-bit ECFP4), from ``src/data/library/drugs.py``;
* pharmacophore: ``PHARM_NODE_DIM`` 11, ``PHARM_EDGE_DIM`` 13,
  ``PHARM_GLOBAL_DIM`` 258, from ``src/data/library/pharmacophore.py``.

The 3D pharmacophore distance-edge layout (radial basis width) is defined in
M5 (PLAN C4), not here.
"""

from __future__ import annotations

from pydantic import NonNegativeInt, PositiveInt

from modephere.types.base import FrozenModel
from modephere.types.scalars import NonEmptyStr


class GraphLayout(FrozenModel):
    """Feature widths of one graph kind, at one version."""

    name: NonEmptyStr
    version: PositiveInt
    node_dim: PositiveInt
    edge_dim: PositiveInt
    global_dim: NonNegativeInt

    @property
    def key(self) -> str:
        return f"{self.name}/v{self.version}"


class FingerprintLayout(FrozenModel):
    """Width of one bit-vector fingerprint, at one version (stored as on-bits)."""

    name: NonEmptyStr
    version: PositiveInt
    num_bits: PositiveInt

    @property
    def key(self) -> str:
        return f"{self.name}/v{self.version}"


MOLECULE_GRAPH_V1 = GraphLayout(
    name="molecule", version=1, node_dim=61, edge_dim=18, global_dim=1038
)
PHARMACOPHORE_GRAPH_V1 = GraphLayout(
    name="pharmacophore", version=1, node_dim=11, edge_dim=13, global_dim=258
)

#: RDKit Gobbi 2D pharmacophore fingerprint (``Gobbi_Pharm2D.factory`` sig size).
GOBBI_PHARM2D_V1 = FingerprintLayout(name="gobbi_pharm2d", version=1, num_bits=39972)
#: Modephere 3D feature-triplet fingerprint (hashed; see pharmacophore.fingerprints).
TRIPLET_PHARM3D_V1 = FingerprintLayout(name="triplet_pharm3d", version=1, num_bits=4096)

__all__ = [
    "GOBBI_PHARM2D_V1",
    "MOLECULE_GRAPH_V1",
    "PHARMACOPHORE_GRAPH_V1",
    "TRIPLET_PHARM3D_V1",
    "FingerprintLayout",
    "GraphLayout",
]
