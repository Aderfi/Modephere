"""Numpy graphs: what the chemistry layers emit and the storage layer writes.

They mirror :mod:`modephere.types.tensors` without torch (PLAN C3): the
featurization layer converts them with ``MoleculeTensors.from_graph`` and
friends. Validation is the same: layout widths, index ranges, no self-loops,
every edge stored in both directions.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
from pydantic import Field, PositiveInt, model_validator

from modephere.types.arrays import Float32Array, Int64Array
from modephere.types.base import ArrayModel
from modephere.types.layouts import (
    MOLECULE_GRAPH_V1,
    PHARMACOPHORE_GRAPH_V1,
    GraphLayout,
)


def _expect_shape(a: np.ndarray, shape: tuple[int | None, ...], name: str) -> None:
    ok = a.ndim == len(shape) and all(
        want is None or got == want for got, want in zip(a.shape, shape, strict=True)
    )
    if not ok:
        pretty = ", ".join("*" if s is None else str(s) for s in shape)
        msg = f"{name} has shape {a.shape}, expected [{pretty}]"
        raise ValueError(msg)


class GraphArrays(ArrayModel):
    """Common shape of a homogeneous graph; subclasses fix ``LAYOUT``."""

    LAYOUT: ClassVar[GraphLayout]

    x: Float32Array = Field(..., description="[N, node_dim] node features.")
    edge_index: Int64Array = Field(..., description="[2, E], both directions.")
    edge_attr: Float32Array = Field(..., description="[E, edge_dim].")
    global_feats: Float32Array = Field(..., description="[1, global_dim].")

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @model_validator(mode="after")
    def _check_graph(self) -> Self:
        layout = getattr(type(self), "LAYOUT", None)
        if layout is None:
            msg = f"{type(self).__name__} has no LAYOUT; use a concrete subclass"
            raise TypeError(msg)
        _expect_shape(self.x, (None, layout.node_dim), "x")
        _expect_shape(self.edge_index, (2, None), "edge_index")
        _expect_shape(self.edge_attr, (self.num_edges, layout.edge_dim), "edge_attr")
        _expect_shape(self.global_feats, (1, layout.global_dim), "global_feats")
        if self.num_edges:
            n, (src, dst) = self.num_nodes, self.edge_index
            if self.edge_index.min() < 0 or self.edge_index.max() >= n:
                msg = f"edge_index refers to nodes outside [0, {n})"
                raise ValueError(msg)
            if (src == dst).any():
                msg = "edge_index contains self-loops"
                raise ValueError(msg)
            if not np.array_equal(np.sort(src * n + dst), np.sort(dst * n + src)):
                msg = "edge_index must hold every edge in both directions"
                raise ValueError(msg)
        return self


class MoleculeGraph(GraphArrays):
    """Atom/bond graph of the working form (heavy atoms), ``molecule/v1``."""

    LAYOUT: ClassVar[GraphLayout] = MOLECULE_GRAPH_V1

    atomic_numbers: Int64Array = Field(..., description="[N] (SchNet/ViSNet ``z``).")

    @model_validator(mode="after")
    def _check_z(self) -> Self:
        _expect_shape(self.atomic_numbers, (self.num_nodes,), "atomic_numbers")
        if self.num_nodes and self.atomic_numbers.min() < 1:
            msg = "atomic numbers must be >= 1"
            raise ValueError(msg)
        return self


class PharmacophoreGraph(GraphArrays):
    """Pharmacophore feature graph, ``pharmacophore/v1``."""

    LAYOUT: ClassVar[GraphLayout] = PHARMACOPHORE_GRAPH_V1

    atom2feat: Int64Array = Field(
        ..., description="[M, 2]: (atom index in the molecule graph, feature node)."
    )
    num_drug_atoms: PositiveInt

    @model_validator(mode="after")
    def _check_members(self) -> Self:
        _expect_shape(self.atom2feat, (None, 2), "atom2feat")
        atoms, feats = self.atom2feat[:, 0], self.atom2feat[:, 1]
        if atoms.size and (atoms.min() < 0 or atoms.max() >= self.num_drug_atoms):
            msg = f"atom2feat atoms fall outside [0, {self.num_drug_atoms})"
            raise ValueError(msg)
        if set(feats.tolist()) != set(range(self.num_nodes)):
            msg = "every feature node needs member atoms, and only those nodes"
            raise ValueError(msg)
        if len(np.unique(self.atom2feat, axis=0)) != len(self.atom2feat):
            msg = "atom2feat contains duplicate (atom, feature) pairs"
            raise ValueError(msg)
        return self


__all__ = ["GraphArrays", "MoleculeGraph", "PharmacophoreGraph"]
