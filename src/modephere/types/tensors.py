"""Torch / PyG tensor contracts for the featurization layer (PLAN C2.4).

Needs the ``ml`` extra; ``modephere.types`` imports this module lazily, so the
chemistry layers never import torch.

The featurizers build these validated models, then convert them to PyG:

* :class:`MoleculeTensors` → ``Data`` (atom graph, layout ``molecule/v1``);
* :class:`PharmacophoreTensors` → :class:`PharmacophoreData` (feature graph,
  layout ``pharmacophore/v1``, plus the ``atom2feat`` membership map);
* :class:`HeteroTensors` → ``HeteroData`` with ``atom`` and ``pharm`` nodes;
* :class:`ConformerTensors` holds the ``[C, N, 3]`` heavy-atom coordinates a
  dataset samples ``pos`` from at load time (conformer augmentation).

Validation covers dtypes, the layout widths, index ranges, no self-loops, and
edges stored in both directions (the dev_Pgen convention).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Final, Self

import numpy as np
import torch
from pydantic import Field, PlainValidator, PositiveInt, model_validator
from torch_geometric.data import Data, HeteroData

from modephere.types.base import ArrayModel
from modephere.types.conformer import ConformerEnsemble
from modephere.types.graph import MoleculeGraph, PharmacophoreGraph
from modephere.types.layouts import (
    MOLECULE_GRAPH_V1,
    PHARMACOPHORE_GRAPH_V1,
    GraphLayout,
)

# --------------------------------------------------------------------------- #
# Tensor field types
# --------------------------------------------------------------------------- #


def _coerce_tensor(dtype: torch.dtype) -> Callable[[Any], torch.Tensor]:
    floating = dtype.is_floating_point

    def validate(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value
        else:
            arr = np.asarray(value)
            if arr.dtype == object:
                msg = "tensor input is ragged or contains non-numeric objects"
                raise ValueError(msg)
            t = torch.as_tensor(arr)
        if t.is_complex():
            msg = "complex tensors are not supported"
            raise ValueError(msg)
        # An empty list parses as float64; only non-empty inputs carry a kind.
        if (
            not floating
            and t.numel()
            and (t.is_floating_point() or t.dtype == torch.bool)
        ):
            msg = f"index tensors must be integer, got {t.dtype}"
            raise ValueError(msg)
        t = t.detach().to(dtype)
        if floating and not bool(torch.isfinite(t).all()):
            msg = "tensor contains NaN or infinite values"
            raise ValueError(msg)
        return t

    return validate


#: float32 tensor; integer/bool input is promoted, NaN/inf rejected.
type FloatTensor = Annotated[
    torch.Tensor, PlainValidator(_coerce_tensor(torch.float32))
]
#: int64 tensor; float/bool input is rejected (no silent truncation).
type LongTensor = Annotated[torch.Tensor, PlainValidator(_coerce_tensor(torch.int64))]


def _tensor(a: np.ndarray) -> torch.Tensor:
    # Stored arrays are read-only; torch needs a writable buffer.
    return torch.from_numpy(np.array(a))


def _expect_shape(t: torch.Tensor, shape: tuple[int | None, ...], name: str) -> None:
    ok = t.dim() == len(shape) and all(
        want is None or got == want for got, want in zip(t.shape, shape, strict=True)
    )
    if not ok:
        pretty = ", ".join("*" if s is None else str(s) for s in shape)
        msg = f"{name} has shape {tuple(t.shape)}, expected [{pretty}]"
        raise ValueError(msg)


# --------------------------------------------------------------------------- #
# Heterogeneous graph vocabulary
# --------------------------------------------------------------------------- #

type EdgeType = tuple[str, str, str]

ATOM: Final = "atom"
PHARM: Final = "pharm"
BOND: Final[EdgeType] = (ATOM, "bond", ATOM)
MEMBER_OF: Final[EdgeType] = (ATOM, "member_of", PHARM)
PHARM_DIST: Final[EdgeType] = (PHARM, "dist", PHARM)

# --------------------------------------------------------------------------- #
# Homogeneous graphs
# --------------------------------------------------------------------------- #


class PharmacophoreData(Data):
    """``Data`` that offsets ``atom2feat`` correctly when batched.

    PyG only offsets attributes whose name contains ``index``. Without this, a
    batched ``atom2feat`` would point every graph at the first graph's atoms.
    Column 0 is offset by the molecule's atom count, column 1 by the feature
    count.
    """

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "atom2feat":
            return torch.tensor([[int(self.num_drug_atoms), int(self.num_nodes)]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "atom2feat":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class GraphTensors(ArrayModel):
    """Common shape of a homogeneous graph; subclasses fix ``LAYOUT``."""

    LAYOUT: ClassVar[GraphLayout]
    DATA_CLASS: ClassVar[type[Data]] = Data

    x: FloatTensor = Field(..., description="[N, node_dim] node features.")
    edge_index: LongTensor = Field(..., description="[2, E], both directions.")
    edge_attr: FloatTensor = Field(..., description="[E, edge_dim] edge features.")
    global_feats: FloatTensor = Field(..., description="[1, global_dim].")
    pos: FloatTensor | None = Field(default=None, description="[N, 3] in Å.")

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
        if self.pos is not None:
            _expect_shape(self.pos, (self.num_nodes, 3), "pos")
        if self.num_edges:
            self._check_edges()
        return self

    def _check_edges(self) -> None:
        n, (src, dst) = self.num_nodes, self.edge_index
        if int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= n:
            msg = f"edge_index refers to nodes outside [0, {n})"
            raise ValueError(msg)
        if bool((src == dst).any()):
            msg = "edge_index contains self-loops"
            raise ValueError(msg)
        forward, backward = src * n + dst, dst * n + src
        if not torch.equal(forward.sort().values, backward.sort().values):
            msg = "edge_index must hold every edge in both directions"
            raise ValueError(msg)

    def to_data(self, **attrs: Any) -> Data:
        """Convert to PyG; ``attrs`` adds graph metadata (e.g. ``inchikey``)."""
        fields = {name: getattr(self, name) for name in type(self).model_fields}
        fields = {k: v for k, v in fields.items() if v is not None}
        return self.DATA_CLASS(**fields, **attrs)

    @classmethod
    def from_data(cls, data: Data) -> Self:
        """Validate an existing PyG graph (e.g. one read back from storage)."""
        fields = {name: getattr(data, name, None) for name in cls.model_fields}
        return cls(**{k: v for k, v in fields.items() if v is not None})


class MoleculeTensors(GraphTensors):
    """Atom/bond graph of the working form (heavy atoms), ``molecule/v1``."""

    LAYOUT: ClassVar[GraphLayout] = MOLECULE_GRAPH_V1

    z: LongTensor | None = Field(
        default=None, description="[N] atomic numbers (SchNet / ViSNet input)."
    )

    @model_validator(mode="after")
    def _check_z(self) -> Self:
        if self.z is not None:
            _expect_shape(self.z, (self.num_nodes,), "z")
        return self

    @classmethod
    def from_graph(cls, graph: MoleculeGraph, pos: torch.Tensor | None = None) -> Self:
        """Convert the numpy graph the chemistry layer emits."""
        return cls(
            x=_tensor(graph.x),
            edge_index=_tensor(graph.edge_index),
            edge_attr=_tensor(graph.edge_attr),
            global_feats=_tensor(graph.global_feats),
            z=_tensor(graph.atomic_numbers),
            pos=pos,
        )


class PharmacophoreTensors(GraphTensors):
    """Pharmacophore feature graph, ``pharmacophore/v1``.

    Node ``i`` is feature ``i`` of the compound's canonical feature order
    (:class:`~modephere.types.pharmacophore.Pharmacophore2D`).
    """

    LAYOUT: ClassVar[GraphLayout] = PHARMACOPHORE_GRAPH_V1
    DATA_CLASS: ClassVar[type[Data]] = PharmacophoreData

    atom2feat: LongTensor = Field(
        ..., description="[M, 2]: (atom index in the molecule graph, feature node)."
    )
    num_drug_atoms: PositiveInt = Field(
        ..., description="Node count of the matching molecule graph."
    )

    @model_validator(mode="after")
    def _check_members(self) -> Self:
        _expect_shape(self.atom2feat, (None, 2), "atom2feat")
        atoms, feats = self.atom2feat[:, 0], self.atom2feat[:, 1]
        if self.atom2feat.numel() and (
            int(atoms.min()) < 0 or int(atoms.max()) >= self.num_drug_atoms
        ):
            msg = f"atom2feat atoms fall outside [0, {self.num_drug_atoms})"
            raise ValueError(msg)
        if set(feats.tolist()) != set(range(self.num_nodes)):
            msg = "every feature node needs member atoms, and only those nodes"
            raise ValueError(msg)
        if torch.unique(self.atom2feat, dim=0).shape[0] != self.atom2feat.shape[0]:
            msg = "atom2feat contains duplicate (atom, feature) pairs"
            raise ValueError(msg)
        return self

    @classmethod
    def from_graph(
        cls, graph: PharmacophoreGraph, pos: torch.Tensor | None = None
    ) -> Self:
        """Convert the numpy graph the chemistry layer emits."""
        return cls(
            x=_tensor(graph.x),
            edge_index=_tensor(graph.edge_index),
            edge_attr=_tensor(graph.edge_attr),
            global_feats=_tensor(graph.global_feats),
            atom2feat=_tensor(graph.atom2feat),
            num_drug_atoms=graph.num_drug_atoms,
            pos=pos,
        )


# --------------------------------------------------------------------------- #
# Heterogeneous graph
# --------------------------------------------------------------------------- #


class HeteroTensors(ArrayModel):
    """A molecule graph and its pharmacophore graph, joined by membership."""

    molecule: MoleculeTensors
    pharmacophore: PharmacophoreTensors

    @model_validator(mode="after")
    def _check_join(self) -> Self:
        if self.pharmacophore.num_drug_atoms != self.molecule.num_nodes:
            msg = (
                f"pharmacophore indexes {self.pharmacophore.num_drug_atoms} atoms, "
                f"molecule graph has {self.molecule.num_nodes}"
            )
            raise ValueError(msg)
        return self

    def to_hetero_data(self, **attrs: Any) -> HeteroData:
        """Convert to PyG ``HeteroData``.

        Only ``atom -> pharm`` membership edges are stored; add the reverse
        with ``torch_geometric.transforms.ToUndirected`` if a model needs it.
        """
        mol, ph = self.molecule, self.pharmacophore
        h = HeteroData()
        h[ATOM].x = mol.x
        if mol.z is not None:
            h[ATOM].z = mol.z
        h[PHARM].x = ph.x
        h[PHARM].num_nodes = ph.num_nodes  # explicit: may be 0
        if mol.pos is not None:
            h[ATOM].pos = mol.pos
        if ph.pos is not None:
            h[PHARM].pos = ph.pos
        h[BOND].edge_index = mol.edge_index
        h[BOND].edge_attr = mol.edge_attr
        h[MEMBER_OF].edge_index = ph.atom2feat.t().contiguous()
        h[PHARM_DIST].edge_index = ph.edge_index
        h[PHARM_DIST].edge_attr = ph.edge_attr
        h.molecule_global_feats = mol.global_feats
        h.pharmacophore_global_feats = ph.global_feats
        for key, value in attrs.items():
            setattr(h, key, value)
        return h


# --------------------------------------------------------------------------- #
# Conformers
# --------------------------------------------------------------------------- #


class ConformerTensors(ArrayModel):
    """Heavy-atom coordinates of a conformer ensemble, for ``pos`` sampling."""

    positions: FloatTensor = Field(..., description="[C, N, 3] in Å.")
    relative_energies: FloatTensor | None = Field(
        default=None, description="[C] kcal/mol above the lowest conformer."
    )

    @property
    def num_conformers(self) -> int:
        return int(self.positions.shape[0])

    @model_validator(mode="after")
    def _check(self) -> Self:
        _expect_shape(self.positions, (None, None, 3), "positions")
        if self.num_conformers == 0:
            msg = "an ensemble needs at least one conformer"
            raise ValueError(msg)
        if self.relative_energies is not None:
            _expect_shape(
                self.relative_energies, (self.num_conformers,), "relative_energies"
            )
        return self

    @classmethod
    def from_ensemble(cls, ensemble: ConformerEnsemble) -> Self:
        """Heavy-atom slice of an ensemble (it lines up with ``MoleculeTensors``)."""
        heavy = np.array(ensemble.positions[:, : ensemble.num_heavy_atoms])
        energies = ensemble.energies
        relative = None if energies is None else energies - energies.min()
        return cls(positions=torch.from_numpy(heavy), relative_energies=relative)

    def lowest(self) -> torch.Tensor:
        """``[N, 3]`` coordinates of the lowest-energy conformer."""
        return self.positions[self.lowest_index()]

    def lowest_index(self) -> int:
        """Position of the lowest-energy conformer (0 without energies)."""
        if self.relative_energies is None:
            return 0
        return int(torch.argmin(self.relative_energies))

    def sample_index(self, generator: torch.Generator | None = None) -> int:
        """Position of a uniformly drawn conformer.

        Draw the index once and use it for every per-conformer tensor (atom
        ``pos`` and pharmacophore ``pos``) so they stay in one conformation.
        """
        return int(torch.randint(self.num_conformers, (1,), generator=generator))

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """``[N, 3]`` coordinates of a uniformly drawn conformer."""
        return self.positions[self.sample_index(generator)]


__all__ = [
    "ATOM",
    "BOND",
    "MEMBER_OF",
    "PHARM",
    "PHARM_DIST",
    "ConformerTensors",
    "EdgeType",
    "FloatTensor",
    "GraphTensors",
    "HeteroTensors",
    "LongTensor",
    "MoleculeTensors",
    "PharmacophoreData",
    "PharmacophoreTensors",
]
