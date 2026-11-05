"""Stored compound -> PyG ``Data`` / ``HeteroData`` (PLAN C2.4).

``conformer`` picks the coordinates: ``None`` gives a 2D graph (no ``pos``),
an int gives conformer ``i`` of the ensemble (``0`` is the lowest-energy one).
Atom ``pos`` and pharmacophore ``pos`` always come from the *same* conformer.

Geometric models (SchNet, DimeNet++, ViSNet) read ``z`` and ``pos`` and build
their own radius graphs, so no distance edges are precomputed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from modephere.core import StorageError
from modephere.types.tensors import HeteroTensors, MoleculeTensors, PharmacophoreTensors

if TYPE_CHECKING:
    from torch_geometric.data import Data, HeteroData

    from modephere.types import CompoundArtifacts


def _check_conformer(art: CompoundArtifacts, conformer: int) -> None:
    if art.conformers is None:
        msg = f"{art.inchikey} was built 2D-only: it has no conformer {conformer}"
        raise StorageError(msg)
    if not 0 <= conformer < art.conformers.num_conformers:
        msg = (
            f"{art.inchikey} has {art.conformers.num_conformers} conformers, "
            f"no conformer {conformer}"
        )
        raise StorageError(msg)


def atom_positions(art: CompoundArtifacts, conformer: int) -> torch.Tensor:
    """``[N, 3]`` heavy-atom coordinates of one conformer (float32, Å)."""
    _check_conformer(art, conformer)
    assert art.conformers is not None
    heavy = art.conformers.positions[conformer, : art.conformers.num_heavy_atoms]
    return torch.from_numpy(np.array(heavy, dtype=np.float32))


def feature_positions(art: CompoundArtifacts, conformer: int) -> torch.Tensor:
    """``[K, 3]`` pharmacophore point coordinates of one conformer (float32, Å)."""
    _check_conformer(art, conformer)
    points = art.pharmacophores_3d[conformer].points
    xyz = np.asarray([p.position for p in points], dtype=np.float32).reshape(-1, 3)
    return torch.from_numpy(xyz)


def molecule_tensors(
    art: CompoundArtifacts, conformer: int | None = None
) -> MoleculeTensors:
    pos = None if conformer is None else atom_positions(art, conformer)
    return MoleculeTensors.from_graph(art.molecule, pos=pos)


def pharmacophore_tensors(
    art: CompoundArtifacts, conformer: int | None = None
) -> PharmacophoreTensors:
    pos = None if conformer is None else feature_positions(art, conformer)
    return PharmacophoreTensors.from_graph(art.pharmacophore, pos=pos)


def molecule_data(art: CompoundArtifacts, conformer: int | None = None) -> Data:
    """Atom graph (``x``, ``edge_index``, ``edge_attr``, ``global_feats``, ``z``[, ``pos``])."""
    return molecule_tensors(art, conformer).to_data(inchikey=art.inchikey)


def pharmacophore_data(art: CompoundArtifacts, conformer: int | None = None) -> Data:
    """Feature graph; batches with ``atom2feat`` offset correctly."""
    return pharmacophore_tensors(art, conformer).to_data(inchikey=art.inchikey)


def hetero_data(art: CompoundArtifacts, conformer: int | None = None) -> HeteroData:
    """``atom`` + ``pharm`` nodes; ``bond``, ``member_of`` and ``dist`` edges."""
    tensors = HeteroTensors(
        molecule=molecule_tensors(art, conformer),
        pharmacophore=pharmacophore_tensors(art, conformer),
    )
    return tensors.to_hetero_data(inchikey=art.inchikey)


__all__ = [
    "atom_positions",
    "feature_positions",
    "hetero_data",
    "molecule_data",
    "molecule_tensors",
    "pharmacophore_data",
    "pharmacophore_tensors",
]
