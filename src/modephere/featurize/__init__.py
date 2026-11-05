"""PyTorch Geometric featurization (needs the ``ml`` extra).

The only Modephere package that imports torch: everything upstream stays
numpy, so the chemistry layers install and test without CUDA (PLAN C3).
"""

from modephere.featurize.convert import (
    atom_positions,
    feature_positions,
    hetero_data,
    molecule_data,
    molecule_tensors,
    pharmacophore_data,
    pharmacophore_tensors,
)
from modephere.featurize.dataset import ConformerMode, GraphKind, LibraryDataset

__all__ = [
    "ConformerMode",
    "GraphKind",
    "LibraryDataset",
    "atom_positions",
    "feature_positions",
    "hetero_data",
    "molecule_data",
    "molecule_tensors",
    "pharmacophore_data",
    "pharmacophore_tensors",
]
