"""A PyTorch dataset over a stored library.

    >>> from torch_geometric.loader import DataLoader
    >>> ds = LibraryDataset(store, kind="molecule", conformer="sample", targets=labels)
    >>> for batch in DataLoader(ds, batch_size=32, shuffle=True): ...

``conformer``:

* ``"none"`` — 2D graphs (no ``pos``);
* ``"lowest"`` — always the lowest-energy conformer;
* ``"sample"`` — a uniformly drawn conformer per access (augmentation: each
  epoch sees another conformation). Pass a ``generator`` for reproducible
  draws in a single process; with several DataLoader workers torch seeds each
  worker itself.

``targets`` is a polars frame with an ``inchikey`` column; its other columns
become ``y`` ``[1, T]`` (float32, NaN kept for missing multitask labels).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from modephere.core import ConfigurationError
from modephere.featurize.convert import hetero_data, molecule_data, pharmacophore_data
from modephere.library.storage import LibraryStore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch_geometric.data import Data, HeteroData

    from modephere.types import CompoundArtifacts

type GraphKind = Literal["molecule", "pharmacophore", "hetero"]
type ConformerMode = Literal["none", "lowest", "sample"]

_CONVERT = {
    "molecule": molecule_data,
    "pharmacophore": pharmacophore_data,
    "hetero": hetero_data,
}


class LibraryDataset(Dataset):
    """Graphs of a stored library, one item per compound."""

    def __init__(
        self,
        store: LibraryStore | Path,
        *,
        kind: GraphKind = "molecule",
        conformer: ConformerMode = "lowest",
        inchikeys: Sequence[str] | None = None,
        targets: pl.DataFrame | None = None,
        cache: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if kind not in _CONVERT:
            msg = f"kind must be one of {sorted(_CONVERT)}, got {kind!r}"
            raise ConfigurationError(msg)
        if conformer not in ("none", "lowest", "sample"):
            msg = f"conformer must be none/lowest/sample, got {conformer!r}"
            raise ConfigurationError(msg)
        self.store = store if isinstance(store, LibraryStore) else LibraryStore(store)
        self.kind = kind
        self.conformer = conformer
        self.generator = generator
        self._cache: dict[str, CompoundArtifacts] | None = {} if cache else None
        keys = list(inchikeys) if inchikeys is not None else self.store.inchikeys()
        self._targets: dict[str, np.ndarray] = {}
        self.target_names: list[str] = []
        if targets is not None:
            self.target_names = [c for c in targets.columns if c != "inchikey"]
            self._targets = self._index_targets(targets)
            # Only compounds that are both stored and labelled.
            keys = [k for k in keys if k in self._targets]
        self.inchikeys = keys

    @staticmethod
    def _index_targets(targets: pl.DataFrame) -> dict[str, np.ndarray]:
        if "inchikey" not in targets.columns:
            msg = "targets need an 'inchikey' column"
            raise ConfigurationError(msg)
        columns = [c for c in targets.columns if c != "inchikey"]
        if not columns:
            msg = "targets have no target column besides 'inchikey'"
            raise ConfigurationError(msg)
        if targets["inchikey"].is_duplicated().any():
            msg = "targets list an InChIKey more than once"
            raise ConfigurationError(msg)
        values = targets.select(pl.col(columns).cast(pl.Float32)).to_numpy()
        return {
            k: values[i : i + 1].astype(np.float32)
            for i, k in enumerate(targets["inchikey"].to_list())
        }

    def __len__(self) -> int:
        return len(self.inchikeys)

    def artifacts(self, index: int) -> CompoundArtifacts:
        key = self.inchikeys[index]
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        art = self.store.load(key)
        if self._cache is not None:
            self._cache[key] = art
        return art

    def _conformer_index(self, art: CompoundArtifacts) -> int | None:
        if self.conformer == "none":
            return None
        if art.conformers is None:
            msg = f"{art.inchikey} has no conformers; use conformer='none' for a 2D library"
            raise ConfigurationError(msg)
        if self.conformer == "lowest":
            return art.conformers.lowest
        n = art.conformers.num_conformers
        return int(torch.randint(n, (1,), generator=self.generator))

    def __getitem__(self, index: int) -> Data | HeteroData:
        art = self.artifacts(index)
        data = _CONVERT[self.kind](art, self._conformer_index(art))
        if self._targets:
            data.y = torch.from_numpy(self._targets[art.inchikey])
        return data


__all__ = ["ConformerMode", "GraphKind", "LibraryDataset"]
