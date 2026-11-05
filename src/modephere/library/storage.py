"""Library storage: safetensors per compound + Parquet tables (PLAN Q14).

Layout under the library root::

    compounds/<InChIKey>.safetensors   every array of one compound (numpy, no pickle)
    compounds.parquet                  registry: one row per compound
    conformers.parquet                 one row per conformer (metadata)
    failures.parquet                   failures of the latest build run
    build_manifest.json

A compound file holds its arrays under ``<part>/<name>`` keys and its
schemas (record, features, conformer metadata) as JSON in the safetensors
metadata. Reading rebuilds and *re-validates* the
:class:`~modephere.types.CompoundArtifacts`, so a corrupt or hand-edited file
fails loudly instead of feeding a model. Writes go to a temp file first and
are renamed into place, so a crash never leaves a half-written compound.

One file per compound keeps builds resumable and lets a dataset load lazily.
For millions of compounds, shard several compounds per file (PLAN Q4 open).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from pydantic import ValidationError
from safetensors import SafetensorError
from safetensors.numpy import load_file, save_file

from modephere.core import StorageError
from modephere.types import (
    COMPOUNDS_SCHEMA,
    CONFORMERS_SCHEMA,
    FAILURES_SCHEMA,
    CompoundArtifacts,
    ConformerEnsemble,
    FrameSchemaError,
    MoleculeGraph,
    Pharmacophore2D,
    Pharmacophore3D,
    PharmacophoreGraph,
    PharmacophorePoint,
    StandardizedCompound,
    check_frame,
    compounds_frame,
    conformers_frame,
    failures_frame,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from modephere.types import FailureRecord

logger = logging.getLogger(__name__)

FORMAT = "modephere-compound/1"
_GRAPH_FIELDS = ("x", "edge_index", "edge_attr", "global_feats")


# --------------------------------------------------------------------------- #
# CompoundArtifacts <-> flat arrays + metadata
# --------------------------------------------------------------------------- #


def _encode(art: CompoundArtifacts) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    for name in (*_GRAPH_FIELDS, "atomic_numbers"):
        arrays[f"molecule/{name}"] = getattr(art.molecule, name)
    for name in (*_GRAPH_FIELDS, "atom2feat"):
        arrays[f"pharmacophore/{name}"] = getattr(art.pharmacophore, name)
    arrays["fingerprint/pharm2d"] = art.pharm2d_fingerprint
    meta = {
        "format": FORMAT,
        "compound": art.compound.model_dump_json(),
        "pharmacophore_2d": art.pharmacophore_2d.model_dump_json(),
        "num_drug_atoms": str(art.pharmacophore.num_drug_atoms),
        "molecule_layout": art.molecule.LAYOUT.key,
        "pharmacophore_layout": art.pharmacophore.LAYOUT.key,
    }
    if art.conformers is not None and art.pharm3d_fingerprint is not None:
        arrays["conformers/positions"] = art.conformers.positions
        arrays["fingerprint/pharm3d"] = art.pharm3d_fingerprint
        meta["conformers"] = art.conformers.model_dump_json(exclude={"positions"})
        k = len(art.pharmacophore_2d.features)
        c = art.conformers.num_conformers
        # float64: points are computed in float64 and must reload bit-exact.
        pos = np.zeros((c, k, 3), dtype=np.float64)
        direction = np.zeros((c, k, 3), dtype=np.float64)
        has_dir = np.zeros((c, k), dtype=np.uint8)
        for ci, p3d in enumerate(art.pharmacophores_3d):
            for ki, point in enumerate(p3d.points):
                pos[ci, ki] = point.position
                if point.direction is not None:
                    direction[ci, ki] = point.direction
                    has_dir[ci, ki] = 1
        arrays["pharmacophore_3d/positions"] = pos
        arrays["pharmacophore_3d/directions"] = direction
        arrays["pharmacophore_3d/has_direction"] = has_dir
    # safetensors refuses read-only / non-contiguous buffers.
    return {k: np.ascontiguousarray(v).copy() for k, v in arrays.items()}, meta


def _decode(arrays: dict[str, np.ndarray], meta: dict[str, str]) -> CompoundArtifacts:
    if meta.get("format") != FORMAT:
        msg = f"unknown compound file format {meta.get('format')!r} (expected {FORMAT})"
        raise StorageError(msg)
    compound = StandardizedCompound.model_validate_json(meta["compound"])
    p2d = Pharmacophore2D.model_validate_json(meta["pharmacophore_2d"])
    molecule = MoleculeGraph(
        **{n: arrays[f"molecule/{n}"] for n in (*_GRAPH_FIELDS, "atomic_numbers")}
    )
    pharmacophore = PharmacophoreGraph(
        **{n: arrays[f"pharmacophore/{n}"] for n in (*_GRAPH_FIELDS, "atom2feat")},
        num_drug_atoms=int(meta["num_drug_atoms"]),
    )
    conformers = None
    p3d: tuple[Pharmacophore3D, ...] = ()
    if "conformers" in meta:
        conformers = ConformerEnsemble.model_validate(
            {
                **json.loads(meta["conformers"]),
                "positions": arrays["conformers/positions"],
            }
        )
        pos = arrays["pharmacophore_3d/positions"]
        direction = arrays["pharmacophore_3d/directions"]
        has_dir = arrays["pharmacophore_3d/has_direction"]
        p3d = tuple(
            Pharmacophore3D(
                inchikey=compound.inchikey,
                conformer_id=record.conformer_id,
                num_heavy_atoms=compound.num_heavy_atoms,
                points=tuple(
                    PharmacophorePoint(
                        family=f.family,
                        atom_ids=f.atom_ids,
                        position=tuple(pos[ci, ki].tolist()),
                        direction=tuple(direction[ci, ki].tolist())
                        if has_dir[ci, ki]
                        else None,
                    )
                    for ki, f in enumerate(p2d.features)
                ),
            )
            for ci, record in enumerate(conformers.conformers)
        )
    return CompoundArtifacts(
        compound=compound,
        molecule=molecule,
        pharmacophore=pharmacophore,
        pharmacophore_2d=p2d,
        pharm2d_fingerprint=arrays["fingerprint/pharm2d"],
        conformers=conformers,
        pharmacophores_3d=p3d,
        pharm3d_fingerprint=arrays.get("fingerprint/pharm3d"),
    )


def _read_metadata(path: Path) -> dict[str, str]:
    """The safetensors header metadata (an 8-byte length, then JSON)."""
    with path.open("rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    meta: dict[str, str] = header.get("__metadata__", {})
    return meta


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class LibraryStore:
    """Read and write a library rooted at ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.compounds_dir = root / "compounds"

    # ----- per-compound files -------------------------------------------- #

    def path_for(self, inchikey: str) -> Path:
        return self.compounds_dir / f"{inchikey}.safetensors"

    def exists(self, inchikey: str) -> bool:
        return self.path_for(inchikey).exists()

    def inchikeys(self) -> list[str]:
        """Every stored compound, sorted."""
        if not self.compounds_dir.exists():
            return []
        return sorted(p.stem for p in self.compounds_dir.glob("*.safetensors"))

    def __iter__(self) -> Iterator[CompoundArtifacts]:
        return (self.load(k) for k in self.inchikeys())

    def __len__(self) -> int:
        return len(self.inchikeys())

    def save(self, art: CompoundArtifacts) -> Path:
        """Write one compound atomically; returns its path."""
        arrays, meta = _encode(art)
        path = self.path_for(art.inchikey)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp.", suffix=".safetensors"
        )
        os.close(fd)
        try:
            save_file(arrays, tmp, metadata=meta)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def load(self, inchikey: str) -> CompoundArtifacts:
        """Read and re-validate one compound."""
        path = self.path_for(inchikey)
        if not path.exists():
            msg = f"compound {inchikey} is not in the library at {self.root}"
            raise StorageError(msg)
        try:
            arrays = load_file(path)
            return _decode(arrays, _read_metadata(path))
        except (SafetensorError, ValidationError, KeyError, ValueError) as e:
            msg = f"corrupt compound file {path}: {e}"
            raise StorageError(msg) from e

    # ----- tables --------------------------------------------------------- #

    def table_path(self, table: str) -> Path:
        return self.root / f"{table}.parquet"

    def read_table(self, table: str) -> pl.DataFrame:
        """Read a table, checked against its schema (empty if absent)."""
        schema = _SCHEMAS[table]
        path = self.table_path(table)
        if not path.exists():
            return pl.DataFrame(schema=schema)
        frame = pl.read_parquet(path)
        try:
            check_frame(frame, schema, table=table)
        except FrameSchemaError as e:
            raise StorageError(str(e)) from e
        return frame

    def _write_table(self, table: str, frame: pl.DataFrame) -> None:
        path = self.table_path(table)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp)
        tmp.replace(path)

    def upsert(self, arts: Iterable[CompoundArtifacts]) -> None:
        """Replace the registry and conformer rows of ``arts``' compounds."""
        arts = list(arts)
        if not arts:
            return
        keys = [a.inchikey for a in arts]
        compounds = pl.concat(
            [
                self.read_table("compounds").filter(~pl.col("inchikey").is_in(keys)),
                compounds_frame(a.compound for a in arts),
            ]
        ).sort("inchikey")
        self._write_table("compounds", compounds)
        conformers = pl.concat(
            [
                self.read_table("conformers").filter(~pl.col("inchikey").is_in(keys)),
                conformers_frame(
                    a.conformers for a in arts if a.conformers is not None
                ),
            ]
        ).sort("inchikey", "conformer_id")
        self._write_table("conformers", conformers)

    def write_failures(self, records: Iterable[FailureRecord]) -> None:
        """Replace the failures table with this run's failures."""
        self._write_table("failures", failures_frame(records))

    # ----- export --------------------------------------------------------- #

    def export_sdf(self, inchikey: str, path: Path) -> int:
        """Write a compound's conformers (microspecies, explicit H) to SDF."""
        from modephere.library.export import write_sdf

        return write_sdf(self.load(inchikey), path)


_SCHEMAS: dict[str, Any] = {
    "compounds": COMPOUNDS_SCHEMA,
    "conformers": CONFORMERS_SCHEMA,
    "failures": FAILURES_SCHEMA,
}

__all__ = ["FORMAT", "LibraryStore"]
