"""Offline name / ChEMBL ID / InChIKey lookup over the staging tables.

Names match case- and whitespace-insensitively against preferred names and
every synonym (INN, USAN, trade names, research codes, ...). A hit is the
molecule the name belongs to, with its parent: a salt name such as
"imatinib mesylate" finds the mesylate record, whose parent is imatinib.
Hits are ranked: preferred-name matches first, then by ``max_phase``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from typing import TYPE_CHECKING

import polars as pl

from modephere.chembl.extract import StagingManifest, read_table

if TYPE_CHECKING:
    from pathlib import Path

_COLUMNS = (
    "chembl_id",
    "parent_chembl_id",
    "pref_name",
    "canonical_smiles",
    "inchikey",
    "max_phase",
)


@dataclass(frozen=True, slots=True)
class ChemblHit:
    """One ChEMBL molecule matched by a lookup."""

    chembl_id: str
    parent_chembl_id: str
    pref_name: str | None
    smiles: str | None
    inchikey: str | None
    max_phase: float | None
    matched: str  # "pref_name", a synonym type (INN, TRADE_NAME, ...), "chembl_id", "inchikey"

    @property
    def is_parent(self) -> bool:
        return self.chembl_id == self.parent_chembl_id


def normalise_name(name: str) -> str:
    """Lookup key of a name: case-folded, whitespace collapsed."""
    return " ".join(name.casefold().split())


class ChemblLookup:
    """In-memory index of one staging directory (~20k molecules, ~150k names)."""

    def __init__(self, staging: Path) -> None:
        manifest = StagingManifest.read(staging)
        self.staging = staging
        self.release: str = manifest.release
        #: The release date (midnight UTC): the "retrieved at" of its records.
        self.released_at: datetime = (
            datetime.fromisoformat(manifest.release_date).replace(tzinfo=UTC)
            if manifest.release_date
            else manifest.created_at
        )
        compounds = read_table(staging, "compounds").select(_COLUMNS)
        self._rows: dict[str, dict[str, object]] = {
            r["chembl_id"]: r for r in compounds.iter_rows(named=True)
        }
        names = pl.concat(
            [
                compounds.select(
                    "chembl_id", name="pref_name", kind=pl.lit("pref_name")
                ).drop_nulls("name"),
                read_table(staging, "synonyms").select(
                    "chembl_id", name="synonym", kind="syn_type"
                ),
            ]
        )
        self._names: dict[str, list[tuple[str, str]]] = {}
        for chembl_id, name, kind in names.iter_rows():
            entry = (chembl_id, kind)
            bucket = self._names.setdefault(normalise_name(name), [])
            if entry not in bucket:
                bucket.append(entry)
        self._inchikeys: dict[str, list[str]] = {}
        for chembl_id, key in compounds.select("chembl_id", "inchikey").iter_rows():
            if key:
                self._inchikeys.setdefault(key, []).append(chembl_id)

    def __len__(self) -> int:
        return len(self._rows)

    def _hit(self, chembl_id: str, matched: str) -> ChemblHit:
        r = self._rows[chembl_id]
        return ChemblHit(
            chembl_id=chembl_id,
            parent_chembl_id=str(r["parent_chembl_id"]),
            pref_name=r["pref_name"],  # type: ignore[arg-type]
            smiles=r["canonical_smiles"],  # type: ignore[arg-type]
            inchikey=r["inchikey"],  # type: ignore[arg-type]
            max_phase=r["max_phase"],  # type: ignore[arg-type]
            matched=matched,
        )

    def by_chembl_id(self, chembl_id: str) -> ChemblHit | None:
        key = chembl_id.strip().upper()
        return self._hit(key, "chembl_id") if key in self._rows else None

    def by_inchikey(self, inchikey: str) -> list[ChemblHit]:
        return [
            self._hit(c, "inchikey")
            for c in self._inchikeys.get(inchikey.strip().upper(), [])
        ]

    @cached_property
    def _atc_class_names(self) -> frozenset[str]:
        classes = read_table(self.staging, "atc_classes").filter(pl.col("level") < 5)
        return frozenset(normalise_name(n) for n in classes["name"].drop_nulls())

    def is_atc_class(self, name: str) -> bool:
        """Whether ``name`` is an ATC class (levels 1-4), e.g. "corticosteroids"."""
        return normalise_name(name) in self._atc_class_names

    def parent_of(self, hit: ChemblHit) -> ChemblHit:
        """The parent molecule of ``hit`` (``hit`` itself for a parent)."""
        return hit if hit.is_parent else self._hit(hit.parent_chembl_id, hit.matched)

    def by_name(self, name: str) -> list[ChemblHit]:
        """Every molecule ``name`` belongs to, best first."""
        found: dict[str, str] = {}
        for chembl_id, kind in self._names.get(normalise_name(name), []):
            if found.get(chembl_id) != "pref_name":
                found[chembl_id] = kind
        hits = [self._hit(c, kind) for c, kind in found.items()]
        return sorted(
            hits,
            key=lambda h: (
                h.matched != "pref_name",
                -(h.max_phase if h.max_phase is not None else -9.0),
                not h.is_parent,
                h.chembl_id,
            ),
        )


__all__ = ["ChemblHit", "ChemblLookup", "normalise_name"]
