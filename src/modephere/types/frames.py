"""Polars table schemas for the library's tabular outputs (PLAN Q14).

Tabular data is polars, never pandas. Each table has one ``pl.Schema``; rows
are produced from validated models only, so every value in a table has passed
a Pydantic validator first:

* ``compounds`` — one row per :class:`StandardizedCompound` (the registry,
  keyed by InChIKey);
* ``conformers`` — one row per conformer (metadata; coordinates live in the
  tensor shards);
* ``failures`` — one row per :class:`FailureRecord`.

Categorical columns are ``pl.Enum`` built from the matching ``StrEnum``, so the
table vocabulary cannot drift from the code.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

import polars as pl

from modephere.types.compound import StandardizedCompound
from modephere.types.conformer import ConformerEnsemble
from modephere.types.enums import (
    BuildStage,
    ChemblMatch,
    FailureCategory,
    ForceField,
    FragmentPolicy,
    IdentifierKind,
    ProtonationMethod,
    ResolutionSource,
)
from modephere.types.failures import FailureRecord


def _enum(values: type[StrEnum]) -> pl.DataType:
    return pl.Enum([m.value for m in values])


COMPOUNDS_SCHEMA = pl.Schema(
    {
        "inchikey": pl.String(),
        "query_key": pl.String(),
        "query_kind": _enum(IdentifierKind),
        "label": pl.String(),
        "source": _enum(ResolutionSource),
        "cid": pl.Int64(),
        "title": pl.String(),
        "is_ambiguous": pl.Boolean(),
        "chembl_id": pl.String(),
        "chembl_record_id": pl.String(),
        "chembl_match": _enum(ChemblMatch),
        "chembl_release": pl.String(),
        "registration_smiles": pl.String(),
        "working_smiles": pl.String(),
        "microspecies_smiles": pl.String(),
        "fragment_policy": _enum(FragmentPolicy),
        "protonation": _enum(ProtonationMethod),
        "fragments_removed": pl.Int64(),
        "num_heavy_atoms": pl.Int64(),
        "formal_charge": pl.Int64(),
        "toolkit_version": pl.String(),
    }
)

CONFORMERS_SCHEMA = pl.Schema(
    {
        "inchikey": pl.String(),
        "conformer_id": pl.Int64(),
        "stereoisomer": pl.Int64(),
        "energy": pl.Float64(),
        "relative_energy": pl.Float64(),
        "converged": pl.Boolean(),
        "force_field": _enum(ForceField),
    }
)

FAILURES_SCHEMA = pl.Schema(
    {
        "key": pl.String(),
        "stage": _enum(BuildStage),
        "category": _enum(FailureCategory),
        "reason": pl.String(),
        "occurred_at": pl.Datetime(time_unit="us", time_zone="UTC"),
    }
)


class FrameSchemaError(ValueError):
    """A DataFrame does not match the schema of the table it claims to be."""


def check_frame(frame: pl.DataFrame, schema: pl.Schema, *, table: str) -> None:
    """Raise :class:`FrameSchemaError` unless ``frame`` has exactly ``schema``.

    Checks column names, order and dtypes, e.g. on a Parquet file read back
    from disk before it is trusted.
    """
    if frame.schema == schema:
        return
    expected, actual = dict(schema), dict(frame.schema)
    problems = [f"missing {c!r}" for c in expected if c not in actual]
    problems += [f"unexpected {c!r}" for c in actual if c not in expected]
    problems += [
        f"{c!r} is {actual[c]}, expected {dtype}"
        for c, dtype in expected.items()
        if c in actual and actual[c] != dtype
    ]
    if not problems:
        problems = [f"column order {list(actual)} != {list(expected)}"]
    msg = f"{table} table does not match its schema: " + "; ".join(problems)
    raise FrameSchemaError(msg)


def compounds_frame(compounds: Iterable[StandardizedCompound]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for c in compounds:
        r = c.resolved
        rows.append(
            {
                "inchikey": c.inchikey,
                "query_key": r.query.key,
                "query_kind": r.query.kind.value,
                "label": r.query.label,
                "source": r.source.value,
                "cid": r.cid,
                "title": r.title,
                "is_ambiguous": r.is_ambiguous,
                "chembl_id": r.chembl_id,
                "chembl_record_id": r.chembl_record_id,
                "chembl_match": None
                if r.chembl_match is None
                else r.chembl_match.value,
                "chembl_release": r.chembl_release,
                "registration_smiles": c.registration_smiles,
                "working_smiles": c.working_smiles,
                "microspecies_smiles": c.microspecies_smiles,
                "fragment_policy": c.fragment_policy.value,
                "protonation": c.protonation.value,
                "fragments_removed": c.fragments_removed,
                "num_heavy_atoms": c.num_heavy_atoms,
                "formal_charge": c.formal_charge,
                "toolkit_version": c.toolkit_version,
            }
        )
    return pl.DataFrame(rows, schema=COMPOUNDS_SCHEMA, orient="row")


def conformers_frame(ensembles: Iterable[ConformerEnsemble]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for e in ensembles:
        energies = e.energies
        minimum = None if energies is None else float(energies.min())
        rows.extend(
            {
                "inchikey": e.inchikey,
                "conformer_id": c.conformer_id,
                "stereoisomer": c.stereoisomer,
                "energy": c.energy,
                "relative_energy": (
                    None if c.energy is None or minimum is None else c.energy - minimum
                ),
                "converged": c.converged,
                "force_field": e.force_field.value,
            }
            for c in e.conformers
        )
    return pl.DataFrame(rows, schema=CONFORMERS_SCHEMA, orient="row")


def failures_frame(records: Iterable[FailureRecord]) -> pl.DataFrame:
    rows = [
        {
            "key": f.key,
            "stage": f.stage.value,
            "category": f.category.value,
            "reason": f.reason,
            "occurred_at": f.occurred_at,
        }
        for f in records
    ]
    return pl.DataFrame(rows, schema=FAILURES_SCHEMA, orient="row")


__all__ = [
    "COMPOUNDS_SCHEMA",
    "CONFORMERS_SCHEMA",
    "FAILURES_SCHEMA",
    "FrameSchemaError",
    "check_frame",
    "compounds_frame",
    "conformers_frame",
    "failures_frame",
]
