"""Catalog readers: query lists on disk -> validated :class:`CompoundQuery` s.

Formats, by extension:

* ``.tsv`` / ``.txt`` / ``.csv`` / ``.parquet`` — tabular, read with polars.
  Either explicit ``kind, value[, label]`` columns, or any of ``smiles`` /
  ``chembl_id`` / ``cid`` / ``name`` columns (dev_Pgen's ``cid, smiles,
  cmpd_name_cleaned`` catalog works unchanged). Per row the first non-empty of
  smiles > chembl_id > cid > name is used, so a row with a SMILES or a ChEMBL
  ID never needs the network.
* ``.json`` — a list of query objects (``{"kind": "cid", "cid": 2244}``) or
  dev_Pgen's flat ``{cid: smiles}`` mapping.

A bad row never aborts the load: it becomes a ``FailureRecord`` (stage
``resolve``, category ``invalid_query``) keyed ``row:<n>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import ValidationError

from modephere.core import DataError
from modephere.types import (
    COMPOUND_QUERY,
    BuildStage,
    ChemblQuery,
    CidQuery,
    FailureCategory,
    FailureRecord,
    IdentifierKind,
    NameQuery,
    SmilesQuery,
    parse_query,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from modephere.types import CompoundQuery

_NAME_COLUMNS = ("name", "cmpd_name_cleaned", "title")
_SEPARATORS = {".tsv": "\t", ".txt": "\t", ".csv": ","}


@dataclass
class CatalogLoad:
    """Queries read from a catalog, plus the rows that were rejected."""

    queries: list[CompoundQuery] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)

    def reject(self, row: int, reason: str) -> None:
        self.failures.append(
            FailureRecord(
                key=f"row:{row}",
                stage=BuildStage.RESOLVE,
                category=FailureCategory.INVALID_QUERY,
                reason=reason,
            )
        )


def _first_error(e: ValidationError) -> str:
    err = e.errors()[0]
    loc = ".".join(str(p) for p in err["loc"])
    return f"{loc}: {err['msg']}" if loc else err["msg"]


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_frame(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pl.read_parquet(path)
        return frame.select(pl.all().cast(pl.String))
    # All columns as strings: CIDs are validated by the query models, never
    # guessed by polars' type inference.
    return pl.read_csv(path, separator=_SEPARATORS[suffix], infer_schema=False)


def _row_query(row: dict[str, Any], *, explicit: bool) -> CompoundQuery:
    if explicit:
        return parse_query(
            row["kind"] or "", row.get("value"), label=_clean(row.get("label"))
        )
    smiles, cid = _clean(row.get("smiles")), _clean(row.get("cid"))
    chembl_id = _clean(row.get("chembl_id"))
    name = next((_clean(row.get(c)) for c in _NAME_COLUMNS if _clean(row.get(c))), None)
    label = _clean(row.get("label")) or name or cid
    if smiles:
        return SmilesQuery(smiles=smiles, label=label)
    if chembl_id:
        return ChemblQuery(chembl_id=chembl_id, label=label)
    if cid:
        return CidQuery(cid=cid, label=label)
    if name:
        return NameQuery(name=name, label=_clean(row.get("label")))
    msg = "row has no smiles, chembl_id, cid or name"
    raise ValueError(msg)


def _load_tabular(path: Path) -> CatalogLoad:
    frame = _read_frame(path)
    columns = set(frame.columns)
    explicit = {"kind", "value"} <= columns
    if not explicit and not columns & {"smiles", "chembl_id", "cid", *_NAME_COLUMNS}:
        msg = (
            f"{path}: expected 'kind'+'value' columns or any of "
            f"smiles/chembl_id/cid/{'/'.join(_NAME_COLUMNS)}; found {sorted(columns)}"
        )
        raise DataError(msg)
    load = CatalogLoad()
    for n, row in enumerate(frame.iter_rows(named=True), start=1):
        try:
            load.queries.append(_row_query(row, explicit=explicit))
        except ValidationError as e:
            load.reject(n, _first_error(e))
        except ValueError as e:
            load.reject(n, str(e))
    return load


def _load_json(path: Path) -> CatalogLoad:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    load = CatalogLoad()
    if isinstance(data, dict):  # dev_Pgen {cid: smiles}
        items: Iterable[Any] = (
            {"kind": "smiles", "smiles": smiles, "label": str(cid).strip()}
            for cid, smiles in data.items()
        )
    elif isinstance(data, list):
        items = data
    else:
        msg = f"{path}: expected a JSON list of queries or a {{cid: smiles}} object"
        raise DataError(msg)
    for n, item in enumerate(items, start=1):
        try:
            load.queries.append(COMPOUND_QUERY.validate_python(item))
        except ValidationError as e:
            load.reject(n, _first_error(e))
    return load


def load_catalog(path: Path) -> CatalogLoad:
    """Read a query catalog, dispatching on the file extension."""
    if not path.exists():
        msg = f"catalog not found: {path}"
        raise DataError(msg)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix in _SEPARATORS or suffix == ".parquet":
        return _load_tabular(path)
    msg = f"unsupported catalog format {suffix!r} for {path} (tsv/csv/txt/parquet/json)"
    raise DataError(msg)


def queries_from_values(
    *,
    smiles: Iterable[str] = (),
    cids: Iterable[str | int] = (),
    names: Iterable[str] = (),
    chembl_ids: Iterable[str] = (),
) -> CatalogLoad:
    """Queries given directly (e.g. on the command line), validated the same way."""
    load = CatalogLoad()
    pairs = [
        *((IdentifierKind.SMILES, v) for v in smiles),
        *((IdentifierKind.CID, v) for v in cids),
        *((IdentifierKind.NAME, v) for v in names),
        *((IdentifierKind.CHEMBL, v) for v in chembl_ids),
    ]
    for n, (kind, value) in enumerate(pairs, start=1):
        try:
            load.queries.append(parse_query(kind, value))
        except ValidationError as e:
            load.reject(n, f"{kind} {value!r}: {_first_error(e)}")
    return load


__all__ = ["CatalogLoad", "load_catalog", "queries_from_values"]
