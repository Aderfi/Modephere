"""Read-only access to a local ChEMBL SQLite release.

The release file is never modified: it is opened with ``mode=ro`` (temporary
tables live in SQLite's separate temp database). Every query is read with an
explicit polars schema, because ChEMBL's ``NUMERIC`` columns mix SQLite
integer and real storage classes and inference from the first rows would be
wrong for later ones.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Self

import polars as pl

from modephere.core import DataError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from polars._typing import PolarsDataType

logger = logging.getLogger(__name__)

_RELEASE = re.compile(r"^ChEMBL_(\d+)$")
_CHUNK = 1 << 22
#: Rows fetched per round trip; bounds the Python objects alive at once.
FETCH_ROWS = 200_000

#: ChEMBL's licence; kept with every derived dataset (attribution required).
LICENSE = "CC BY-SA 3.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"


@dataclass(frozen=True)
class ChemblRelease:
    """One ChEMBL SQLite release, identified by its ``version`` table."""

    path: Path
    number: int
    release_date: date | None
    components: tuple[tuple[str, str], ...]

    @property
    def id(self) -> str:
        """Dataset id, e.g. ``chembl_37``."""
        return f"chembl_{self.number}"

    @classmethod
    def open(cls, path: Path) -> Self:
        """Check the file is a ChEMBL release and read its version table."""
        if not path.is_file():
            msg = f"ChEMBL SQLite file not found: {path}"
            raise DataError(msg)
        try:
            with closing(_connect(path)) as conn:
                rows = conn.execute(
                    "SELECT name, creation_date, comments FROM version"
                ).fetchall()
        except sqlite3.DatabaseError as e:
            msg = f"{path} is not a readable ChEMBL SQLite release: {e}"
            raise DataError(msg) from e
        number, released = None, None
        components: list[tuple[str, str]] = []
        for name, created, comment in rows:
            if m := _RELEASE.match(str(name)):
                number = int(m.group(1))
                released = _parse_date(created)
            else:
                components.append((str(name), str(comment or "")))
        if number is None:
            msg = f"{path}: no 'ChEMBL_<N>' row in the version table"
            raise DataError(msg)
        return cls(path, number, released, tuple(sorted(components)))

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """A read-only connection, closed on exit.

        (``sqlite3.Connection``'s own context manager only ends the
        transaction; it never closes.)
        """
        with closing(_connect(self.path)) as conn:
            yield conn

    def sha256(self) -> str:
        """Checksum of the release file (a full read: ~1-2 min for 28 GB)."""
        digest = hashlib.sha256()
        with self.path.open("rb") as f:
            while chunk := f.read(_CHUNK):
                digest.update(chunk)
        return digest.hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def _parse_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def read_frame(
    conn: sqlite3.Connection,
    sql: str,
    schema: Mapping[str, PolarsDataType],
    params: Sequence[object] = (),
) -> pl.DataFrame:
    """Run ``sql`` and return its rows with exactly ``schema``.

    Columns are matched by position (the schema names them), so the query
    needs no aliases; rows are fetched in chunks of :data:`FETCH_ROWS`.
    """
    cursor = conn.execute(sql, tuple(params))
    if len(cursor.description) != len(schema):
        msg = (
            f"query returns {len(cursor.description)} columns, "
            f"schema has {len(schema)}: {list(schema)}"
        )
        raise DataError(msg)
    parts = []
    while rows := cursor.fetchmany(FETCH_ROWS):
        parts.append(pl.DataFrame(rows, schema=dict(schema), orient="row"))
    if not parts:
        return pl.DataFrame(schema=dict(schema))
    return pl.concat(parts, how="vertical", rechunk=True)


__all__ = ["FETCH_ROWS", "LICENSE", "LICENSE_URL", "ChemblRelease", "read_frame"]
