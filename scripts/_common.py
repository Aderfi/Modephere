"""Shared helpers for the inspection scripts in ``scripts/`` (not part of the package).

Every script runs from the project root with ``uv run python scripts/<name>.py``;
``scripts/`` is then on ``sys.path``, so ``from _common import ...`` works.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import TYPE_CHECKING, Any

import polars as pl
from rdkit import Chem, rdBase

from modephere.config import KGSettings, get_settings
from modephere.types import (
    ChemblQuery,
    CidQuery,
    NameQuery,
    SmilesQuery,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from modephere.chembl.lookup import ChemblLookup
    from modephere.kg.graph import KnowledgeGraph
    from modephere.library.ingest.pubchem import PubChemClient
    from modephere.types import CompoundQuery

_CHEMBL_ID = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)
_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_SMILES_HINT = re.compile(r"[=#()\[\]@+\\/]|\d|^[A-Z][a-z]?$|[cnos]{2}")

# Neo4j reports "property does not exist" notices on queries that probe
# optional properties; they are expected here.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

WIDTH = min(shutil.get_terminal_size((100, 20)).columns, 120)

# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def section(title: str, char: str = "=") -> None:
    """A full-width header."""
    print()
    print(char * WIDTH)
    print(f" {title}")
    print(char * WIDTH)


def subsection(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, WIDTH - len(title) - 5))


def note(text: str) -> None:
    print(f"  ({text})")


def kv(pairs: Mapping[str, Any], *, skip_none: bool = True) -> None:
    """Aligned ``key: value`` lines; long lists are shortened."""
    items = [(k, v) for k, v in pairs.items() if not (skip_none and v is None)]
    if not items:
        note("nothing")
        return
    pad = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"  {k:<{pad}} : {short(v)}")


def short(value: Any, limit: int = 200) -> str:
    if isinstance(value, (list, tuple)):
        shown = ", ".join(str(v) for v in value[:12])
        more = f" ... (+{len(value) - 12})" if len(value) > 12 else ""
        text = f"[{shown}{more}]"
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def show(
    df: pl.DataFrame,
    rows: int = 20,
    *,
    title: str | None = None,
    max_str: int = 36,
) -> None:
    """Print a frame with every column visible; lists joined, long strings cut."""
    if title:
        subsection(title)
    if df.height == 0:
        note("no rows")
        return
    df = df.head(rows).with_columns(
        *(
            pl.col(name).cast(pl.List(pl.String)).list.join(", ")
            for name, dtype in df.schema.items()
            if isinstance(dtype, pl.List)
        )
    )
    df = df.with_columns(
        *(
            pl.when(pl.col(name).str.len_chars() > max_str)
            .then(pl.col(name).str.slice(0, max_str - 1) + "…")
            .otherwise(pl.col(name))
            .alias(name)
            for name, dtype in df.schema.items()
            if dtype == pl.String
        )
    )
    with pl.Config(
        tbl_rows=rows,
        tbl_cols=-1,
        fmt_str_lengths=max_str,
        tbl_width_chars=WIDTH,
        tbl_formatting="UTF8_FULL_CONDENSED",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(df)
    if df.height > rows:
        note(f"showing {rows} of {df.height} rows; raise --limit to see more")


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def looks_like_smiles(text: str) -> bool:
    """Heuristic: parses as SMILES and carries SMILES syntax (not just a word)."""
    if " " in text or not _SMILES_HINT.search(text):
        return False
    with rdBase.BlockLogs():  # a failed parse is an answer here, not an error
        return Chem.MolFromSmiles(text) is not None


def make_query(text: str, kind: str = "auto") -> CompoundQuery:
    """``CHEMBL25`` -> ChEMBL ID, ``2244`` -> CID, SMILES -> SMILES, else a name."""
    text = text.strip()
    if kind == "auto":
        if _CHEMBL_ID.match(text):
            kind = "chembl_id"
        elif text.isdigit():
            kind = "cid"
        elif looks_like_smiles(text):
            kind = "smiles"
        else:
            kind = "name"
    match kind:
        case "chembl_id":
            return ChemblQuery(chembl_id=text.upper())
        case "cid":
            return CidQuery(cid=int(text))
        case "smiles":
            return SmilesQuery(smiles=text)
        case "name":
            return NameQuery(name=text)
    msg = f"unknown query kind {kind!r}"
    raise ValueError(msg)


def is_inchikey(text: str) -> bool:
    return bool(_INCHIKEY.match(text.strip()))


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #


def chembl_lookup() -> ChemblLookup | None:
    """The newest staged ChEMBL release, or None (with a hint) if none."""
    from modephere.chembl.extract import latest_staging
    from modephere.chembl.lookup import ChemblLookup

    staging = latest_staging(get_settings().paths.chembl)
    if staging is None:
        note("no staged ChEMBL release: run 'python -m modephere.chembl extract'")
        return None
    return ChemblLookup(staging)


def pubchem_client(*, offline: bool) -> PubChemClient:
    from modephere.library.ingest.pubchem import PubChemClient

    s = get_settings()
    return PubChemClient(s.pubchem, s.paths.cache / "pubchem", offline=offline)


def modephere_kg_settings() -> KGSettings:
    return get_settings().kg


def pgen_kg_settings() -> KGSettings:
    """dev_Pgen's graph (PharmGKB), same variables as its own CLI."""
    return KGSettings(
        uri=os.environ.get("PHARMAGEN_KG__URI", "bolt://localhost:7687"),
        user=os.environ.get("PHARMAGEN_KG__USER", "neo4j"),
        password=os.environ.get("PHARMAGEN_KG__PASSWORD", "pharmagen-dev"),
    )


DATABASES = {"modephere": modephere_kg_settings, "pgen": pgen_kg_settings}


class GraphUnavailable(Exception):
    """A Neo4j database could not be reached (stopped container, bad password)."""


def open_graph(name: str) -> KnowledgeGraph:
    """Connect to ``modephere`` or ``pgen``; raises GraphUnavailable with a hint."""
    from modephere.kg.graph import KnowledgeGraph

    settings = DATABASES[name]()
    try:
        graph = KnowledgeGraph.connect(settings)
        graph.run("RETURN 1 AS ok")
    except Exception as e:  # noqa: BLE001 -- driver raises several types
        container = "modephere-neo4j" if name == "modephere" else "pharmagen-neo4j"
        msg = (
            f"{name} graph at {settings.uri} is not reachable "
            f"({type(e).__name__}: {e}). Start it with: docker start {container}"
        )
        raise GraphUnavailable(msg) from e
    return graph


def iter_graphs(names: list[str]) -> Iterator[tuple[str, KnowledgeGraph | None]]:
    """Yield (name, graph or None); a None graph has already printed why."""
    for name in names:
        try:
            graph = open_graph(name)
        except GraphUnavailable as e:
            note(str(e))
            yield name, None
            continue
        with graph:
            yield name, graph


def output_dir(base: Path, slug: str) -> Path:
    out = base / re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)[:80]
    out.mkdir(parents=True, exist_ok=True)
    return out
