"""Bioactivity labels for :class:`~modephere.featurize.LibraryDataset`.

The dataset joins its ``targets`` frame on ``inchikey``. Give ``library`` and
the labels are keyed exactly: each library compound's registry ``chembl_id``
(its ChEMBL parent, set at resolution) picks its row, and the frame carries
the library's own InChIKey. Without it the key is the ChEMBL parent's
standard InChIKey, which agrees with Modephere's (canonical-tautomer) key for
about 88% of approved drugs; the rest get no label.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from modephere.chembl.extract import read_table

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

type LabelValue = Literal[
    "pchembl_median", "pchembl_mean", "pchembl_max", "pchembl_min"
]


def bioactivity_labels(
    staging: Path,
    targets: Sequence[str] | None = None,
    *,
    value: LabelValue = "pchembl_median",
    min_measurements: int = 1,
    wide: bool = True,
    library: Path | None = None,
) -> pl.DataFrame:
    """Per-compound pChEMBL labels from ``bioactivity.parquet``.

    ``targets`` restricts to these target ChEMBL IDs (all when ``None``). With
    ``wide`` the result has ``inchikey``, ``chembl_id`` and one Float64 column
    per target (null = not measured); otherwise one row per pair with ``n``.
    ``library`` (a library root) keys the rows by its compounds, see above.
    """
    pairs = read_table(staging, "bioactivity").filter(pl.col("n") >= min_measurements)
    if targets is not None:
        pairs = pairs.filter(pl.col("target_chembl_id").is_in(list(targets)))
    if library is None:
        keys = read_table(staging, "compounds").select("chembl_id", "inchikey")
    else:
        keys = (
            pl.read_parquet(
                library / "compounds.parquet", columns=["chembl_id", "inchikey"]
            )
            .drop_nulls("chembl_id")
            .unique()
        )
    long = (
        pairs.join(keys, left_on="parent_chembl_id", right_on="chembl_id", how="inner")
        .filter(pl.col("inchikey").is_not_null())
        .select(
            "inchikey",
            chembl_id="parent_chembl_id",
            target="target_chembl_id",
            value=pl.col(value),
            n="n",
        )
        .sort("chembl_id", "target")
    )
    if not wide:
        return long
    order = (
        list(targets)
        if targets is not None
        else long["target"].unique().sort().to_list()
    )
    out = long.pivot(
        on="target",
        index=["inchikey", "chembl_id"],
        values="value",
        aggregate_function="first",
    )
    present = [t for t in order if t in out.columns]
    return out.select("inchikey", "chembl_id", *present).sort("chembl_id")


__all__ = ["bioactivity_labels"]
