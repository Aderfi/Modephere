"""ChEMBL staging tables -> knowledge-graph rows, and the load itself.

:func:`chembl_rows` is pure (Parquet in, row lists out), so the graph content
is testable without a database; :func:`load_chembl` writes those rows through
:class:`~modephere.kg.graph.KnowledgeGraph`. The model is documented in
:mod:`modephere.kg.schema`.

Only what the staging selection references reaches the graph: targets named
by a mechanism, a metabolism record or a bioactivity summary, their protein
components, human genes, the ATC classes above the drugs' codes. Raw
activities stay in ``activities.parquet``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

from modephere.chembl.extract import FLAG_COLUMNS, StagingManifest, read_table
from modephere.chembl.source import LICENSE
from modephere.kg.graph import RelSpec
from modephere.kg.provenance import Dataset, IngestRun, Stamp

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.kg.graph import KnowledgeGraph, Row

logger = logging.getLogger(__name__)

HUMAN_TAX_ID = 9606
RELEASES_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"

HAS_PARENT = RelSpec("HAS_PARENT", "Compound", "Compound")
HAS_MECHANISM = RelSpec("HAS_MECHANISM", "Compound", "Target", key="mec_id")
HAS_BIOACTIVITY = RelSpec("HAS_BIOACTIVITY", "Compound", "Target")
INDICATED_FOR = RelSpec("INDICATED_FOR", "Compound", "Disease")
HAS_WARNING = RelSpec("HAS_WARNING", "Compound", "DrugWarning")
IN_ATC = RelSpec("IN_ATC", "Compound", "ATC")
SUBCLASS_OF = RelSpec("SUBCLASS_OF", "ATC", "ATC")
METABOLIZED_TO = RelSpec("METABOLIZED_TO", "Compound", "Compound", key="met_id")
METABOLIZED_BY = RelSpec("METABOLIZED_BY", "Compound", "Target", key="met_id")
HAS_COMPONENT = RelSpec("HAS_COMPONENT", "Target", "Protein")
ENCODED_BY = RelSpec("ENCODED_BY", "Protein", "Gene")


@dataclass
class GraphRows:
    """Rows to merge, in load order (nodes before the edges that need them)."""

    nodes: dict[str, list[Row]] = field(default_factory=dict)
    rels: list[tuple[RelSpec, list[Row]]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.nodes.items()} | {
            spec.type: len(rows) for spec, rows in self.rels
        }


def _flag(name: str) -> pl.Expr:
    """ChEMBL 1 / 0 / -1 (unknown) flag -> true / false / null."""
    c = pl.col(name)
    return (
        pl.when(c == 1).then(True).when(c == 0).then(False).otherwise(None).alias(name)
    )


def _bool(name: str) -> pl.Expr:
    return (pl.col(name) == 1).alias(name)


def _rows(
    frame: pl.DataFrame, key: str | None, props: list[str], **ends: str
) -> list[Row]:
    """``[{"key", "start", "end", "props": {...}}]`` from columns of ``frame``."""
    cols: list[pl.Expr] = [pl.col(c).alias(alias) for alias, c in ends.items()]
    if key is not None:
        cols.append(pl.col(key).alias("key"))
    cols.append(pl.struct(props).alias("props") if props else pl.lit({}).alias("props"))
    return frame.select(cols).to_dicts()


def _lift(frame: pl.DataFrame, column: str, parents: pl.DataFrame) -> pl.DataFrame:
    """Move ``column`` from the curated record to its parent compound.

    ChEMBL curates many annotations on a salt record (imatinib *mesylate*
    carries imatinib's mechanisms); the graph attaches them to the parent,
    which is what a structure resolves to, and keeps the record as
    ``<column minus _chembl_id>_form`` (``form`` for ``chembl_id``).
    """
    form = (
        "form" if column == "chembl_id" else column.removesuffix("_chembl_id") + "_form"
    )
    return (
        frame.join(parents, left_on=column, right_on="chembl_id", how="left")
        .with_columns(
            pl.col(column).alias(form),
            pl.coalesce("parent_chembl_id", column).alias(column),
        )
        .drop("parent_chembl_id")
    )


def _compound_rows(
    staging: Path, mechanisms: pl.DataFrame
) -> tuple[pl.DataFrame, list[Row]]:
    compounds = read_table(staging, "compounds")
    synonyms = (
        read_table(staging, "synonyms")
        .group_by("chembl_id")
        .agg(synonyms=pl.col("synonym").unique().sort())
    )
    moa = (
        mechanisms.drop_nulls("mechanism_of_action")
        .group_by("chembl_id")
        .agg(mechanisms_of_action=pl.col("mechanism_of_action").unique().sort())
    )
    frame = (
        compounds.join(synonyms, on="chembl_id", how="left")
        .join(moa, on="chembl_id", how="left")
        .with_columns(
            *(_flag(f) for f in FLAG_COLUMNS),
            is_parent=pl.col("chembl_id") == pl.col("parent_chembl_id"),
            synonyms=pl.col("synonyms").fill_null([]),
            mechanisms_of_action=pl.col("mechanisms_of_action").fill_null([]),
            # one string for the full-text index (it cannot index lists)
            names_text=pl.concat_list("pref_name", pl.col("synonyms").fill_null([]))
            .list.drop_nulls()
            .list.unique(maintain_order=True)
            .list.join(" | "),
        )
        .rename({"canonical_smiles": "smiles"})
    )
    props = [
        c
        for c in frame.columns
        if c not in {"chembl_id", "parent_chembl_id", "standard_inchi"}
    ]
    return frame, _rows(frame, "chembl_id", props)


def chembl_rows(staging: Path) -> GraphRows:
    """Everything the ChEMBL layer writes, from one staging directory."""
    manifest = StagingManifest.read(staging)
    out = GraphRows()
    parents = read_table(staging, "compounds").select("chembl_id", "parent_chembl_id")
    known = set(parents["chembl_id"].to_list())

    def lifted(name: str, *columns: str) -> pl.DataFrame:
        frame = read_table(staging, name).filter(pl.col(columns[0]).is_in(known))
        for column in columns:
            frame = _lift(frame, column, parents)
        return frame

    all_mechanisms = lifted("mechanisms", "chembl_id")
    compounds, out.nodes["Compound"] = _compound_rows(staging, all_mechanisms)
    mechanisms = all_mechanisms.drop_nulls("target_chembl_id")
    metabolism = lifted("metabolism", "substrate_chembl_id", "metabolite_chembl_id")
    bioactivity = read_table(staging, "bioactivity").filter(
        pl.col("parent_chembl_id").is_in(known)
    )

    # Targets, proteins, human genes ---------------------------------------
    used_targets = set(mechanisms["target_chembl_id"].to_list())
    used_targets |= set(metabolism["enzyme_target_chembl_id"].drop_nulls().to_list())
    used_targets |= set(bioactivity["target_chembl_id"].to_list())
    targets = read_table(staging, "targets").filter(
        pl.col("chembl_id").is_in(used_targets)
    )
    out.nodes["Target"] = _rows(
        targets,
        "chembl_id",
        ["pref_name", "target_type", "organism", "tax_id", "species_group_flag"],
    )
    membership = read_table(staging, "target_components").filter(
        pl.col("target_chembl_id").is_in(used_targets)
    )
    proteins = read_table(staging, "components").filter(
        pl.col("component_type") == "PROTEIN",
        pl.col("accession").is_in(membership["accession"].implode()),
    )
    out.nodes["Protein"] = _rows(
        proteins.with_columns(pl.col("gene_symbols").fill_null([])),
        "accession",
        ["description", "organism", "tax_id", "gene_symbols"],
    )
    encoded = (
        proteins.filter(pl.col("tax_id") == HUMAN_TAX_ID)
        .explode("gene_symbols")
        .drop_nulls("gene_symbols")
        .select("accession", symbol="gene_symbols")
        .unique()
        .sort("accession", "symbol")
    )
    out.nodes["Gene"] = [
        {"key": s, "props": {}} for s in encoded["symbol"].unique().sort().to_list()
    ]

    # Disease, warnings, ATC -----------------------------------------------
    indications = lifted("indications", "chembl_id")
    diseases = indications.group_by("mesh_id").agg(
        mesh_heading=pl.col("mesh_heading").drop_nulls().first(),
        efo_ids=pl.col("efo_id").drop_nulls().unique().sort(),
        efo_terms=pl.col("efo_term").drop_nulls().unique().sort(),
    )
    out.nodes["Disease"] = _rows(
        diseases.sort("mesh_id"), "mesh_id", ["mesh_heading", "efo_ids", "efo_terms"]
    )
    warnings = lifted("warnings", "chembl_id")
    warning_props = [
        "form",
        "warning_type",
        "warning_class",
        "description",
        "country",
        "year",
        "efo_id",
        "efo_term",
        "class_efo_id",
    ]
    out.nodes["DrugWarning"] = _rows(warnings, "warning_id", warning_props)
    atc = (
        lifted("atc", "chembl_id")
        .group_by("chembl_id", "code")
        .agg(forms=pl.col("form").unique().sort())
        .sort("chembl_id", "code")
    )
    classes = read_table(staging, "atc_classes")
    out.nodes["ATC"] = _rows(
        _atc_closure(classes, set(atc["code"].to_list())), "code", ["level", "name"]
    )
    atc_nodes = {r["key"] for r in out.nodes["ATC"]}

    # Relationships ---------------------------------------------------------
    salts = compounds.filter(~pl.col("is_parent"))
    out.rels.append(
        (HAS_PARENT, _rows(salts, None, [], start="chembl_id", end="parent_chembl_id"))
    )
    out.rels.append(
        (
            HAS_MECHANISM,
            _rows(
                mechanisms.with_columns(
                    _bool("direct_interaction"),
                    _bool("molecular_mechanism"),
                    _bool("disease_efficacy"),
                    pl.col("refs").fill_null([]),
                    pl.col("ref_urls").fill_null([]),
                ),
                "mec_id",
                [
                    "mechanism_of_action",
                    "action_type",
                    "direct_interaction",
                    "molecular_mechanism",
                    "disease_efficacy",
                    "mechanism_comment",
                    "selectivity_comment",
                    "binding_site_comment",
                    "form",
                    "refs",
                    "ref_urls",
                ],
                start="chembl_id",
                end="target_chembl_id",
            ),
        )
    )
    activity_props = [
        c
        for c in bioactivity.columns
        if c not in {"parent_chembl_id", "target_chembl_id"}
    ]
    out.rels.append(
        (
            HAS_BIOACTIVITY,
            _rows(
                bioactivity.with_columns(
                    min_confidence=pl.lit(manifest.min_confidence, pl.Int8)
                ),
                None,
                [*activity_props, "min_confidence"],
                start="parent_chembl_id",
                end="target_chembl_id",
            ),
        )
    )
    indicated = indications.group_by("chembl_id", "mesh_id").agg(
        max_phase=pl.col("max_phase_for_ind").max(),
        efo_ids=pl.col("efo_id").drop_nulls().unique().sort(),
        forms=pl.col("form").unique().sort(),
    )
    out.rels.append(
        (
            INDICATED_FOR,
            _rows(
                indicated.sort("chembl_id", "mesh_id"),
                None,
                ["max_phase", "efo_ids", "forms"],
                start="chembl_id",
                end="mesh_id",
            ),
        )
    )
    out.rels.append(
        (HAS_WARNING, _rows(warnings, None, [], start="chembl_id", end="warning_id"))
    )
    out.rels.append(
        (IN_ATC, _rows(atc, None, ["forms"], start="chembl_id", end="code"))
    )
    tree = classes.filter(
        pl.col("code").is_in(atc_nodes), pl.col("parent_code").is_not_null()
    )
    out.rels.append(
        (SUBCLASS_OF, _rows(tree, None, [], start="code", end="parent_code"))
    )
    met_props = [
        "enzyme_name",
        "conversion",
        "organism",
        "tax_id",
        "pathway_id",
        "pathway_key",
        "comment",
        "drug_chembl_id",
        "substrate_form",
        "metabolite_form",
    ]
    out.rels.append(
        (
            METABOLIZED_TO,
            _rows(
                metabolism.filter(pl.col("metabolite_chembl_id").is_in(known)),
                "met_id",
                met_props,
                start="substrate_chembl_id",
                end="metabolite_chembl_id",
            ),
        )
    )
    out.rels.append(
        (
            METABOLIZED_BY,
            _rows(
                metabolism.drop_nulls("enzyme_target_chembl_id"),
                "met_id",
                met_props,
                start="substrate_chembl_id",
                end="enzyme_target_chembl_id",
            ),
        )
    )
    out.rels.append(
        (
            HAS_COMPONENT,
            _rows(
                membership.filter(
                    pl.col("accession").is_in(proteins["accession"].implode())
                ),
                None,
                ["homologue"],
                start="target_chembl_id",
                end="accession",
            ),
        )
    )
    out.rels.append(
        (ENCODED_BY, _rows(encoded, None, [], start="accession", end="symbol"))
    )
    return out


def _atc_closure(classes: pl.DataFrame, codes: set[str]) -> pl.DataFrame:
    """ATC classes of ``codes`` and all their ancestors."""
    parent: dict[str, str | None] = dict(
        classes.select("code", "parent_code").iter_rows()
    )
    keep: set[str] = set()
    for start in codes:
        code: str | None = start
        while code and code not in keep:
            keep.add(code)
            code = parent.get(code)
    return classes.filter(pl.col("code").is_in(keep)).sort("level", "code")


def chembl_dataset(manifest: StagingManifest) -> Dataset:
    """The ``:Dataset`` node of a staged release."""
    return Dataset(
        id=manifest.release,
        name="ChEMBL",
        version=str(manifest.release_number),
        uri=f"{RELEASES_URL}/{manifest.release}/",
        retrieved_at=datetime.fromtimestamp(manifest.sqlite_mtime, tz=UTC),
        sha256=manifest.sqlite_sha256,
        license=manifest.license or LICENSE,
    )


@dataclass(frozen=True)
class LoadSummary:
    run_id: str
    dataset: str
    nodes: dict[str, int]
    relationships: dict[str, int]


def load_chembl(
    graph: KnowledgeGraph, staging: Path, *, repo: Path | None = None
) -> LoadSummary:
    """Merge a staged ChEMBL release into ``graph`` (idempotent)."""
    manifest = StagingManifest.read(staging)
    dataset = chembl_dataset(manifest)
    run = IngestRun.start(
        {"staging": str(staging), "min_confidence": str(manifest.min_confidence)},
        repo=repo,
    )
    stamp = Stamp(sources=(dataset.id,), run_id=run.id)
    rows = chembl_rows(staging)
    graph.ensure_schema()
    graph.record_run(dataset, run)
    nodes = {
        label: graph.merge_nodes(label, r, stamp) for label, r in rows.nodes.items()
    }
    rels = {
        spec.type: graph.merge_relationships(spec, r, stamp) for spec, r in rows.rels
    }
    finished = run.model_copy(update={"finished_at": datetime.now(UTC)})
    graph.record_run(dataset, finished)
    return LoadSummary(run.id, dataset.id, nodes, rels)


__all__ = [
    "GraphRows",
    "LoadSummary",
    "chembl_dataset",
    "chembl_rows",
    "load_chembl",
]
