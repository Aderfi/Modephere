"""ChEMBL SQLite -> staging Parquet: the drug / clinical set and its activities.

Compound selection (PLAN F2):

1. **seeds** — every molecule with a ``max_phase`` (approved, clinical,
   preclinical with a phase of -1 "unknown"), plus any molecule named by a
   mechanism, indication, warning, ATC code or metabolism record;
2. their **parents** (``molecule_hierarchy``), and
3. every **form of those parents** (salts, hydrates), so a drug's activities
   measured on any salt are found.

Tables written to ``<staging>/<release>/`` (one Parquet file each):

========================  ====================================================
``compounds``             selected molecules: flags, structures, properties
``synonyms``              names of the selected molecules (INN, trade, ...)
``mechanisms``            drug -> target mechanisms of action (+ references)
``indications``           drug -> MeSH / EFO indications with phase
``warnings``              black-box warnings and withdrawals
``metabolism``            substrate -> metabolite, with the enzyme target
``atc``                   drug -> ATC level-5 code; ``atc_classes`` the tree
``targets``               every ChEMBL target; ``components`` its proteins and
                          genes; ``target_components`` the membership
``activities``            *raw* activities of the selected molecules
``bioactivity``           activities aggregated per (parent, target)
========================  ====================================================

Raw activities stay in Parquet; only ``bioactivity`` goes to the graph.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, Field

from modephere.chembl.source import LICENSE, LICENSE_URL, ChemblRelease, read_frame

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
_SEP = "\x1f"  # group_concat separator (unit separator: not in ChEMBL text)

# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

_SELECTION = (
    "CREATE TEMP TABLE seed (molregno INTEGER PRIMARY KEY)",
    "INSERT OR IGNORE INTO seed SELECT molregno FROM molecule_dictionary "
    "WHERE max_phase IS NOT NULL",
    *(
        f"INSERT OR IGNORE INTO seed SELECT molregno FROM {t} WHERE molregno IS NOT NULL"
        for t in (
            "drug_mechanism",
            "drug_indication",
            "drug_warning",
            "molecule_atc_classification",
        )
    ),
    *(
        "INSERT OR IGNORE INTO seed SELECT r.molregno FROM metabolism x "
        f"JOIN compound_records r ON r.record_id = x.{col} WHERE r.molregno IS NOT NULL"
        for col in ("drug_record_id", "substrate_record_id", "metabolite_record_id")
    ),
    "CREATE TEMP TABLE sel (molregno INTEGER PRIMARY KEY, parent_molregno INTEGER)",
    # seeds, with their parent (a molecule without a hierarchy row is its own)
    "INSERT OR IGNORE INTO sel SELECT s.molregno, coalesce(h.parent_molregno, s.molregno) "
    "FROM seed s LEFT JOIN molecule_hierarchy h ON h.molregno = s.molregno",
    # the parents themselves
    "INSERT OR IGNORE INTO sel SELECT DISTINCT p.parent_molregno, "
    "coalesce(h.parent_molregno, p.parent_molregno) FROM sel p "
    "LEFT JOIN molecule_hierarchy h ON h.molregno = p.parent_molregno",
    # every other form of a selected parent
    "INSERT OR IGNORE INTO sel SELECT h.molregno, h.parent_molregno "
    "FROM molecule_hierarchy h "
    "WHERE h.parent_molregno IN (SELECT DISTINCT parent_molregno FROM sel)",
)

# --------------------------------------------------------------------------- #
# Queries and their schemas
# --------------------------------------------------------------------------- #

#: Integer flags as ChEMBL stores them (1 yes, 0 no, -1 unknown where allowed).
FLAG_COLUMNS: tuple[str, ...] = (
    "oral",
    "parenteral",
    "topical",
    "black_box_warning",
    "withdrawn",
    "prodrug",
    "natural_product",
    "first_in_class",
    "therapeutic",
    "inorganic",
    "polymer",
    "orphan",
    "veterinary",
    "chemical_probe",
)

COMPOUNDS_SCHEMA: dict[str, pl.DataType] = {
    "chembl_id": pl.String(),
    "parent_chembl_id": pl.String(),
    "is_seed": pl.Boolean(),
    "pref_name": pl.String(),
    "max_phase": pl.Float64(),
    "molecule_type": pl.String(),
    "structure_type": pl.String(),
    "first_approval": pl.Int32(),
    "usan_year": pl.Int32(),
    "usan_stem": pl.String(),
    "chirality": pl.Int8(),
    "availability_type": pl.Int8(),
    **{f: pl.Int8() for f in FLAG_COLUMNS},
    "canonical_smiles": pl.String(),
    "standard_inchi": pl.String(),
    "inchikey": pl.String(),
    "full_molformula": pl.String(),
    "mw_freebase": pl.Float64(),
    "full_mwt": pl.Float64(),
    "alogp": pl.Float64(),
    "hba": pl.Int32(),
    "hbd": pl.Int32(),
    "psa": pl.Float64(),
    "rtb": pl.Int32(),
    "aromatic_rings": pl.Int32(),
    "heavy_atoms": pl.Int32(),
    "num_ro5_violations": pl.Int8(),
    "qed_weighted": pl.Float64(),
    "np_likeness_score": pl.Float64(),
}
_COMPOUNDS_SQL = """
SELECT md.chembl_id, pmd.chembl_id,
       sel.molregno IN (SELECT molregno FROM seed),
       md.pref_name, CAST(md.max_phase AS REAL), md.molecule_type, md.structure_type,
       md.first_approval, md.usan_year, md.usan_stem, md.chirality, md.availability_type,
       md.oral, md.parenteral, md.topical, md.black_box_warning, md.withdrawn_flag,
       md.prodrug, md.natural_product, md.first_in_class, md.therapeutic_flag,
       md.inorganic_flag, md.polymer_flag, md.orphan, md.veterinary, md.chemical_probe,
       cs.canonical_smiles, cs.standard_inchi, cs.standard_inchi_key,
       cp.full_molformula, CAST(cp.mw_freebase AS REAL), CAST(cp.full_mwt AS REAL),
       CAST(cp.alogp AS REAL), cp.hba, cp.hbd, CAST(cp.psa AS REAL), cp.rtb,
       cp.aromatic_rings, cp.heavy_atoms, cp.num_ro5_violations,
       CAST(cp.qed_weighted AS REAL), CAST(cp.np_likeness_score AS REAL)
FROM sel
JOIN molecule_dictionary md ON md.molregno = sel.molregno
JOIN molecule_dictionary pmd ON pmd.molregno = sel.parent_molregno
LEFT JOIN compound_structures cs ON cs.molregno = sel.molregno
LEFT JOIN compound_properties cp ON cp.molregno = sel.molregno
ORDER BY md.chembl_id
"""

SYNONYMS_SCHEMA: dict[str, pl.DataType] = {
    "chembl_id": pl.String(),
    "synonym": pl.String(),
    "syn_type": pl.String(),
}
_SYNONYMS_SQL = """
SELECT DISTINCT md.chembl_id, trim(ms.synonyms), ms.syn_type
FROM sel
JOIN molecule_dictionary md ON md.molregno = sel.molregno
JOIN molecule_synonyms ms ON ms.molregno = sel.molregno
WHERE ms.synonyms IS NOT NULL AND trim(ms.synonyms) != ''
ORDER BY 1, 3, 2
"""

MECHANISMS_SCHEMA: dict[str, pl.DataType] = {
    "mec_id": pl.Int64(),
    "chembl_id": pl.String(),
    "target_chembl_id": pl.String(),
    "mechanism_of_action": pl.String(),
    "action_type": pl.String(),
    "direct_interaction": pl.Int8(),
    "molecular_mechanism": pl.Int8(),
    "disease_efficacy": pl.Int8(),
    "mechanism_comment": pl.String(),
    "selectivity_comment": pl.String(),
    "binding_site_comment": pl.String(),
    "refs": pl.String(),
    "ref_urls": pl.String(),
}
_MECHANISMS_SQL = f"""
SELECT dm.mec_id, md.chembl_id, td.chembl_id, dm.mechanism_of_action, dm.action_type,
       dm.direct_interaction, dm.molecular_mechanism, dm.disease_efficacy,
       dm.mechanism_comment, dm.selectivity_comment, dm.binding_site_comment,
       (SELECT group_concat(r.ref_type || ':' || r.ref_id, '{_SEP}')
          FROM mechanism_refs r WHERE r.mec_id = dm.mec_id),
       (SELECT group_concat(r.ref_url, '{_SEP}')
          FROM mechanism_refs r WHERE r.mec_id = dm.mec_id AND r.ref_url IS NOT NULL)
FROM drug_mechanism dm
JOIN molecule_dictionary md ON md.molregno = dm.molregno
LEFT JOIN target_dictionary td ON td.tid = dm.tid
ORDER BY dm.mec_id
"""

INDICATIONS_SCHEMA: dict[str, pl.DataType] = {
    "drugind_id": pl.Int64(),
    "chembl_id": pl.String(),
    "max_phase_for_ind": pl.Float64(),
    "mesh_id": pl.String(),
    "mesh_heading": pl.String(),
    "efo_id": pl.String(),
    "efo_term": pl.String(),
}
_INDICATIONS_SQL = """
SELECT di.drugind_id, md.chembl_id, CAST(di.max_phase_for_ind AS REAL),
       di.mesh_id, di.mesh_heading, di.efo_id, di.efo_term
FROM drug_indication di
JOIN molecule_dictionary md ON md.molregno = di.molregno
ORDER BY di.drugind_id
"""

WARNINGS_SCHEMA: dict[str, pl.DataType] = {
    "warning_id": pl.Int64(),
    "chembl_id": pl.String(),
    "warning_type": pl.String(),
    "warning_class": pl.String(),
    "description": pl.String(),
    "country": pl.String(),
    "year": pl.Int32(),
    "efo_id": pl.String(),
    "efo_term": pl.String(),
    "class_efo_id": pl.String(),
}
_WARNINGS_SQL = """
SELECT dw.warning_id, md.chembl_id, dw.warning_type, dw.warning_class,
       dw.warning_description, dw.warning_country, dw.warning_year,
       dw.efo_id, dw.efo_term, dw.efo_id_for_warning_class
FROM drug_warning dw
JOIN molecule_dictionary md ON md.molregno = dw.molregno
ORDER BY dw.warning_id
"""

METABOLISM_SCHEMA: dict[str, pl.DataType] = {
    "met_id": pl.Int64(),
    "drug_chembl_id": pl.String(),
    "substrate_chembl_id": pl.String(),
    "metabolite_chembl_id": pl.String(),
    "enzyme_target_chembl_id": pl.String(),
    "enzyme_name": pl.String(),
    "conversion": pl.String(),
    "organism": pl.String(),
    "tax_id": pl.Int64(),
    "pathway_id": pl.Int64(),
    "pathway_key": pl.String(),
    "comment": pl.String(),
}
_METABOLISM_SQL = """
SELECT x.met_id, d.chembl_id, s.chembl_id, m.chembl_id, td.chembl_id,
       x.enzyme_name, x.met_conversion, x.organism, x.tax_id,
       x.pathway_id, x.pathway_key, x.met_comment
FROM metabolism x
LEFT JOIN compound_records rd ON rd.record_id = x.drug_record_id
LEFT JOIN molecule_dictionary d ON d.molregno = rd.molregno
LEFT JOIN compound_records rs ON rs.record_id = x.substrate_record_id
LEFT JOIN molecule_dictionary s ON s.molregno = rs.molregno
LEFT JOIN compound_records rm ON rm.record_id = x.metabolite_record_id
LEFT JOIN molecule_dictionary m ON m.molregno = rm.molregno
LEFT JOIN target_dictionary td ON td.tid = x.enzyme_tid
ORDER BY x.met_id
"""

ATC_SCHEMA: dict[str, pl.DataType] = {"chembl_id": pl.String(), "code": pl.String()}
_ATC_SQL = """
SELECT DISTINCT md.chembl_id, mac.level5
FROM molecule_atc_classification mac
JOIN molecule_dictionary md ON md.molregno = mac.molregno
ORDER BY 1, 2
"""

_ATC_RAW_SCHEMA: dict[str, pl.DataType] = {
    "who_name": pl.String(),
    **{f"level{i}": pl.String() for i in range(1, 6)},
    **{f"level{i}_description": pl.String() for i in range(1, 5)},
}
_ATC_CLASSES_SQL = """
SELECT who_name, level1, level2, level3, level4, level5,
       level1_description, level2_description, level3_description, level4_description
FROM atc_classification
"""
ATC_CLASSES_SCHEMA: dict[str, pl.DataType] = {
    "code": pl.String(),
    "level": pl.Int8(),
    "name": pl.String(),
    "parent_code": pl.String(),
}

TARGETS_SCHEMA: dict[str, pl.DataType] = {
    "chembl_id": pl.String(),
    "pref_name": pl.String(),
    "target_type": pl.String(),
    "organism": pl.String(),
    "tax_id": pl.Int64(),
    "species_group_flag": pl.Int8(),
}
_TARGETS_SQL = """
SELECT chembl_id, pref_name, target_type, organism, tax_id, species_group_flag
FROM target_dictionary ORDER BY chembl_id
"""

COMPONENTS_SCHEMA: dict[str, pl.DataType] = {
    "accession": pl.String(),
    "component_type": pl.String(),
    "description": pl.String(),
    "organism": pl.String(),
    "tax_id": pl.Int64(),
    "gene_symbols": pl.String(),
}
_COMPONENTS_SQL = f"""
SELECT cs.accession, cs.component_type, cs.description, cs.organism, cs.tax_id,
       (SELECT group_concat(trim(y.component_synonym), '{_SEP}')
          FROM component_synonyms y
         WHERE y.component_id = cs.component_id AND y.syn_type = 'GENE_SYMBOL')
FROM component_sequences cs
WHERE cs.accession IS NOT NULL
ORDER BY cs.accession
"""

TARGET_COMPONENTS_SCHEMA: dict[str, pl.DataType] = {
    "target_chembl_id": pl.String(),
    "accession": pl.String(),
    "homologue": pl.Int8(),
}
_TARGET_COMPONENTS_SQL = """
SELECT DISTINCT td.chembl_id, cs.accession, tc.homologue
FROM target_components tc
JOIN target_dictionary td ON td.tid = tc.tid
JOIN component_sequences cs ON cs.component_id = tc.component_id
WHERE cs.accession IS NOT NULL
ORDER BY 1, 2
"""

ACTIVITIES_SCHEMA: dict[str, pl.DataType] = {
    "activity_id": pl.Int64(),
    "chembl_id": pl.String(),
    "parent_chembl_id": pl.String(),
    "assay_chembl_id": pl.String(),
    "assay_type": pl.String(),
    "confidence_score": pl.Int8(),
    "relationship_type": pl.String(),
    "assay_organism": pl.String(),
    "bao_format": pl.String(),
    "target_chembl_id": pl.String(),
    "document_chembl_id": pl.String(),
    "year": pl.Int32(),
    "standard_type": pl.String(),
    "standard_relation": pl.String(),
    "standard_value": pl.Float64(),
    "standard_units": pl.String(),
    "pchembl_value": pl.Float64(),
    "activity_comment": pl.String(),
    "data_validity_comment": pl.String(),
    "potential_duplicate": pl.Int8(),
    "bao_endpoint": pl.String(),
    "action_type": pl.String(),
    "modality": pl.String(),
    "src_id": pl.Int32(),
}
_ACTIVITIES_SQL = """
SELECT a.activity_id, md.chembl_id, pmd.chembl_id,
       s.chembl_id, s.assay_type, s.confidence_score, s.relationship_type,
       s.assay_organism, s.bao_format, td.chembl_id, d.chembl_id, d.year,
       a.standard_type, a.standard_relation, CAST(a.standard_value AS REAL),
       a.standard_units, CAST(a.pchembl_value AS REAL), a.activity_comment,
       a.data_validity_comment, a.potential_duplicate, a.bao_endpoint,
       a.action_type, a.modality, a.src_id
FROM sel
JOIN activities a ON a.molregno = sel.molregno
JOIN molecule_dictionary md ON md.molregno = sel.molregno
JOIN molecule_dictionary pmd ON pmd.molregno = sel.parent_molregno
JOIN assays s ON s.assay_id = a.assay_id
LEFT JOIN target_dictionary td ON td.tid = s.tid
LEFT JOIN docs d ON d.doc_id = a.doc_id
"""

# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

#: ``data_validity_comment`` values that do not disqualify a measurement.
VALID_COMMENTS: frozenset[str] = frozenset({"Manually validated"})

BIOACTIVITY_SCHEMA: dict[str, pl.DataType] = {
    "parent_chembl_id": pl.String(),
    "target_chembl_id": pl.String(),
    "n": pl.UInt32(),
    "pchembl_median": pl.Float64(),
    "pchembl_mean": pl.Float64(),
    "pchembl_std": pl.Float64(),
    "pchembl_min": pl.Float64(),
    "pchembl_max": pl.Float64(),
    "n_assays": pl.UInt32(),
    "n_documents": pl.UInt32(),
    "max_confidence": pl.Int8(),
    "assay_types": pl.List(pl.String()),
    "standard_types": pl.List(pl.String()),
    "first_year": pl.Int32(),
    "last_year": pl.Int32(),
}


def eligible_activities(raw: pl.DataFrame, min_confidence: int) -> pl.DataFrame:
    """Activities that count towards a compound-target summary.

    A pChEMBL value with an exact (``=``) relation, a target, an assay target
    confidence ``>= min_confidence``, not flagged as a potential duplicate and
    without a data-validity issue (``Manually validated`` is not an issue).
    """
    return raw.filter(
        pl.col("pchembl_value").is_not_null(),
        pl.col("standard_relation") == "=",
        pl.col("target_chembl_id").is_not_null(),
        pl.col("confidence_score") >= min_confidence,
        pl.col("potential_duplicate").fill_null(0) == 0,
        pl.col("data_validity_comment").is_null()
        | pl.col("data_validity_comment").is_in(sorted(VALID_COMMENTS)),
    )


def aggregate_bioactivity(raw: pl.DataFrame, min_confidence: int) -> pl.DataFrame:
    """One row per (parent compound, target) over :func:`eligible_activities`.

    Salt forms count towards their parent: the summary is keyed by
    ``parent_chembl_id``.
    """
    pchembl = pl.col("pchembl_value")
    out = (
        eligible_activities(raw, min_confidence)
        .group_by("parent_chembl_id", "target_chembl_id")
        .agg(
            n=pl.len(),
            pchembl_median=pchembl.median(),
            pchembl_mean=pchembl.mean(),
            pchembl_std=pchembl.std(),
            pchembl_min=pchembl.min(),
            pchembl_max=pchembl.max(),
            n_assays=pl.col("assay_chembl_id").n_unique(),
            n_documents=pl.col("document_chembl_id").drop_nulls().n_unique(),
            max_confidence=pl.col("confidence_score").max(),
            assay_types=pl.col("assay_type").drop_nulls().unique().sort(),
            standard_types=pl.col("standard_type").drop_nulls().unique().sort(),
            first_year=pl.col("year").min(),
            last_year=pl.col("year").max(),
        )
        .sort("parent_chembl_id", "target_chembl_id")
    )
    return out.cast(BIOACTIVITY_SCHEMA)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


class StagingManifest(BaseModel):
    """What a staging directory holds and where it came from."""

    release: str
    release_number: int
    release_date: str | None
    sqlite_path: str
    sqlite_size: int
    sqlite_mtime: float
    sqlite_sha256: str | None = None
    license: str = LICENSE
    license_url: str = LICENSE_URL
    components: dict[str, str] = Field(
        default_factory=dict, description="Ontology / toolkit versions of the release."
    )
    min_confidence: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    modephere_version: str = Field(default_factory=lambda: version("modephere"))
    rows: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def read(cls, staging: Path) -> StagingManifest:
        return cls.model_validate_json((staging / MANIFEST).read_text("utf-8"))


def _split(column: str) -> pl.Expr:
    return pl.col(column).str.split(_SEP)


def _atc_classes(conn: sqlite3.Connection) -> pl.DataFrame:
    raw = read_frame(conn, _ATC_CLASSES_SQL, _ATC_RAW_SCHEMA)
    parts = [
        raw.select(
            code=pl.col(f"level{i}"),
            level=pl.lit(i, pl.Int8),
            name=pl.col(f"level{i}_description"),
            parent_code=pl.col(f"level{i - 1}") if i > 1 else pl.lit(None, pl.String),
        )
        for i in range(1, 5)
    ]
    parts.append(
        raw.select(
            code=pl.col("level5"),
            level=pl.lit(5, pl.Int8),
            name=pl.col("who_name"),
            parent_code=pl.col("level4"),
        )
    )
    return (
        pl.concat(parts)
        .filter(pl.col("code").is_not_null())
        .unique("code", keep="first", maintain_order=True)
        .sort("level", "code")
        .cast(ATC_CLASSES_SCHEMA)  # type: ignore[arg-type]
    )


def _write(frame: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp, compression="zstd", statistics=True)
    tmp.replace(path)


def extract(
    release: ChemblRelease,
    staging_root: Path,
    *,
    min_confidence: int = 7,
    checksum: bool = True,
) -> Path:
    """Write the staging tables of ``release`` under ``staging_root/<release.id>``.

    Returns the release's staging directory. Files are replaced atomically; the
    manifest is written last, so a directory with a manifest is complete.
    """
    out = staging_root / release.id
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).unlink(missing_ok=True)
    rows: dict[str, int] = {}

    def save(name: str, frame: pl.DataFrame) -> pl.DataFrame:
        _write(frame, out / f"{name}.parquet")
        rows[name] = frame.height
        logger.info("%s: %d rows", name, frame.height)
        return frame

    with release.connect() as conn:
        for statement in _SELECTION:
            conn.execute(statement)
        simple: dict[str, tuple[str, dict[str, pl.DataType]]] = {
            "compounds": (_COMPOUNDS_SQL, COMPOUNDS_SCHEMA),
            "synonyms": (_SYNONYMS_SQL, SYNONYMS_SCHEMA),
            "indications": (_INDICATIONS_SQL, INDICATIONS_SCHEMA),
            "warnings": (_WARNINGS_SQL, WARNINGS_SCHEMA),
            "metabolism": (_METABOLISM_SQL, METABOLISM_SCHEMA),
            "atc": (_ATC_SQL, ATC_SCHEMA),
            "targets": (_TARGETS_SQL, TARGETS_SCHEMA),
            "target_components": (_TARGET_COMPONENTS_SQL, TARGET_COMPONENTS_SCHEMA),
        }
        for name, (sql, schema) in simple.items():
            save(name, read_frame(conn, sql, schema))
        mechanisms = read_frame(conn, _MECHANISMS_SQL, MECHANISMS_SCHEMA)
        save("mechanisms", mechanisms.with_columns(_split("refs"), _split("ref_urls")))
        components = read_frame(conn, _COMPONENTS_SQL, COMPONENTS_SCHEMA)
        save("components", components.with_columns(_split("gene_symbols")))
        save("atc_classes", _atc_classes(conn))
        logger.info("Reading activities (a few minutes on a full release)...")
        raw = read_frame(conn, _ACTIVITIES_SQL, ACTIVITIES_SCHEMA).sort("activity_id")
    save("activities", raw)
    save("bioactivity", aggregate_bioactivity(raw, min_confidence))

    stat = release.path.stat()
    manifest = StagingManifest(
        release=release.id,
        release_number=release.number,
        release_date=None
        if release.release_date is None
        else str(release.release_date),
        sqlite_path=str(release.path),
        sqlite_size=stat.st_size,
        sqlite_mtime=stat.st_mtime,
        sqlite_sha256=release.sha256() if checksum else None,
        components=dict(release.components),
        min_confidence=min_confidence,
        rows=rows,
    )
    _write_json(out / MANIFEST, manifest.model_dump_json(indent=2))
    return out


def _write_json(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    Path(tmp).replace(path)


def latest_staging(root: Path) -> Path | None:
    """The newest complete (manifest-bearing) release under ``root``, if any."""
    done = sorted(
        (p.parent for p in root.glob(f"chembl_*/{MANIFEST}")),
        key=lambda p: int(p.name.removeprefix("chembl_")),
    )
    return done[-1] if done else None


def read_table(staging: Path, name: str) -> pl.DataFrame:
    """One staging table (``compounds``, ``bioactivity``, ...)."""
    return pl.read_parquet(staging / f"{name}.parquet")


__all__ = [
    "ACTIVITIES_SCHEMA",
    "BIOACTIVITY_SCHEMA",
    "COMPOUNDS_SCHEMA",
    "FLAG_COLUMNS",
    "MANIFEST",
    "VALID_COMMENTS",
    "StagingManifest",
    "aggregate_bioactivity",
    "eligible_activities",
    "extract",
    "latest_staging",
    "read_table",
]
