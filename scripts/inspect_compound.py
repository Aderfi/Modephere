"""Look up one compound in every source Modephere knows about.

Usage (from the project root)::

    uv run python scripts/inspect_compound.py warfarin
    uv run python scripts/inspect_compound.py CHEMBL25
    uv run python scripts/inspect_compound.py 2244                 # PubChem CID
    uv run python scripts/inspect_compound.py "CC(=O)Oc1ccccc1C(=O)O"
    uv run python scripts/inspect_compound.py gleevec --only chembl pubchem
    uv run python scripts/inspect_compound.py clopidogrel --offline --limit 30

Sections (``--only`` picks some):

``resolve``
    Modephere's own view: the resolver (ChEMBL first, PubChem for the rest),
    then standardization (registration / working / pH 7.4 forms).
``chembl``
    The staged ChEMBL release (``data/chembl/chembl_<N>``): record and parent,
    salt forms, synonyms, ATC lineage, mechanisms, indications, warnings,
    metabolism and aggregated bioactivity per target (with gene symbols).
``pubchem``
    PubChem properties of the CID(s), through Modephere's raw cache (network
    only when not cached and not ``--offline``).
``pharmgkb``
    dev_Pgen's Neo4j (PharmGKB annotations): the matching ``Drug`` nodes,
    annotation counts by source / phenotype / significance, the genes and
    variants involved, example annotations, and the genes shared with the
    ChEMBL targets of the compound.

The identity cross-check at the end compares the InChIKeys each source gives.
For the compound's node in Modephere's own graph use ``inspect_graph.py node``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl
from _common import (
    GraphUnavailable,
    chembl_lookup,
    kv,
    make_query,
    note,
    open_graph,
    pubchem_client,
    section,
    show,
    subsection,
)

from modephere.core import CompoundError
from modephere.library.ingest.resolve import Resolver
from modephere.library.standardize import standardize
from modephere.types import ChemblQuery, NameQuery

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.chembl.lookup import ChemblLookup
    from modephere.kg.graph import KnowledgeGraph
    from modephere.library.ingest.pubchem import PubChemClient
    from modephere.types import CompoundQuery, ResolvedCompound

SECTIONS = ("resolve", "chembl", "pubchem", "pharmgkb")


@dataclass
class Findings:
    """What earlier sections learned; later sections and the cross-check use it."""

    query: CompoundQuery
    resolved: ResolvedCompound | None = None
    registration_inchikey: str | None = None
    chembl_parent: str | None = None
    chembl_ids: list[str] = field(default_factory=list)  # parent + every form
    chembl_inchikey: str | None = None
    chembl_names: list[str] = field(default_factory=list)
    chembl_genes: set[str] = field(default_factory=set)
    pubchem_inchikeys: dict[int, str | None] = field(default_factory=dict)
    cids: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. Modephere resolver + standardizer
# --------------------------------------------------------------------------- #


def section_resolve(
    f: Findings, chembl: ChemblLookup | None, client: PubChemClient
) -> None:
    section("1. Modephere: resolution and standardization")
    resolver = Resolver(client, chembl=chembl)
    try:
        resolved = resolver.resolve(f.query)
    except CompoundError as e:
        note(f"resolution failed: {e}")
        return
    f.resolved = resolved
    if resolved.cid is not None:
        f.cids.append(resolved.cid)
    f.chembl_parent = resolved.chembl_id
    kv(
        {
            "query": f"{f.query.kind}: {_query_value(f.query)}",
            "resolved by": resolved.source,
            "title": resolved.title,
            "smiles (as resolved)": resolved.smiles,
            "PubChem CID": resolved.cid,
            "candidate CIDs": list(resolved.candidate_cids) or None,
            "upstream InChIKey": resolved.inchikey,
            "ChEMBL parent": resolved.chembl_id,
            "ChEMBL record": resolved.chembl_record_id,
            "ChEMBL match method": resolved.chembl_match,
            "candidate ChEMBL IDs": list(resolved.candidate_chembl_ids) or None,
            "ChEMBL release": resolved.chembl_release,
            "ambiguous": resolved.is_ambiguous or None,
            "warnings": list(resolved.warnings) or None,
        }
    )
    subsection("Standardized forms")
    try:
        std = standardize(resolved)
    except CompoundError as e:
        note(f"standardization failed: {e}")
        return
    rec = std.record
    f.registration_inchikey = rec.inchikey
    kv(
        {
            "registration InChIKey": rec.inchikey,
            "registration SMILES": rec.registration_smiles,
            "working SMILES": rec.working_smiles,
            "pH 7.4 microspecies": rec.microspecies_smiles,
            "formal charge @ pH 7.4": rec.formal_charge,
            "heavy atoms": rec.num_heavy_atoms,
            "fragments removed": rec.fragments_removed,
            "fragment policy": rec.fragment_policy,
            "protonation": rec.protonation,
            "toolkit": rec.toolkit_version,
        }
    )
    note(
        "registration = canonical tautomer (identity key); working = input "
        "tautomer (2D graph); microspecies = working form protonated (3D, "
        "pharmacophores)"
    )


def _query_value(q: CompoundQuery) -> str:
    return str(q.model_dump(exclude={"kind", "label"}, exclude_none=True))


# --------------------------------------------------------------------------- #
# 2. ChEMBL staging tables
# --------------------------------------------------------------------------- #

_COMPOUND_COLS = (
    "pref_name",
    "molecule_type",
    "max_phase",
    "first_approval",
    "full_molformula",
    "mw_freebase",
    "alogp",
    "hba",
    "hbd",
    "psa",
    "rtb",
    "aromatic_rings",
    "heavy_atoms",
    "num_ro5_violations",
    "usan_stem",
    "canonical_smiles",
    "inchikey",
)
_FLAGS = (
    "oral",
    "parenteral",
    "topical",
    "black_box_warning",
    "withdrawn",
    "prodrug",
    "natural_product",
    "first_in_class",
    "orphan",
    "inorganic",
    "chemical_probe",
)


def _chembl_ids_for(f: Findings, chembl: ChemblLookup) -> str | None:
    """The ChEMBL parent: from the resolver, else by name / InChIKey / ID."""
    if f.chembl_parent:
        return f.chembl_parent
    q = f.query
    hits = []
    if isinstance(q, ChemblQuery):
        hit = chembl.by_chembl_id(q.chembl_id)
        hits = [hit] if hit else []
    elif isinstance(q, NameQuery):
        hits = chembl.by_name(q.name)
    if not hits and f.registration_inchikey:
        hits = chembl.by_inchikey(f.registration_inchikey)
    return chembl.parent_of(hits[0]).chembl_id if hits else None


def section_chembl(f: Findings, chembl: ChemblLookup | None, limit: int) -> None:
    section("2. ChEMBL")
    if chembl is None:
        return
    staging = chembl.staging
    parent = _chembl_ids_for(f, chembl)
    if parent is None:
        note(f"not in {chembl.release} (only drugs / clinical candidates are staged)")
        return
    f.chembl_parent = parent
    compounds = pl.read_parquet(staging / "compounds.parquet")
    family = compounds.filter(
        (pl.col("chembl_id") == parent) | (pl.col("parent_chembl_id") == parent)
    )
    f.chembl_ids = family["chembl_id"].to_list()
    row = family.filter(pl.col("chembl_id") == parent).row(0, named=True)
    f.chembl_inchikey = row["inchikey"]
    note(f"release {chembl.release}; parent {parent}; staging {staging}")

    subsection(f"Parent record {parent}")
    kv({k: row[k] for k in _COMPOUND_COLS})
    flags = [k for k in _FLAGS if row.get(k)]
    kv({"flags": flags or "none"})

    forms = family.filter(pl.col("chembl_id") != parent)
    show(
        forms.select("chembl_id", "pref_name", "max_phase", "full_molformula"),
        limit,
        title=f"Other forms (salts, hydrates) of {parent}: {forms.height}",
    )

    syn = pl.read_parquet(staging / "synonyms.parquet").filter(
        pl.col("chembl_id").is_in(f.chembl_ids)
    )
    f.chembl_names = sorted({row["pref_name"], *syn["synonym"].to_list()} - {None})
    show(
        syn.group_by("syn_type")
        .agg(pl.len().alias("n"), pl.col("synonym").unique().sort().head(8))
        .sort("n", descending=True),
        limit,
        title=f"Synonyms: {syn.height}",
    )

    _chembl_atc(staging, f.chembl_ids, limit)
    targets = pl.read_parquet(staging / "targets.parquet").select(
        pl.col("chembl_id").alias("target_chembl_id"),
        pl.col("pref_name").alias("target"),
        "target_type",
        "organism",
    )
    genes = _target_genes(staging)

    mech = (
        pl.read_parquet(staging / "mechanisms.parquet")
        .filter(pl.col("chembl_id").is_in(f.chembl_ids))
        .join(targets, on="target_chembl_id", how="left")
        .join(genes, on="target_chembl_id", how="left")
    )
    f.chembl_genes |= _flatten(mech["genes"])
    show(
        mech.select(
            "chembl_id",
            "mechanism_of_action",
            "action_type",
            "target_chembl_id",
            "target",
            "genes",
        ),
        limit,
        title=f"Mechanisms of action: {mech.height}",
    )

    ind = (
        pl.read_parquet(staging / "indications.parquet")
        .filter(pl.col("chembl_id").is_in(f.chembl_ids))
        .group_by("mesh_id", "mesh_heading")
        .agg(pl.col("max_phase_for_ind").max().alias("max_phase"))
        .sort(["max_phase", "mesh_heading"], descending=[True, False])
    )
    show(ind, limit, title=f"Indications (MeSH): {ind.height}")

    warn = pl.read_parquet(staging / "warnings.parquet").filter(
        pl.col("chembl_id").is_in(f.chembl_ids)
    )
    show(
        warn.select(
            "chembl_id", "warning_type", "warning_class", "country", "year", "efo_term"
        ),
        limit,
        title=f"Drug warnings: {warn.height}",
    )

    met = pl.read_parquet(staging / "metabolism.parquet").filter(
        pl.col("drug_chembl_id").is_in(f.chembl_ids)
        | pl.col("substrate_chembl_id").is_in(f.chembl_ids)
        | pl.col("metabolite_chembl_id").is_in(f.chembl_ids)
    )
    names = compounds.select(pl.col("chembl_id"), pl.col("pref_name"))
    met = met.join(
        names.rename({"chembl_id": "substrate_chembl_id", "pref_name": "substrate"}),
        on="substrate_chembl_id",
        how="left",
    ).join(
        names.rename({"chembl_id": "metabolite_chembl_id", "pref_name": "metabolite"}),
        on="metabolite_chembl_id",
        how="left",
    )
    show(
        met.select(
            pl.coalesce("substrate", "substrate_chembl_id").alias("substrate"),
            pl.coalesce("metabolite", "metabolite_chembl_id").alias("metabolite"),
            "enzyme_name",
            "conversion",
        ),
        limit,
        title=f"Metabolism records: {met.height}",
    )
    note("unnamed IDs are metabolites outside the staged drug set")

    _chembl_bioactivity(staging, parent, targets=targets, genes=genes, f=f, limit=limit)


def _flatten(series: pl.Series) -> set[str]:
    out: set[str] = set()
    for v in series.drop_nulls().to_list():
        out.update(v)
    return out


def _target_genes(staging: Path) -> pl.DataFrame:
    """target_chembl_id -> sorted gene symbols of its components."""
    comps = pl.read_parquet(staging / "components.parquet").select(
        "accession", "gene_symbols"
    )
    return (
        pl.read_parquet(staging / "target_components.parquet")
        .join(comps, on="accession", how="left")
        .explode("gene_symbols")
        .group_by("target_chembl_id")
        .agg(pl.col("gene_symbols").drop_nulls().unique().sort().alias("genes"))
    )


def _chembl_atc(staging: Path, ids: list[str], limit: int) -> None:
    atc = pl.read_parquet(staging / "atc.parquet").filter(
        pl.col("chembl_id").is_in(ids)
    )
    subsection(f"ATC classification: {atc.height} code(s)")
    if atc.height == 0:
        note("no ATC code")
        return
    classes = {
        r["code"]: r
        for r in pl.read_parquet(staging / "atc_classes.parquet").to_dicts()
    }
    for code in sorted(set(atc["code"].to_list()))[:limit]:
        chain = []
        node = classes.get(code)
        while node is not None:
            chain.append(f"{node['code']} {node['name']}")
            node = classes.get(node["parent_code"]) if node["parent_code"] else None
        print("  " + "  <  ".join(chain or [code]))


def _chembl_bioactivity(
    staging: Path,
    parent: str,
    *,
    targets: pl.DataFrame,
    genes: pl.DataFrame,
    f: Findings,
    limit: int,
) -> None:
    bio = (
        pl.read_parquet(staging / "bioactivity.parquet")
        .filter(pl.col("parent_chembl_id") == parent)
        .join(targets, on="target_chembl_id", how="left")
        .join(genes, on="target_chembl_id", how="left")
        .sort(["n", "pchembl_median"], descending=True)
    )
    raw = (
        pl.scan_parquet(staging / "activities.parquet")
        .filter(pl.col("parent_chembl_id") == parent)
        .select(pl.len())
        .collect()
        .item()
    )
    f.chembl_genes |= _flatten(bio.filter(pl.col("pchembl_median") >= 6)["genes"])
    show(
        bio.select(
            "target_chembl_id",
            pl.col("target").str.slice(0, 28),
            pl.col("organism").str.slice(0, 16),
            "genes",
            "n",
            pl.col("pchembl_median").round(2).alias("median"),
            pl.format(
                "{}-{}", pl.col("pchembl_min").round(1), pl.col("pchembl_max").round(1)
            ).alias("range"),
            pl.col("max_confidence").alias("conf"),
        ),
        limit,
        title=(
            f"Aggregated bioactivity: {bio.height} targets "
            f"(from {raw} raw activity rows in activities.parquet)"
        ),
    )
    note("pChEMBL = -log10(molar); >= 6 is 1 uM or better. Sorted by measurements")


# --------------------------------------------------------------------------- #
# 3. PubChem
# --------------------------------------------------------------------------- #


def section_pubchem(f: Findings, client: PubChemClient, limit: int) -> None:
    section("3. PubChem" + (" (offline: cache only)" if client.offline else ""))
    cids = list(dict.fromkeys(f.cids))
    if not cids:
        names = []
        if isinstance(f.query, NameQuery):
            names.append(f.query.name)
        names += [n for n in f.chembl_names[:1] if n not in names]
        for name in names:
            found = client.name_to_cids(name)
            note(f"name {name!r} -> {found.status} {found.value or ''}")
            if found.value:
                cids = list(found.value)
                break
    if not cids:
        note("no CID to look up (SMILES-only query, or the name is unknown)")
        return
    shown = cids[:limit]
    records = client.properties(shown)
    rows: list[dict[str, Any]] = []
    for cid in shown:
        look = records[cid]
        rec = look.value
        f.pubchem_inchikeys[cid] = rec.inchikey if rec else None
        rows.append(
            {
                "cid": cid,
                "status": str(look.status),
                "title": rec.title if rec else None,
                "inchikey": rec.inchikey if rec else None,
                "iupac_name": rec.iupac_name if rec else None,
                "retrieved_at": str(look.retrieved_at)[:19]
                if look.retrieved_at
                else None,
            }
        )
    show(pl.DataFrame(rows), limit, title=f"CID properties ({len(cids)} candidate(s))")
    first = records[shown[0]].value
    if first is not None:
        subsection(f"CID {first.cid} structures")
        kv(
            {
                "SMILES (isomeric)": first.smiles,
                "connectivity SMILES": first.connectivity_smiles,
                "page": f"https://pubchem.ncbi.nlm.nih.gov/compound/{first.cid}",
            }
        )


# --------------------------------------------------------------------------- #
# 4. PharmGKB through dev_Pgen's graph
# --------------------------------------------------------------------------- #

FIND_DRUGS = """
MATCH (d:Drug)
WHERE toLower(d.name) IN $names OR d.cid IN $cids
RETURN d.name AS name, d.cid AS cid, d.sources AS sources
ORDER BY name
"""
SAME_AS_DRUGS = """
MATCH (d:Drug)-[r:SAME_AS]->(c:Compound) WHERE c.chembl_id = $parent
RETURN d.name AS name, r.method AS method, r.evidence AS evidence
"""
ANN_BY = """
MATCH (a:Annotation)-[:FOR_DRUG]->(d:Drug {name: $name})
RETURN a.source AS source, a.significance AS significance, count(*) AS n
ORDER BY n DESC
"""
ANN_PHENO = """
MATCH (a:Annotation)-[:FOR_DRUG]->(d:Drug {name: $name})
UNWIND coalesce(a.phenotype_categories, ['(none)']) AS category
RETURN category, count(*) AS n ORDER BY n DESC
"""
ANN_GENES = """
MATCH (a:Annotation)-[:FOR_DRUG]->(d:Drug {name: $name})
MATCH (a)-[:CONCERNS_GENE]->(g:Gene)
RETURN g.symbol AS gene, count(DISTINCT a) AS annotations,
       sum(CASE WHEN a.significance = 'yes' THEN 1 ELSE 0 END) AS significant
ORDER BY annotations DESC
"""
ANN_VARIANTS = """
MATCH (a:Annotation)-[:FOR_DRUG]->(d:Drug {name: $name})
MATCH (a)-[:CONCERNS_ALLELE]->(al:Allele)
OPTIONAL MATCH (al)-[:REFERS_TO]->(site)
RETURN al.label AS allele, al.kind AS kind,
       coalesce(site.rsid, site.name) AS refers_to, count(DISTINCT a) AS annotations
ORDER BY annotations DESC
"""
ANN_EXAMPLES = """
MATCH (a:Annotation)-[:FOR_DRUG]->(d:Drug {name: $name})
OPTIONAL MATCH (a)-[:CONCERNS_GENE]->(g:Gene)
WITH a, collect(DISTINCT g.symbol) AS genes
RETURN a.id AS id, a.source AS source, genes, a.alleles AS alleles,
       a.direction AS direction, a.phenotype_categories AS phenotypes,
       a.significance AS significant, a.pmid AS pmid
ORDER BY (a.significance = 'yes') DESC, a.pmid DESC
LIMIT $limit
"""


def section_pharmgkb(f: Findings, limit: int) -> None:
    section("4. PharmGKB (dev_Pgen's graph)")
    names = {n.lower() for n in f.chembl_names}
    if isinstance(f.query, NameQuery):
        names.add(f.query.name.lower())
    if f.resolved and f.resolved.title:
        names.add(f.resolved.title.lower())
    linked = _linked_drug_names(f)
    names |= {n.lower() for n in linked}

    try:
        graph = open_graph("pgen")
    except GraphUnavailable as e:
        note(str(e))
        return
    with graph:
        drugs = pl.DataFrame(
            graph.run(FIND_DRUGS, names=sorted(names), cids=f.cids),
            schema={"name": pl.String, "cid": pl.Int64, "sources": pl.List(pl.String)},
        )
        show(drugs, limit, title=f"Drug nodes matched by name or CID: {drugs.height}")
        if drugs.height == 0:
            note("not a PharmGKB drug (tried: " + ", ".join(sorted(names)[:8]) + ")")
            return
        for name in drugs["name"].to_list():
            _pharmgkb_drug(graph, name, f, limit)


def _linked_drug_names(f: Findings) -> list[str]:
    """Drug names that Modephere's graph links to the ChEMBL parent (SAME_AS)."""
    if not f.chembl_parent:
        return []
    try:
        graph = open_graph("modephere")
    except GraphUnavailable as e:
        note(str(e))
        return []
    with graph:
        rows = graph.run(SAME_AS_DRUGS, parent=f.chembl_parent)
    if rows:
        show(
            pl.DataFrame(rows),
            title=f"SAME_AS links to {f.chembl_parent} in Modephere's graph",
        )
    return [r["name"] for r in rows]


def _pharmgkb_drug(graph: KnowledgeGraph, name: str, f: Findings, limit: int) -> None:
    subsection(f"Drug '{name}'")
    by = pl.DataFrame(graph.run(ANN_BY, name=name))
    if by.height == 0:
        note("no annotations")
        return
    total = int(by["n"].sum())
    print(f"  {total} annotations")
    show(by, limit, title="By PharmGKB table and significance")
    show(pl.DataFrame(graph.run(ANN_PHENO, name=name)), limit, title="By phenotype")
    genes = pl.DataFrame(graph.run(ANN_GENES, name=name))
    show(genes, limit, title=f"Genes: {genes.height}")
    show(
        pl.DataFrame(graph.run(ANN_VARIANTS, name=name)),
        limit,
        title="Alleles / variants (most annotated first)",
    )
    show(
        pl.DataFrame(graph.run(ANN_EXAMPLES, name=name, limit=limit)),
        limit,
        title="Example annotations (significant first)",
    )
    if genes.height and f.chembl_genes:
        pgx = set(genes["gene"].to_list())
        shared = sorted(pgx & f.chembl_genes)
        subsection("Genes in both PharmGKB and ChEMBL")
        print(f"  shared       : {shared or 'none'}")
        note(
            "ChEMBL side = mechanism targets + bioactivity targets with pChEMBL >= 6; "
            "PharmGKB genes are often PK enzymes (CYPs) that ChEMBL does not "
            "list as targets"
        )


# --------------------------------------------------------------------------- #
# Identity cross-check
# --------------------------------------------------------------------------- #


def cross_check(f: Findings) -> None:
    section("Identity cross-check (InChIKey)")
    keys: dict[str, str | None] = {
        "Modephere registration": f.registration_inchikey,
        f"ChEMBL parent {f.chembl_parent or ''}".strip(): f.chembl_inchikey,
    }
    for cid, key in f.pubchem_inchikeys.items():
        keys[f"PubChem CID {cid}"] = key
    ref = f.registration_inchikey
    pad = max(len(k) for k in keys)
    for label, key in keys.items():
        if key is None or ref is None:
            verdict = "-"
        elif key == ref:
            verdict = "identical"
        elif key[:14] == ref[:14]:
            verdict = "same skeleton, stereo / protonation layer differs"
        else:
            verdict = "DIFFERENT connectivity"
        print(f"  {label:<{pad}} : {key or '-':<27}  {verdict}")
    note(
        "the registration key is the canonical tautomer of the neutral parent; "
        "keto/enol tautomers and salts can legitimately differ in connectivity"
    )


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument("query", help="Name, ChEMBL ID, PubChem CID or SMILES.")
    parser.add_argument(
        "--kind",
        default="auto",
        choices=["auto", "name", "chembl_id", "cid", "smiles"],
        help="Force how the query is read (default: guessed).",
    )
    parser.add_argument(
        "--only", nargs="+", choices=SECTIONS, default=list(SECTIONS), metavar="SECTION"
    )
    parser.add_argument(
        "--offline", action="store_true", help="PubChem from the cache only."
    )
    parser.add_argument("--limit", type=int, default=15, help="Rows per table.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    f = Findings(query=make_query(args.query, args.kind))
    print(f"Query: {args.query!r} read as {f.query.kind}")
    chembl = chembl_lookup() if {"resolve", "chembl"} & set(args.only) else None
    with pubchem_client(offline=args.offline) as client:
        if "resolve" in args.only:
            section_resolve(f, chembl, client)
        if "chembl" in args.only:
            section_chembl(f, chembl, args.limit)
        if "pubchem" in args.only:
            section_pubchem(f, client, args.limit)
    if "pharmgkb" in args.only:
        section_pharmgkb(f, args.limit)
    cross_check(f)
    if f.chembl_parent:
        print(
            "\nGraph neighbourhood: uv run python scripts/inspect_graph.py node "
            f"{f.chembl_parent}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
