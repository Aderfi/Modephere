"""Pure KG build: local files + raw cache → typed rows with provenance.

Nothing here touches the network or Neo4j. Every node and relationship is
stamped with the datasets it derives from and the run that produced it;
derived relationships also carry the method and the raw evidence. A missing
cache entry anywhere aborts the whole build (``CacheMissError`` listing every
missing key) before a single row is returned.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from modephere.kg.provenance import Dataset, IngestRun
from modephere.pgx.allele_inference import METHOD as INFERENCE_METHOD
from modephere.pgx.allele_inference import infer_variant
from modephere.pgx.annotation import AnnotationSource
from modephere.pgx.cache import CacheMissError
from modephere.pgx.genotype import METHOD as GENOTYPE_METHOD
from modephere.pgx.genotype import (
    Genotype,
    HaplotypeHgvs,
    Locus,
    ParsedGenotype,
    Role,
    annotation_locus,
    is_snv_site,
    parse_field,
    read_indel_site,
    resolve_indel,
    resolve_null,
    resolve_snv,
    resolve_star,
    strand_ambiguous,
    unresolved_site,
)
from modephere.pgx.labels import AlleleKind, ClassifiedLabel, Structure, classify_label
from modephere.pgx.provenance import ElementStamp, sha256_file
from modephere.pgx.resolve import ResolvedRefSNP, canonicalize_row, resolve_refsnps
from modephere.pgx.sources.allele_tables import LICENSE as CLINPGX_LICENSE
from modephere.pgx.sources.dbsnp import Unresolved
from modephere.pgx.sources.gene import GeneLocation, gene_location
from modephere.pgx.variant import cis_hgvs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from modephere.library.ingest.raw_cache import CacheEntry
    from modephere.pgx.annotation import PGxAnnotation
    from modephere.pgx.ncbi import NCBIClient
    from modephere.pgx.sources.definitions import HaplotypeDefinition
    from modephere.pgx.variant import CanonicalVariant

logger = logging.getLogger(__name__)

PHARMGKB_LICENSE = "CC BY-SA 4.0"
NCBI_LICENSE = "public domain (NCBI)"


class _DatasetIds(BaseModel):
    """Dataset ids every element is stamped with, resolved once per build."""

    annotation: dict[str, str]  # AnnotationSource value -> dataset id
    dbsnp: dict[str, str]  # dbSNP build -> dataset id
    spdi: str

    def for_refsnp(self, resolved: ResolvedRefSNP) -> str:
        return self.dbsnp[resolved.record.dbsnp_build]


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


class NodeRow(BaseModel):
    label: str
    key: dict[str, str]
    props: dict[str, object] = Field(default_factory=dict)
    stamp: ElementStamp


class RelRow(BaseModel):
    type: str
    start_label: str
    start_key: dict[str, str]
    end_label: str
    end_key: dict[str, str]
    props: dict[str, object] = Field(default_factory=dict)
    stamp: ElementStamp


class ResolutionRow(BaseModel):
    label: str
    kind: str
    status: Literal["resolved", "unresolved"]
    reason: str | None
    n_annotations: int
    n_variants: int
    path: Literal["rsid", "haplotype", "structural"] | None


class DisagreementRow(BaseModel):
    issue: Literal["allele_not_in_dbsnp", "rsid_unresolved"]
    haplotype: str
    rsid: str
    pharmvar_spdi: str
    dbsnp_spdis: tuple[str, ...]


class UnresolvedCase(BaseModel):
    """One thing the build could not resolve, and why (``unresolved.log``).

    Categories: ``allele_label`` (a PharmGKB label with no ``REFERS_TO``),
    ``haplotype`` (a definition whose variants did not canonicalize),
    ``genotype`` (a genotype mention left unresolved), ``genotype_skipped``
    (an annotation or field that names no genotype), ``allele_inference``
    (an rsID annotation not narrowed to one variant).
    """

    category: Literal[
        "allele_label", "haplotype", "genotype", "genotype_skipped", "allele_inference"
    ]
    subject: str
    reason: str
    annotation_id: str | None = None
    detail: str | None = None


class KGBuild(BaseModel):
    datasets: list[Dataset]
    run: IngestRun
    nodes: list[NodeRow]
    rels: list[RelRow]
    resolution: list[ResolutionRow]
    disagreements: list[DisagreementRow]
    inference_counts: dict[str, int]
    genotype_counts: dict[str, int] = Field(default_factory=dict)
    unresolved: list[UnresolvedCase] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def pharmgkb_dataset(tsv: Path, annotations_dir: Path) -> Dataset:
    stamps = sorted(annotations_dir.glob("CREATED_*.txt"))
    version = stamps[-1].stem.removeprefix("CREATED_") if stamps else "unknown"
    return Dataset(
        id=f"pharmgkb:{tsv.stem}@{sha256_file(tsv)[:12]}",
        name=f"PharmGKB {tsv.stem}",
        version=version,
        uri="https://www.clinpgx.org/downloads",
        retrieved_at=datetime.fromtimestamp(tsv.stat().st_mtime, UTC),
        sha256=sha256_file(tsv),
        license=PHARMGKB_LICENSE,
    )


def local_file_dataset(logical_id: str, path: Path, name: str) -> Dataset:
    return Dataset(
        id=f"{logical_id}@{sha256_file(path)[:12]}",
        name=name,
        version=datetime.fromtimestamp(path.stat().st_mtime, UTC).date().isoformat(),
        uri=f"file:{path.name}",
        retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
        sha256=sha256_file(path),
    )


def ncbi_dataset(entries: list[CacheEntry], name: str, version: str) -> Dataset:
    """Content-addressed dataset over exactly the cached responses it covers."""
    digest = hashlib.sha256()
    for e in sorted(entries, key=lambda e: e.key):
        digest.update(e.model_dump_json().encode() + b"\n")
    sha = digest.hexdigest()
    latest = max((e.retrieved_at for e in entries), default=datetime.now(UTC))
    return Dataset(
        id=f"{name}@{version}#{sha[:12]}",
        name=name,
        version=version,
        uri="https://www.ncbi.nlm.nih.gov/",
        retrieved_at=latest,
        sha256=sha,
        license=NCBI_LICENSE,
    )


def allele_table_dataset(
    xlsx: Path, gene: str, url: str, retrieved_at: datetime
) -> Dataset:
    return Dataset(
        id=f"clinpgx:{gene}_allele_definition_table@{sha256_file(xlsx)[:12]}",
        name=f"ClinPGx {gene} allele definition table",
        version=retrieved_at.date().isoformat(),
        uri=url,
        retrieved_at=retrieved_at,
        sha256=sha256_file(xlsx),
        license=CLINPGX_LICENSE,
    )


# --------------------------------------------------------------------------- #
# Accumulator — merges repeated nodes/relationships, unioning their sources
# --------------------------------------------------------------------------- #


def _frozen(key: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(key.items()))


def _merge_stamp(old: ElementStamp, new: ElementStamp) -> ElementStamp:
    sources = tuple(dict.fromkeys((*old.sources, *new.sources)))
    return old.model_copy(update={"sources": sources})


class _Graph:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.nodes: dict[tuple, NodeRow] = {}
        self.rels: dict[tuple, RelRow] = {}
        self.unresolved: list[UnresolvedCase] = []

    def unresolved_case(
        self, category: str, subject: str, reason: str, **kw: object
    ) -> None:
        self.unresolved.append(
            UnresolvedCase.model_validate(
                {"category": category, "subject": subject, "reason": reason, **kw}
            )
        )

    def stamp(
        self,
        sources: Iterable[str],
        method: str | None = None,
        evidence: str | None = None,
    ) -> ElementStamp:
        return ElementStamp(
            sources=tuple(dict.fromkeys(sources)),
            run_id=self.run_id,
            method=method,
            evidence=evidence,
        )

    def node(
        self,
        label: str,
        key: dict[str, str],
        props: dict[str, object],
        stamp: ElementStamp,
    ) -> None:
        k = (label, _frozen(key))
        if k in self.nodes:
            old = self.nodes[k]
            merged = {**old.props, **{p: v for p, v in props.items() if v is not None}}
            self.nodes[k] = old.model_copy(
                update={"props": merged, "stamp": _merge_stamp(old.stamp, stamp)}
            )
        else:
            self.nodes[k] = NodeRow(label=label, key=key, props=props, stamp=stamp)

    def rel(
        self,
        rel_type: str,
        start: tuple[str, dict[str, str]],
        end: tuple[str, dict[str, str]],
        stamp: ElementStamp,
        props: dict[str, object] | None = None,
    ) -> None:
        k = (rel_type, start[0], _frozen(start[1]), end[0], _frozen(end[1]))
        if k in self.rels:
            old = self.rels[k]
            self.rels[k] = old.model_copy(
                update={"stamp": _merge_stamp(old.stamp, stamp)}
            )
        else:
            self.rels[k] = RelRow(
                type=rel_type,
                start_label=start[0],
                start_key=start[1],
                end_label=end[0],
                end_key=end[1],
                props=props or {},
                stamp=stamp,
            )

    def variant(
        self, v: CanonicalVariant, stamp: ElementStamp
    ) -> tuple[str, dict[str, str]]:
        self.node("Variant", variant_key(v), v.graph_props(), stamp)
        return ("Variant", variant_key(v))


def variant_key(v: CanonicalVariant) -> dict[str, str]:
    """A variant's node key: its GRCh38 genomic HGVS (``hgvs_g38``)."""
    return {"hgvs_g38": v.hgvs_g}


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_kg(
    *,
    annotations: list[PGxAnnotation],
    definitions: list[HaplotypeDefinition],
    client: NCBIClient,
    drug_cids: Mapping[str, tuple[int, str]],
    datasets: list[Dataset],
    run: IngestRun,
    annotation_datasets: Mapping[str, str] | None = None,
) -> KGBuild:
    """``annotation_datasets`` maps an AnnotationSource value to its Dataset id
    (default ``pharmgkb:<source>``)."""
    missing: list[str] = []

    def on_miss(e: CacheMissError) -> None:
        missing.extend(e.missing)

    classified = {lab: classify_label(lab) for a in annotations for lab in a.variants}
    label_rsids = {c.rsid for c in classified.values() if c.rsid is not None}
    pharmvar_rsids = {
        int(r.rsid[2:])
        for d in definitions
        if d.dataset_id.startswith("pharmvar:")
        for r in d.rows
        if r.rsid and r.rsid[2:].isdigit()
    }
    try:
        refsnps = resolve_refsnps(label_rsids | pharmvar_rsids, client, on_miss)
    except CacheMissError as e:
        on_miss(e)
        refsnps = {}

    defs_by_name: dict[str, HaplotypeDefinition] = {}
    for d in definitions:  # PharmVar first: it wins over a ClinPGx table
        defs_by_name.setdefault(d.name, d)
    canonical: dict[
        str, list[tuple[CanonicalVariant | None, str, tuple[str, ...], str | None]]
    ] = {}
    for name, d in defs_by_name.items():
        rows = []
        for row in d.rows:
            try:
                c = canonicalize_row(row, client)
            except CacheMissError as e:
                on_miss(e)
                continue
            rows.append((c.variant, row.evidence, c.warnings, c.failure))
        canonical[name] = rows

    locations = _gene_locations(classified, client, on_miss)

    if missing:
        raise CacheMissError(sorted(set(missing)))

    g = _Graph(run.id)
    ncbi, spdi_ds, gene_ds = _ncbi_datasets(client, refsnps, locations)
    ids = _DatasetIds(
        annotation={
            src.value: (annotation_datasets or {}).get(
                src.value, f"pharmgkb:{src.value}"
            )
            for src in AnnotationSource
        },
        dbsnp={b: ds.id for b, ds in ncbi.items()},
        spdi=spdi_ds.id,
    )

    _add_refsnps(g, refsnps, ids)
    hap_variants, hap_failed = _add_haplotypes(g, defs_by_name, canonical, ids)
    hap_hgvs = _add_haplotype_hgvs(g, defs_by_name, canonical, hap_failed, ids)
    structural = _StructuralAlleles(
        g, locations, gene_ds.id if gene_ds else None, hap_hgvs
    )
    _add_variant_metadata(g, defs_by_name)
    disagreements = _cross_check(defs_by_name, canonical, refsnps)
    inference = _add_annotations(
        g,
        annotations=annotations,
        classified=classified,
        refsnps=refsnps,
        drug_cids=drug_cids,
        ids=ids,
    )
    resolution = _add_alleles(
        g,
        annotations=annotations,
        classified=classified,
        refsnps=refsnps,
        defs_by_name=defs_by_name,
        hap_variants=hap_variants,
        hap_failed=hap_failed,
        structural=structural,
        ids=ids,
    )
    genotypes = _add_genotypes(
        g,
        annotations=annotations,
        classified=classified,
        refsnps=refsnps,
        defs_by_name=defs_by_name,
        hap_failed=hap_failed,
        structural=structural,
        ids=ids,
    )

    # Exactly the datasets some element cites — no more, no fewer.
    stamps = [n.stamp for n in g.nodes.values()] + [r.stamp for r in g.rels.values()]
    cited = {s for stamp in stamps for s in stamp.sources}
    extra = [gene_ds] if gene_ds else []
    available = {d.id: d for d in [*datasets, *ncbi.values(), spdi_ds, *extra]}
    uncited = cited - set(available)
    if uncited:
        msg = f"elements cite datasets that were not provided: {sorted(uncited)}"
        raise ValueError(msg)
    out_datasets = [available[i] for i in sorted(cited)]

    logger.info(
        "KG build: %d nodes, %d relationships, %d labels, %d disagreements",
        len(g.nodes),
        len(g.rels),
        len(resolution),
        len(disagreements),
    )
    return KGBuild(
        datasets=out_datasets,
        run=run,
        nodes=list(g.nodes.values()),
        rels=list(g.rels.values()),
        resolution=resolution,
        disagreements=disagreements,
        inference_counts=dict(inference),
        genotype_counts=dict(genotypes),
        unresolved=g.unresolved,
    )


def _ncbi_datasets(
    client: NCBIClient,
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    locations: Mapping[str, GeneLocation],
) -> tuple[dict[str, Dataset], Dataset, Dataset | None]:
    """dbSNP datasets per build, the SPDI dataset, the NCBI Gene dataset."""
    ncbi: dict[str, Dataset] = {}
    builds = {
        r.record.dbsnp_build for r in refsnps.values() if isinstance(r, ResolvedRefSNP)
    }
    rs_entries = [
        e for e in client.dbsnp_cache.entries() if e.key.startswith("dbsnp:rs")
    ]
    for b in sorted(builds):
        ds = ncbi_dataset(
            [e for e in rs_entries if e.source_version == b], "dbsnp_esummary", b
        )
        ncbi[b] = ds
    spdi_ds = ncbi_dataset(client.spdi_cache.entries(), "ncbi_spdi", "variation-v0")
    gene_entries = [
        e
        for e in client.gene_cache.entries()
        if e.key.removeprefix("gene:") in locations
    ]
    gene_ds = (
        ncbi_dataset(gene_entries, "ncbi_gene", "gene-esummary")
        if gene_entries
        else None
    )
    return ncbi, spdi_ds, gene_ds


def _gene_locations(
    classified: Mapping[str, ClassifiedLabel],
    client: NCBIClient,
    on_miss: Callable[[CacheMissError], None],
) -> dict[str, GeneLocation]:
    """NCBI Gene spans of every gene a whole-gene deletion label names."""
    symbols = {
        c.haplotype_name.split("*", 1)[0]
        for c in classified.values()
        if c.structure is Structure.GENE_DELETION and c.haplotype_name
    }
    out: dict[str, GeneLocation] = {}
    for symbol in sorted(symbols):
        try:
            location = gene_location(client.gene_summary(symbol), symbol)
        except CacheMissError as e:
            on_miss(e)
            continue
        if location is not None:
            out[symbol] = location
    return out


def _add_refsnps(
    g: _Graph, refsnps: Mapping[int, ResolvedRefSNP | Unresolved], ids: _DatasetIds
) -> None:
    for resolved in refsnps.values():
        if not isinstance(resolved, ResolvedRefSNP):
            continue
        rec = resolved.record
        dbsnp_id = ids.for_refsnp(resolved)
        stamp = g.stamp([dbsnp_id])
        current = ("RefSNP", {"rsid": rec.current_rsid})
        g.node(
            "RefSNP",
            {"rsid": rec.current_rsid},
            {"dbsnp_build": rec.dbsnp_build},
            stamp,
        )
        if rec.merged:
            g.node(
                "RefSNP",
                {"rsid": rec.requested_rsid},
                {"dbsnp_build": rec.dbsnp_build, "merged": True},
                stamp,
            )
            g.rel(
                "MERGED_INTO",
                ("RefSNP", {"rsid": rec.requested_rsid}),
                current,
                g.stamp(
                    [dbsnp_id],
                    "dbsnp_esummary_merge",
                    f"requested {rec.requested_rsid}; record snp_id {rec.current_rsid}",
                ),
            )
        for gene in rec.genes:
            g.node(
                "Gene", {"symbol": gene.symbol}, {"entrez_id": gene.entrez_id}, stamp
            )
        for v in resolved.variants:
            vref = g.variant(v, stamp)
            g.node(
                "Variant",
                variant_key(v),
                {
                    "so_terms": list(rec.so_terms) or None,
                    "p_hgvs": rec.protein_hgvs.get(v.spdi),
                },
                stamp,
            )
            g.rel(
                "HAS_ALLELE",
                current,
                vref,
                g.stamp([dbsnp_id], "dbsnp_esummary_spdi", v.spdi),
            )
            for gene in rec.genes:
                g.rel(
                    "IN_GENE",
                    vref,
                    ("Gene", {"symbol": gene.symbol}),
                    g.stamp(
                        [dbsnp_id],
                        "dbsnp_esummary_gene",
                        f"{gene.symbol}:{gene.entrez_id}",
                    ),
                )


def _add_haplotypes(
    g: _Graph,
    defs_by_name: Mapping[str, HaplotypeDefinition],
    canonical: Mapping[str, list],
    ids: _DatasetIds,
) -> tuple[dict[str, int], set[str]]:
    n_variants: dict[str, int] = {}
    failed: set[str] = set()
    for name, d in defs_by_name.items():
        g.node(
            "Haplotype",
            {"name": name},
            {"gene": d.gene, "is_reference": d.is_reference},
            g.stamp([d.dataset_id]),
        )
        rows = canonical.get(name, [])
        failures = [f for _v, _e, _w, f in rows if f]
        if d.unresolved_reason or failures:
            failed.add(name)
            g.unresolved_case(
                "haplotype",
                name,
                d.unresolved_reason or "canonicalization_failed",
                detail="; ".join(failures) or None,
            )
            continue  # never write a partial cis set
        method = (
            "pharmvar_haplotype_row"
            if d.dataset_id.startswith("pharmvar:")
            else "clinpgx_allele_table"
        )
        for variant, evidence, warnings, _failure in rows:
            assert variant is not None
            vref = g.variant(variant, g.stamp([d.dataset_id, ids.spdi]))
            note = evidence + ("".join(f"\tncbi_warning={w}" for w in warnings))
            g.rel(
                "HAS_VARIANT",
                ("Haplotype", {"name": name}),
                vref,
                g.stamp(
                    [d.dataset_id, ids.spdi],
                    f"{method}+ncbi_spdi_canonical",
                    note,
                ),
                props={"phase": "cis"},
            )
        n_variants[name] = len(rows)
    for name, d in defs_by_name.items():
        if d.core_name != name and d.core_name in defs_by_name:
            g.rel(
                "SUBALLELE_OF",
                ("Haplotype", {"name": name}),
                ("Haplotype", {"name": d.core_name}),
                g.stamp(
                    [d.dataset_id], "haplotype_name_suffix", f"{name} -> {d.core_name}"
                ),
            )
    return n_variants, failed


def _cross_check(
    defs_by_name: Mapping[str, HaplotypeDefinition],
    canonical: Mapping[str, list],
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
) -> list[DisagreementRow]:
    out: list[DisagreementRow] = []
    for name, d in defs_by_name.items():
        if not d.dataset_id.startswith("pharmvar:"):
            continue
        for row, (variant, _e, _w, _f) in zip(
            d.rows, canonical.get(name, []), strict=False
        ):
            if variant is None or not row.rsid or not row.rsid[2:].isdigit():
                continue
            resolved = refsnps.get(int(row.rsid[2:]))
            dbsnp = (
                tuple(v.spdi for v in resolved.variants)
                if isinstance(resolved, ResolvedRefSNP)
                else ()
            )
            if variant.spdi not in dbsnp:
                issue = "allele_not_in_dbsnp" if dbsnp else "rsid_unresolved"
                out.append(
                    DisagreementRow(
                        issue=issue,
                        haplotype=name,
                        rsid=row.rsid,
                        pharmvar_spdi=variant.spdi,
                        dbsnp_spdis=dbsnp,
                    )
                )
    return out


def _add_annotations(
    g: _Graph,
    *,
    annotations: list[PGxAnnotation],
    classified: Mapping[str, ClassifiedLabel],
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    drug_cids: Mapping[str, tuple[int, str]],
    ids: _DatasetIds,
) -> Counter[str]:
    inference: Counter[str] = Counter()
    for a in annotations:
        src = ids.annotation[a.source.value]
        stamp = g.stamp([src])
        aref = ("Annotation", {"id": a.id})
        g.node(
            "Annotation",
            {"id": a.id},
            {
                "source": a.source.value,
                "pmid": a.pmid,
                "phenotype_categories": a.phenotype_categories,
                "direction": a.direction,
                "significance": a.significance,
                "associated": a.associated,
                "alleles": a.alleles,
                "comparison_alleles": a.comparison_alleles,
            },
            stamp,
        )
        for symbol in a.genes:
            g.node("Gene", {"symbol": symbol}, {}, stamp)
            g.rel("CONCERNS_GENE", aref, ("Gene", {"symbol": symbol}), stamp)
        for drug in a.drugs:
            cid = drug_cids.get(drug)
            dstamp = g.stamp([src, cid[1]]) if cid else stamp
            g.node("Drug", {"name": drug}, {"cid": cid[0] if cid else None}, dstamp)
            g.rel("FOR_DRUG", aref, ("Drug", {"name": drug}), stamp)
        for label in a.variants:
            g.rel("CONCERNS_ALLELE", aref, ("Allele", {"label": label}), stamp)

        # Allele-specific link only when the annotation is about a single rsID.
        if len(a.variants) != 1:
            continue
        c = classified[a.variants[0]]
        resolved = refsnps.get(c.rsid) if c.rsid is not None else None
        if not isinstance(resolved, ResolvedRefSNP):
            continue
        result = infer_variant(a.alleles, a.comparison_alleles, resolved.variants)
        inference[result.reason or "linked"] += 1
        if result.reason is not None:
            g.unresolved_case(
                "allele_inference",
                a.variants[0],
                result.reason,
                annotation_id=a.id,
                detail=result.evidence,
            )
        if result.variant is not None:
            g.rel(
                "IMPLICATES",
                aref,
                ("Variant", variant_key(result.variant)),
                g.stamp(
                    [src, ids.for_refsnp(resolved)], INFERENCE_METHOD, result.evidence
                ),
            )
    return inference


class _LabelResolution(BaseModel):
    """How one label resolved: its ``REFERS_TO`` target, or why there is none."""

    reason: str | None = None
    n_variants: int = 0
    path: Literal["rsid", "haplotype", "structural"] | None = None
    target: tuple[str, dict[str, str]] | None = None
    method: str | None = None
    sources: tuple[str, ...] = ()  # cited besides the annotation files
    props: dict[str, object] = Field(default_factory=dict)


def _resolve_rsid(
    c: ClassifiedLabel,
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    ids: _DatasetIds,
) -> _LabelResolution:
    resolved = refsnps.get(c.rsid) if c.rsid is not None else None
    if isinstance(resolved, ResolvedRefSNP):
        return _LabelResolution(
            n_variants=len(resolved.variants),
            path="rsid",
            target=("RefSNP", {"rsid": resolved.record.requested_rsid}),
            method="label_rsid",
            sources=(ids.for_refsnp(resolved),),
        )
    if isinstance(resolved, Unresolved):
        return _LabelResolution(reason=resolved.reason)
    return _LabelResolution(reason="rsid_not_found")


def _resolve_star(
    name: str,
    defs_by_name: Mapping[str, HaplotypeDefinition],
    hap_variants: Mapping[str, int],
    hap_failed: set[str],
) -> _LabelResolution:
    d = defs_by_name.get(name)
    if d is None:
        return _LabelResolution(reason="not_in_pharmvar_or_tables")
    if name in hap_failed:
        return _LabelResolution(reason=d.unresolved_reason or "canonicalization_failed")
    return _LabelResolution(
        n_variants=hap_variants.get(name, 0),
        path="haplotype",
        target=("Haplotype", {"name": name}),
        method="label_haplotype_name",
        sources=(d.dataset_id,),
    )


def _resolve_structural(
    c: ClassifiedLabel,
    structural: _StructuralAlleles,
    star: Callable[[str], _LabelResolution],
    label_sources: Sequence[str],
) -> _LabelResolution:
    """Deletion / HLA alleles name their own haplotype; ``*2xN`` the one it copies."""
    assert c.haplotype_name is not None
    if c.structure is Structure.DUPLICATION:
        base = star(c.haplotype_name)
        if base.target is None:
            return base
        return base.model_copy(
            update={
                "path": "structural",
                "method": "label_duplication",
                "props": {"copies": c.copies, "min_copies": c.copies or 2},
            }
        )
    structural.ensure(c.haplotype_name, label_sources)
    return _LabelResolution(
        path="structural",
        target=("Haplotype", {"name": c.haplotype_name}),
        method=f"label_{c.structure}",
        sources=structural.sources(c.haplotype_name),
    )


def _add_alleles(
    g: _Graph,
    *,
    annotations: list[PGxAnnotation],
    classified: Mapping[str, ClassifiedLabel],
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    defs_by_name: Mapping[str, HaplotypeDefinition],
    hap_variants: Mapping[str, int],
    hap_failed: set[str],
    structural: _StructuralAlleles,
    ids: _DatasetIds,
) -> list[ResolutionRow]:
    counts = Counter(lab for a in annotations for lab in a.variants)
    sources_of: dict[str, set[str]] = {}
    for a in annotations:  # every file that mentions a label is a source of it
        for lab in a.variants:
            sources_of.setdefault(lab, set()).add(ids.annotation[a.source.value])

    def star(name: str) -> _LabelResolution:
        return _resolve_star(name, defs_by_name, hap_variants, hap_failed)

    rows: list[ResolutionRow] = []
    for label, c in classified.items():
        label_sources = sorted(sources_of[label])
        if c.kind is AlleleKind.RSID:
            res = _resolve_rsid(c, refsnps, ids)
        elif c.kind is AlleleKind.STAR and c.haplotype_name is not None:
            res = star(c.haplotype_name)
        elif c.structure is not None and c.haplotype_name is not None:
            res = _resolve_structural(c, structural, star, label_sources)
        else:
            res = _LabelResolution(reason=c.reason)
        status: Literal["resolved", "unresolved"] = (
            "resolved" if res.target else "unresolved"
        )
        g.node(
            "Allele",
            {"label": label},
            {
                "kind": c.kind.value,
                "status": status,
                "reason": None if res.target else res.reason,
            },
            g.stamp(label_sources),
        )
        if res.target is not None:
            g.rel(
                "REFERS_TO",
                ("Allele", {"label": label}),
                res.target,
                g.stamp([*label_sources, *res.sources], res.method, label),
                props=res.props,
            )
        if res.target is None:
            g.unresolved_case(
                "allele_label",
                label,
                res.reason or "unresolved",
                detail=f"kind={c.kind.value}; annotations={counts[label]}",
            )
        rows.append(
            ResolutionRow(
                label=label,
                kind=c.kind.value,
                status=status,
                reason=None if res.target else res.reason,
                n_annotations=counts[label],
                n_variants=res.n_variants,
                path=res.path,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Structural alleles: whole-gene deletions and HLA alleles
# --------------------------------------------------------------------------- #


class _StructuralAlleles:
    """Haplotypes that are no set of small variants, created as they are named.

    A deletion (``CYP2D6*5``, ``GSTM1*0``) gets HGVS over its gene's span per
    assembly NCBI Gene places it on (``hgvs_g38``, ``hgvs_t2t``); an HLA
    allele is identified by its IPD-IMGT/HLA name alone.
    """

    def __init__(
        self,
        g: _Graph,
        locations: Mapping[str, GeneLocation],
        gene_dataset: str | None,
        hap_hgvs: dict[str, HaplotypeHgvs],
    ) -> None:
        self._g = g
        self._locations = locations
        self._gene_dataset = gene_dataset
        self.hap_hgvs = hap_hgvs

    def _location(self, name: str) -> GeneLocation | None:
        c = classify_label(name)
        if c.structure is not Structure.GENE_DELETION:
            return None
        return self._locations.get(name.split("*", 1)[0])

    def sources(self, name: str) -> tuple[str, ...]:
        if self._location(name) is not None and self._gene_dataset:
            return (self._gene_dataset,)
        return ()

    def ensure(self, name: str, sources: Iterable[str]) -> bool:
        """Create ``name``'s node if it is a deletion or HLA allele."""
        c = classify_label(name)
        if c.structure not in {Structure.GENE_DELETION, Structure.HLA_TYPED}:
            return False
        assert c.structure is not None
        props: dict[str, object] = {
            "gene": name.split("*", 1)[0],
            "is_reference": False,
            "structure": c.structure.value,
        }
        location = self._location(name)
        if location is not None:
            for assembly, span in (("g38", location.grch38), ("t2t", location.t2t)):
                if span is not None:
                    props[f"hgvs_{assembly}"] = cis_hgvs(span.seq_id, (span.deletion,))
            if location.grch38 is not None:
                self.hap_hgvs[name] = (
                    location.grch38.seq_id,
                    (location.grch38.deletion,),
                )
        self._g.node(
            "Haplotype",
            {"name": name},
            props,
            self._g.stamp([*sources, *self.sources(name)]),
        )
        return True


# --------------------------------------------------------------------------- #
# HGVS views: haplotypes as cis sets, variants with their rsIDs / star alleles
# --------------------------------------------------------------------------- #


def _natural(name: str) -> list[object]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _add_haplotype_hgvs(
    g: _Graph,
    defs_by_name: Mapping[str, HaplotypeDefinition],
    canonical: Mapping[str, list],
    hap_failed: set[str],
    ids: _DatasetIds,
) -> dict[str, HaplotypeHgvs]:
    """``Haplotype.hgvs_g38``: the cis set, ``NC_…:g.[c1;c2]``; ``g.=`` = reference.

    A reference haplotype has no variants; its sequence is the one every other
    haplotype of the gene sits on. Returns name -> (seq id, changes).
    """
    variants: dict[str, list[CanonicalVariant]] = {
        name: [v for v, _e, _w, _f in canonical.get(name, []) if v is not None]
        for name in defs_by_name
        if name not in hap_failed
    }
    gene_seqs: dict[str, set[str]] = defaultdict(set)
    for name, vs in variants.items():
        gene_seqs[defs_by_name[name].gene].update(v.seq_id for v in vs)
    out: dict[str, HaplotypeHgvs] = {}
    for name, vs in variants.items():
        d = defs_by_name[name]
        seqs = {v.seq_id for v in vs} or (
            set(gene_seqs[d.gene]) if d.is_reference else set()  # a copy: popped below
        )
        if len(seqs) != 1:
            continue  # no variants and not the reference, or two chromosomes
        seq_id = seqs.pop()
        unique = {v.hgvs_g: v for v in vs}.values()
        changes = tuple(
            v.change for v in sorted(unique, key=lambda v: (v.pos, v.change))
        )
        out[name] = (seq_id, changes)
        g.node(
            "Haplotype",
            {"name": name},
            {"hgvs_g38": cis_hgvs(seq_id, changes)},
            g.stamp([d.dataset_id, ids.spdi] if vs else [d.dataset_id]),
        )
    return out


def _add_variant_metadata(
    g: _Graph, defs_by_name: Mapping[str, HaplotypeDefinition]
) -> None:
    """``Variant.rsids`` / ``Variant.star_alleles``: labels kept as metadata.

    Read off the ``HAS_ALLELE`` / ``HAS_VARIANT`` relationships, so they say
    nothing the graph does not; star alleles are core names (``CYP2C19*2``)
    of non-reference haplotypes, sub-alleles stay on the ``Haplotype`` nodes.
    """
    rsids: dict[str, set[str]] = defaultdict(set)
    stars: dict[str, set[str]] = defaultdict(set)
    for r in g.rels.values():
        if r.end_label != "Variant":
            continue
        hgvs = r.end_key["hgvs_g38"]
        if r.type == "HAS_ALLELE":
            rsids[hgvs].add(r.start_key["rsid"])
        elif r.type == "HAS_VARIANT":
            d = defs_by_name[r.start_key["name"]]
            if not d.is_reference:
                stars[hgvs].add(d.core_name)
    for k, node in g.nodes.items():
        if node.label != "Variant":
            continue
        hgvs = node.key["hgvs_g38"]
        props = {
            **node.props,
            "rsids": sorted(rsids.get(hgvs, ()), key=_natural),
            "star_alleles": sorted(stars.get(hgvs, ()), key=_natural),
        }
        g.nodes[k] = node.model_copy(update={"props": props})


# --------------------------------------------------------------------------- #
# Genotypes: the allele pairs an annotation tested and compared against
# --------------------------------------------------------------------------- #


def _add_genotypes(
    g: _Graph,
    *,
    annotations: list[PGxAnnotation],
    classified: Mapping[str, ClassifiedLabel],
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    defs_by_name: Mapping[str, HaplotypeDefinition],
    hap_failed: set[str],
    structural: _StructuralAlleles,
    ids: _DatasetIds,
) -> Counter[str]:
    """``(:Annotation)-[:TESTED|COMPARED_TO]->(:Genotype)-[:HAS_COPY]->(…)``.

    Counts ``<status>:<reason>`` per genotype mention, ``skipped:<reason>``
    per annotation (or field) that names none, and ``strand_assumed`` per
    annotation linked on the forward-strand assumption.
    """
    counts: Counter[str] = Counter()
    for a in annotations:
        locus = annotation_locus([classified[lab] for lab in a.variants])
        if isinstance(locus, str):
            counts[f"skipped:{locus}"] += 1
            g.unresolved_case(
                "genotype_skipped",
                ", ".join(a.variants) or "-",
                locus,
                annotation_id=a.id,
                detail=f"Alleles={a.alleles!r}; Comparison={a.comparison_alleles!r}",
            )
            continue
        src = ids.annotation[a.source.value]
        fields = [
            (role, field or "", parse_field(field, locus))
            for role, field in (
                (Role.TESTED, a.alleles),
                (Role.COMPARED_TO, a.comparison_alleles),
            )
        ]
        for role, field, parsed in fields:
            if parsed is None:
                counts["skipped:unparsed_alleles"] += 1
                g.unresolved_case(
                    "genotype_skipped",
                    locus.name,
                    "unparsed_alleles",
                    annotation_id=a.id,
                    detail=f"{role.value}={field!r}",
                )
        readable = [(r, f, p) for r, f, p in fields if p is not None]
        if locus.kind == "rsid":
            resolved = _site_genotypes(locus, readable, refsnps, ids)
        else:
            resolved = _star_genotypes(
                locus,
                readable,
                src=src,
                defs_by_name=defs_by_name,
                hap_failed=hap_failed,
                structural=structural,
                ids=ids,
            )
        strand_assumed = any(assumed for *_rest, assumed in resolved)
        counts["strand_assumed"] += strand_assumed
        for role, field, gt, extra, _assumed in resolved:
            counts[f"{gt.status}:{gt.reason or '-'}"] += 1
            if gt.status == "unresolved":
                g.unresolved_case(
                    "genotype",
                    gt.name,
                    gt.reason or "unresolved",
                    annotation_id=a.id,
                    detail=f"{role.value}={field!r}",
                )
            _write_genotype(
                g,
                gt,
                role=role,
                field=field,
                annotation=a.id,
                sources=[src, *extra],
                annotation_source=src,
                strand_assumed=strand_assumed,
            )
    return counts


_Resolved = list[tuple[Role, str, Genotype, list[str], bool]]


def _star_genotypes(
    locus: Locus,
    fields: list[tuple[Role, str, list[ParsedGenotype]]],
    *,
    src: str,
    defs_by_name: Mapping[str, HaplotypeDefinition],
    hap_failed: set[str],
    structural: _StructuralAlleles,
    ids: _DatasetIds,
) -> _Resolved:
    def haplotype_reason(name: str) -> str | None:
        if structural.ensure(name, [src]):
            return None
        return _resolve_star(name, defs_by_name, {}, hap_failed).reason

    resolve = resolve_null if locus.kind == "null" else resolve_star
    out: _Resolved = []
    for role, field, parsed in fields:
        for p in parsed:
            gt = resolve(
                p,
                locus.name,
                haplotype_reason=haplotype_reason,
                haplotype_hgvs=structural.hap_hgvs,
            )
            extra = {
                s
                for c in gt.copies
                for s in (
                    (defs_by_name[c.allele].dataset_id,)
                    if c.allele in defs_by_name
                    else structural.sources(c.allele)
                )
            }
            if gt.hgvs_g38 and any(c.allele in defs_by_name for c in gt.copies):
                extra.add(ids.spdi)
            out.append((role, field, gt, sorted(extra), False))
    return out


def _site_genotypes(
    locus: Locus,
    fields: list[tuple[Role, str, list[ParsedGenotype]]],
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    ids: _DatasetIds,
) -> _Resolved:
    resolved = refsnps.get(locus.rsid) if locus.rsid is not None else None
    flat = [(role, field, p) for role, field, parsed in fields for p in parsed]
    if not isinstance(resolved, ResolvedRefSNP):
        reason = (
            resolved.reason if isinstance(resolved, Unresolved) else "rsid_not_found"
        )
        return [
            (role, field, unresolved_site(p, locus.name, reason), [], False)
            for role, field, p in flat
        ]
    site, rsid = resolved.variants, resolved.record.current_rsid
    sources = [ids.for_refsnp(resolved)]
    if is_snv_site(site):
        assumed = strand_ambiguous([p for _r, _f, p in flat], site)
        genotypes = [resolve_snv(p, rsid, site) for _r, _f, p in flat]
    else:
        assumed = False
        reading = read_indel_site([p for _r, _f, p in flat], site)
        if isinstance(reading, str):
            genotypes = [unresolved_site(p, rsid, reading) for _r, _f, p in flat]
        else:
            tokens, mapping = reading
            genotypes = [
                resolve_indel(p, t, mapping, rsid, site)
                for (_r, _f, p), t in zip(flat, tokens, strict=True)
            ]
    return [
        (role, field, gt, sources if gt.status == "resolved" else [], assumed)
        for (role, field, _p), gt in zip(flat, genotypes, strict=True)
    ]


def _write_genotype(
    g: _Graph,
    gt: Genotype,
    *,
    role: Role,
    field: str,
    annotation: str,
    sources: list[str],
    annotation_source: str,
    strand_assumed: bool,
) -> None:
    gref = ("Genotype", {"name": gt.name})
    known = g.nodes.get(("Genotype", _frozen({"name": gt.name})))
    if not (
        known and known.props.get("status") == "resolved" and gt.status != "resolved"
    ):
        # The same name read by another annotation: a resolved reading wins.
        g.node(
            "Genotype",
            {"name": gt.name},
            {
                "locus": gt.locus,
                "zygosity": gt.zygosity,
                "status": gt.status,
                "reason": gt.reason,
                "ref_copies": gt.ref_copies,
                "gene_copies": gt.gene_copies,
                "gene_copies_min": gt.gene_copies_min,
                "hgvs_g38": gt.hgvs_g38,
            },
            g.stamp(sources),
        )
    g.rel(
        role.value,
        ("Annotation", {"id": annotation}),
        gref,
        g.stamp([annotation_source], GENOTYPE_METHOD, field),
        props={"strand_assumed": True} if strand_assumed else None,
    )
    key = "name" if gt.target == "Haplotype" else "hgvs_g38"
    for c in gt.copies:
        g.rel(
            "HAS_COPY",
            gref,
            (gt.target, {key: c.allele}),
            g.stamp(sources, GENOTYPE_METHOD, gt.name),
            props={"copies": c.copies, "min_copies": c.min_copies},
        )


__all__ = [
    "DisagreementRow",
    "KGBuild",
    "NodeRow",
    "RelRow",
    "ResolutionRow",
    "UnresolvedCase",
    "allele_table_dataset",
    "build_kg",
    "variant_key",
    "local_file_dataset",
    "ncbi_dataset",
    "pharmgkb_dataset",
]
