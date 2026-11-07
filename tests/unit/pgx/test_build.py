"""Tests for fetch planning, shared resolution, and the pure graph build."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

import pytest

from modephere.config import NCBISettings
from modephere.kg.provenance import Dataset, IngestRun
from modephere.pgx.annotation import AnnotationSource, PGxAnnotation
from modephere.pgx.build import build_kg
from modephere.pgx.cache import CacheMissError
from modephere.pgx.fetch import plan_fetch, run_fetch
from modephere.pgx.ncbi import NCBIClient
from modephere.pgx.sources.allele_tables import load_allele_table
from modephere.pgx.sources.pharmvar import load_pharmvar

from .conftest import FIXTURES, FakeSession, as_transport

NOW = datetime(2026, 10, 18, tzinfo=UTC)
PV_ID = "pharmvar:CYP2C19@pharmvar-6.2.18"
UGT_ID = "clinpgx:UGT1A1_allele_definition_table"
DRUG_ID = "local:drug_name_cache"
HEADER = (
    "#version=pharmvar-6.2.18\n"
    "Haplotype Name\tGene\trsID\tReferenceSequence\tVariant Start\tVariant Stop\t"
    "Reference Allele\tVariant Allele\tType\n"
)
CYP2C19_ROWS = (
    "CYP2C19*38\tCYP2C19\t\tREFERENCE\t.\t\t\t\t\n"
    "CYP2C19*2\tCYP2C19\trs12769205\tNC_000010.11\t94775367\t94775367\tA\tG\tsubstitution\n"
    "CYP2C19*2\tCYP2C19\trs4244285\tNC_000010.11\t94781859\t94781859\tG\tA\tsubstitution\n"
    "CYP2C19*2\tCYP2C19\trs3758581\tNC_000010.11\t94842866\t94842866\tA\tG\tsubstitution\n"
    "CYP2C19*2.001\tCYP2C19\trs4244285\tNC_000010.11\t94781859\t94781859\tG\tA\tsubstitution\n"
)


def _ann(aid: str, variants: list[str], **kw) -> PGxAnnotation:
    return PGxAnnotation(
        id=aid,
        source=AnnotationSource.DRUG,
        variants=variants,
        genes=kw.pop("genes", ["CYP2D6"]),
        drugs=kw.pop("drugs", ["codeine"]),
        phenotype_categories=["Metabolism/PK"],
        **kw,
    )


ANNOTATIONS = [
    _ann("A_RS3892097_CT", ["rs3892097"], alleles="CT", comparison_alleles="CC"),
    _ann("A_STAR", ["CYP2C19*2", "rs17879456", "rs4244285"], genes=["CYP2C19"]),
    _ann("A_UGT", ["UGT1A1*28"], genes=["UGT1A1"], drugs=["irinotecan"]),
    _ann("A_MISC", ["HLA-B*58:01", "rs999999999999", "NAT1*99"], genes=["HLA-B"]),
]


def _dataset(i: str) -> Dataset:
    return Dataset(
        id=i, name=i, version="v", uri="u", retrieved_at=NOW, sha256="0" * 64
    )


def _client(cache, *, offline: bool) -> NCBIClient:
    ticks = itertools.count(0, 1000)
    return NCBIClient(
        NCBISettings(),
        cache,
        offline=offline,
        transport=None if offline else as_transport(FakeSession()),
        sleep=lambda s: None,
        clock=lambda: float(next(ticks)),
    )


def _build(tmp_path, pharmvar_rows: str = CYP2C19_ROWS, annotations=None):
    annotations = annotations or ANNOTATIONS
    tsv = tmp_path / "CYP2C19.NC_000010.11.haplotypes.tsv"
    tsv.write_text(HEADER + pharmvar_rows)
    pharmvar = load_pharmvar(tsv, PV_ID)
    tables = load_allele_table(FIXTURES / "allele_tables" / "UGT1A1.xlsx", UGT_ID)
    cache = tmp_path / "raw"
    plan = plan_fetch(annotations, pharmvar, {"CYP2C19"})
    run_fetch(plan, pharmvar + tables, _client(cache, offline=False))
    datasets = [_dataset(f"pharmgkb:{s.value}") for s in AnnotationSource]
    datasets += [_dataset(PV_ID), _dataset(UGT_ID), _dataset(DRUG_ID)]
    return build_kg(
        annotations=annotations,
        definitions=pharmvar + tables,
        client=_client(cache, offline=True),
        drug_cids={"codeine": (5284371, DRUG_ID)},
        datasets=datasets,
        run=IngestRun.start({}),
    )


@pytest.fixture
def kg_build(tmp_path):
    return _build(tmp_path)


def rel(build, rel_type, start, end):
    for r in build.rels:
        if (
            r.type == rel_type
            and (r.start_label, r.start_key) == start
            and (r.end_label, r.end_key) == end
        ):
            return r
    return None


def node(build, label, key):
    return next((n for n in build.nodes if n.label == label and n.key == key), None)


def vkey(build, spdi):
    """The ``hgvs_g38`` key of the variant whose GRCh38 SPDI is ``spdi``."""
    (n,) = [
        n for n in build.nodes if n.label == "Variant" and n.props["spdi_g38"] == spdi
    ]
    return n.key


def test_plan_fetch_collects_label_and_pharmvar_rsids(tmp_path):
    tsv = tmp_path / "CYP2C19.NC_000010.11.haplotypes.tsv"
    tsv.write_text(HEADER + CYP2C19_ROWS)
    plan = plan_fetch(ANNOTATIONS, load_pharmvar(tsv, PV_ID), {"CYP2C19"})
    assert {3892097, 17879456, 4244285, 999999999999, 12769205, 3758581} <= plan.rsids
    assert plan.allele_table_genes == {"UGT1A1", "NAT1"}


def test_star_and_rsid_meet_on_same_variant(kg_build):
    v = {"hgvs_g38": "NC_000010.11:g.94781859G>A"}
    assert rel(
        kg_build, "HAS_VARIANT", ("Haplotype", {"name": "CYP2C19*2"}), ("Variant", v)
    )
    assert rel(
        kg_build, "HAS_ALLELE", ("RefSNP", {"rsid": "rs4244285"}), ("Variant", v)
    )
    variant = node(kg_build, "Variant", v)
    assert set(variant.stamp.sources) >= {PV_ID}
    assert any(s.startswith("dbsnp_esummary@") for s in variant.stamp.sources)
    assert variant.props["spdi_g38"] == "NC_000010.11:94781858:G:A"
    assert variant.props["pos_g38"] == 94781859
    assert "spdi" not in variant.props and "assembly" not in variant.props


def test_cis_phase_and_suballele(kg_build):
    r = rel(
        kg_build,
        "HAS_VARIANT",
        ("Haplotype", {"name": "CYP2C19*2"}),
        ("Variant", {"hgvs_g38": "NC_000010.11:g.94775367A>G"}),
    )
    assert r.props["phase"] == "cis"
    assert "rs12769205" in r.stamp.evidence
    assert rel(
        kg_build,
        "SUBALLELE_OF",
        ("Haplotype", {"name": "CYP2C19*2.001"}),
        ("Haplotype", {"name": "CYP2C19*2"}),
    )
    assert node(kg_build, "Haplotype", {"name": "CYP2C19*38"}).props["is_reference"]


def test_merged_labels_share_current_refsnp(kg_build):
    assert rel(
        kg_build,
        "MERGED_INTO",
        ("RefSNP", {"rsid": "rs17879456"}),
        ("RefSNP", {"rsid": "rs4244285"}),
    )
    assert rel(
        kg_build,
        "REFERS_TO",
        ("Allele", {"label": "rs17879456"}),
        ("RefSNP", {"rsid": "rs17879456"}),
    )
    assert not any(
        r.type == "HAS_ALLELE" and r.start_key == {"rsid": "rs17879456"}
        for r in kg_build.rels
    )


def test_ugt1a1_28_canonical_repeat(kg_build):
    r = rel(
        kg_build,
        "HAS_VARIANT",
        ("Haplotype", {"name": "UGT1A1*28"}),
        (
            "Variant",
            vkey(kg_build, "NC_000002.12:233760233:ATATATATATATATA:ATATATATATATATATA"),
        ),
    )
    assert r is not None
    assert UGT_ID in r.stamp.sources
    assert "TA(8)" in r.stamp.evidence


def test_implicates_specific_alt(kg_build):
    r = rel(
        kg_build,
        "IMPLICATES",
        ("Annotation", {"id": "A_RS3892097_CT"}),
        ("Variant", {"hgvs_g38": "NC_000022.11:g.42128945C>T"}),
    )
    assert r.stamp.method == "annotation_alleles_forward_strand"
    assert "'CT'" in r.stamp.evidence
    # an annotation with several labels is never narrowed to one variant
    assert not any(
        r.type == "IMPLICATES" and r.start_key == {"id": "A_STAR"}
        for r in kg_build.rels
    )


def test_variant_in_gene_from_dbsnp(kg_build):
    r = rel(
        kg_build,
        "IN_GENE",
        ("Variant", {"hgvs_g38": "NC_000022.11:g.42128945C>T"}),
        ("Gene", {"symbol": "CYP2D6"}),
    )
    assert r.stamp.method == "dbsnp_esummary_gene"
    assert node(kg_build, "Gene", {"symbol": "CYP2D6"}).props["entrez_id"] == 1565


def test_drug_cid_carries_its_source(kg_build):
    drug = node(kg_build, "Drug", {"name": "codeine"})
    assert drug.props["cid"] == 5284371
    assert DRUG_ID in drug.stamp.sources


def test_every_element_stamped(kg_build):
    known = {d.id for d in kg_build.datasets}
    for row in kg_build.nodes + kg_build.rels:
        assert row.stamp.sources
        assert row.stamp.run_id == kg_build.run.id
        assert set(row.stamp.sources) <= known, row


def test_unresolved_reasons(kg_build):
    reasons = {r.label: (r.status, r.reason) for r in kg_build.resolution}
    assert reasons["HLA-B*58:01"] == ("resolved", None)  # a named HLA allele
    assert reasons["rs999999999999"] == ("unresolved", "rsid_not_found")
    assert reasons["NAT1*99"] == ("unresolved", "not_in_pharmvar_or_tables")
    assert reasons["CYP2C19*2"] == ("resolved", None)
    assert reasons["UGT1A1*28"] == ("resolved", None)


def test_resolution_counts_variants(kg_build):
    row = next(r for r in kg_build.resolution if r.label == "CYP2C19*2")
    assert row.n_variants == 3
    assert row.n_annotations == 1
    assert row.path == "haplotype"


def test_disagreement_reported(tmp_path):
    conflict = CYP2C19_ROWS.replace(
        "rs4244285\tNC_000010.11\t94781859\t94781859\tG\tA",
        "rs4244285\tNC_000010.11\t94781860\t94781860\tG\tA",
        1,
    )
    build = _build(tmp_path, conflict)
    d = next(d for d in build.disagreements if d.rsid == "rs4244285")
    assert d.haplotype == "CYP2C19*2"
    assert d.issue == "allele_not_in_dbsnp"
    assert d.pharmvar_spdi == "NC_000010.11:94781859:G:A"
    assert d.pharmvar_spdi not in d.dbsnp_spdis


def test_unplaceable_pharmvar_rsid_is_its_own_issue(kg_build):
    # rs12769205 has no recorded GRCh38 placement in the fixtures.
    d = next(d for d in kg_build.disagreements if d.rsid == "rs12769205")
    assert d.issue == "rsid_unresolved"
    assert d.dbsnp_spdis == ()


def test_build_with_empty_cache_reports_every_missing_key(tmp_path):
    with pytest.raises(CacheMissError) as e:
        build_kg(
            annotations=ANNOTATIONS,
            definitions=[],
            client=_client(tmp_path / "empty", offline=True),
            drug_cids={},
            datasets=[_dataset(f"pharmgkb:{s.value}") for s in AnnotationSource],
            run=IngestRun.start({}),
        )
    assert len(e.value.missing) >= 4  # einfo-free: one key per rsID label


# ----- final-review fixes ---------------------------------------------------- #


def test_dataset_ids_are_content_addressed(tmp_path):
    from modephere.pgx.build import pharmgkb_dataset

    tsv = tmp_path / "var_drug_ann.tsv"
    tsv.write_text("a\n")
    first = pharmgkb_dataset(tsv, tmp_path).id
    tsv.write_text("b\n")
    second = pharmgkb_dataset(tsv, tmp_path).id
    assert first.startswith("pharmgkb:var_drug_ann@")
    assert first != second


def test_refsnps_cite_the_build_they_were_fetched_under(tmp_path):
    import json as _json

    _build(tmp_path)  # populates the cache under Build250306-1408.1
    cache = tmp_path / "raw" / "dbsnp_esummary.jsonl"
    lines = [_json.loads(x) for x in cache.read_text().splitlines()]
    newer = next(x for x in lines if x["key"] == "dbsnp:rs4244285")
    newer["source_version"] = "Build260101-0000.1"
    with cache.open("a") as f:
        f.write(_json.dumps(newer) + "\n")
    build = _build(tmp_path)
    by_rsid = {n.key["rsid"]: n for n in build.nodes if n.label == "RefSNP"}
    assert any(
        s.startswith("dbsnp_esummary@Build260101-0000.1#")
        for s in by_rsid["rs4244285"].stamp.sources
    )
    assert any(
        s.startswith("dbsnp_esummary@Build250306-1408.1#")
        for s in by_rsid["rs3892097"].stamp.sources
    )
    assert sum(d.id.startswith("dbsnp_esummary@") for d in build.datasets) == 2


def test_allele_cites_every_file_that_mentions_it(tmp_path):
    extra = PGxAnnotation(
        id="A_PHENO",
        source=AnnotationSource.PHENO,
        variants=["rs3892097"],
        genes=["CYP2D6"],
        phenotype_categories=["Other"],
    )
    build = _build(tmp_path, annotations=[*ANNOTATIONS, extra])
    allele = node(build, "Allele", {"label": "rs3892097"})
    assert {"pharmgkb:var_drug_ann", "pharmgkb:var_pheno_ann"} <= set(
        allele.stamp.sources
    )


def test_run_fetch_continues_past_a_persistently_failing_input(tmp_path):
    from modephere.core import ResourceError
    from modephere.pgx.sources.definitions import DefinitionRow, HaplotypeDefinition

    class _Flaky(FakeSession):
        def get(self, url, timeout):
            if "94781858" in url and "canonical" in url:
                self.calls += 1
                from .conftest import FakeResponse, as_transport

                return FakeResponse(500, {})
            return super().get(url, timeout)

    defs = [
        HaplotypeDefinition(
            name="G*1",
            gene="G",
            core_name="G*1",
            dataset_id="x",
            rows=(
                DefinitionRow(spdi_input="NC_000010.11:94781858:G:A", evidence="bad"),
                DefinitionRow(spdi_input="NC_000010.11:94775366:A:G", evidence="good"),
            ),
        )
    ]
    ticks = itertools.count(0, 1000)
    client = NCBIClient(
        NCBISettings(),
        tmp_path,
        offline=False,
        transport=as_transport(_Flaky()),
        sleep=lambda s: None,
        clock=lambda: float(next(ticks)),
    )
    with pytest.raises(ResourceError, match="1 request"):
        run_fetch(plan_fetch([], [], set()), defs, client)
    assert client.spdi_cache.get("spdi_canonical:NC_000010.11:94775366:A:G") is not None


# --------------------------------------------------------------------------- #
# HGVS identity, label metadata, genotypes
# --------------------------------------------------------------------------- #


def test_variant_carries_its_labels_as_metadata(kg_build):
    v = node(kg_build, "Variant", {"hgvs_g38": "NC_000010.11:g.94781859G>A"})
    assert v.props["rsids"] == ["rs4244285"]
    assert v.props["star_alleles"] == ["CYP2C19*2"]  # core name, not *2.001
    # a variant known only from dbSNP still stands on its own
    other = node(kg_build, "Variant", {"hgvs_g38": "NC_000022.11:g.42128945C>T"})
    assert other.props["star_alleles"] == []


def test_haplotype_hgvs_is_its_cis_set(kg_build):
    hap = node(kg_build, "Haplotype", {"name": "CYP2C19*2"})
    assert hap.props["hgvs_g38"] == (
        "NC_000010.11:g.[94775367A>G;94781859G>A;94842866A>G]"
    )
    reference = node(kg_build, "Haplotype", {"name": "CYP2C19*38"})
    assert reference.props["hgvs_g38"] == "NC_000010.11:g.="


def test_every_reference_haplotype_of_a_gene_gets_its_hgvs(tmp_path):
    rows = CYP2C19_ROWS + "CYP2C19*38.001\tCYP2C19\t\tREFERENCE\t.\t\t\t\t\n"
    build = _build(tmp_path, pharmvar_rows=rows)
    for name in ("CYP2C19*38", "CYP2C19*38.001"):
        hap = node(build, "Haplotype", {"name": name})
        assert hap.props["hgvs_g38"] == "NC_000010.11:g.=", name


def test_site_genotypes_of_an_annotation(kg_build):
    a = ("Annotation", {"id": "A_RS3892097_CT"})
    het = ("Genotype", {"name": "rs3892097:C/T"})
    ref = ("Genotype", {"name": "rs3892097:C/C"})
    tested = rel(kg_build, "TESTED", a, het)
    assert tested.stamp.method == "pharmgkb_genotype_notation"
    assert tested.stamp.evidence == "CT"
    assert rel(kg_build, "COMPARED_TO", a, ref)
    copy = rel(
        kg_build,
        "HAS_COPY",
        het,
        ("Variant", {"hgvs_g38": "NC_000022.11:g.42128945C>T"}),
    )
    assert copy.props == {"copies": 1, "min_copies": 1}
    assert any(s.startswith("dbsnp_esummary@") for s in copy.stamp.sources)
    assert node(kg_build, *ref).props["ref_copies"] == 2
    assert not any(
        r.type == "HAS_COPY" and r.start_key == ref[1] for r in kg_build.rels
    )


def test_diplotypes_keep_which_two_haplotypes(tmp_path):
    annotations = [
        _ann(
            "A_DIPLO",
            ["CYP2C19*2", "CYP2C19*38"],
            genes=["CYP2C19"],
            alleles="*2/*2 + *2/*38",
            comparison_alleles="*38/*38",
        )
    ]
    build = _build(tmp_path, annotations=annotations)
    a = ("Annotation", {"id": "A_DIPLO"})
    hom, het = "CYP2C19*2/*2", "CYP2C19*2/*38"
    assert rel(build, "TESTED", a, ("Genotype", {"name": hom}))
    assert rel(build, "TESTED", a, ("Genotype", {"name": het}))
    assert rel(build, "COMPARED_TO", a, ("Genotype", {"name": "CYP2C19*38/*38"}))

    def copies(name):
        return {
            r.end_key["name"]: r.props["copies"]
            for r in build.rels
            if r.type == "HAS_COPY" and r.start_key == {"name": name}
        }

    assert copies(hom) == {"CYP2C19*2": 2}
    assert copies(het) == {"CYP2C19*2": 1, "CYP2C19*38": 1}
    gt = node(build, "Genotype", {"name": het})
    assert gt.props["hgvs_g38"] == (
        "NC_000010.11:g.[94775367A>G;94781859G>A;94842866A>G];[=]"
    )
    assert PV_ID in gt.stamp.sources
    assert build.genotype_counts["resolved:-"] == 3


def test_structural_and_hla_genotypes(tmp_path):
    annotations = [
        _ann("A_HLA", ["HLA-B*58:01"], genes=["HLA-B"], alleles="*58:01"),
        _ann("A_DEL", ["CYP2D6*5"], alleles="*5"),
        _ann("A_HYBRID", ["CYP2D6*36"], alleles="*36"),
        _ann("A_PHENO", ["CYP2C19 poor metabolizer"], genes=["CYP2C19"], alleles="*2"),
    ]
    build = _build(tmp_path, annotations=annotations)
    gt = node(build, "Genotype", {"name": "HLA-B*58:01"})
    assert (gt.props["status"], gt.props["hgvs_g38"]) == ("resolved", None)
    hla = node(build, "Haplotype", {"name": "HLA-B*58:01"})
    assert hla.props["structure"] == "hla_typed"

    deletion = node(build, "Haplotype", {"name": "CYP2D6*5"})
    assert deletion.props["structure"] == "gene_deletion"
    assert deletion.props["hgvs_g38"] == "NC_000022.11:g.(?_42126499)_(42130810_?)del"
    assert deletion.props["hgvs_t2t"] == "NC_060946.1:g.(?_42605990)_(42610301_?)del"
    assert any(s.startswith("ncbi_gene@") for s in deletion.stamp.sources)
    assert rel(
        build,
        "REFERS_TO",
        ("Allele", {"label": "CYP2D6*5"}),
        ("Haplotype", {"name": "CYP2D6*5"}),
    )
    carrier = node(build, "Genotype", {"name": "CYP2D6*5"})
    assert carrier.props["hgvs_g38"] == (
        "NC_000022.11:g.[(?_42126499)_(42130810_?)del];[?]"
    )

    hybrid = node(build, "Genotype", {"name": "CYP2D6*36"})
    assert (hybrid.props["status"], hybrid.props["reason"]) == (
        "unresolved",
        "hybrid_allele",
    )
    assert build.genotype_counts["skipped:not_a_genotype_locus"] == 1


def test_strand_ambiguous_report_is_linked_but_marked(tmp_path):
    # a lone 'T' at a C>A/G/T site reads as a plausible 'A' on the other strand
    annotations = [_ann("A_LONE", ["rs3892097"], alleles="T")]
    build = _build(tmp_path, annotations=annotations)
    tested = rel(
        build,
        "TESTED",
        ("Annotation", {"id": "A_LONE"}),
        ("Genotype", {"name": "rs3892097:T"}),
    )
    assert tested.props == {"strand_assumed": True}
    assert build.genotype_counts["strand_assumed"] == 1


def test_every_unresolved_case_is_recorded(tmp_path):
    annotations = [
        *ANNOTATIONS,
        _ann("A_HYBRID", ["CYP2D6*36"], alleles="*36"),
        _ann("A_PHENO", ["CYP2C19 poor metabolizer"], genes=["CYP2C19"], alleles="*2"),
    ]
    build = _build(tmp_path, annotations=annotations)
    cases = {(c.category, c.subject, c.reason) for c in build.unresolved}
    assert ("allele_label", "NAT1*99", "not_in_pharmvar_or_tables") in cases
    assert ("allele_label", "rs999999999999", "rsid_not_found") in cases
    assert ("genotype", "CYP2D6*36", "hybrid_allele") in cases
    assert (
        "genotype_skipped",
        "CYP2C19 poor metabolizer",
        "not_a_genotype_locus",
    ) in cases
    hybrid = next(
        c for c in build.unresolved if c.subject == "CYP2D6*36" and c.annotation_id
    )
    assert (hybrid.annotation_id, hybrid.detail) == ("A_HYBRID", "TESTED='*36'")


def test_variant_carries_dbsnp_consequences(tmp_path):
    annotations = [
        *ANNOTATIONS,
        _ann("A_RS1065852", ["rs1065852"], alleles="A", comparison_alleles="G"),
    ]
    build = _build(tmp_path, annotations=annotations)
    splice = node(build, "Variant", {"hgvs_g38": "NC_000022.11:g.42128945C>T"})
    assert "splice_acceptor_variant" in splice.props["so_terms"]
    assert "p_hgvs" not in splice.props  # dbSNP lists no protein change for it
    # rs1065852: each alternative allele gets its own protein change
    p34s = node(build, "Variant", {"hgvs_g38": "NC_000022.11:g.42130692G>A"})
    p34a = node(build, "Variant", {"hgvs_g38": "NC_000022.11:g.42130692G>C"})
    assert p34s.props["p_hgvs"] == "NP_000097.3:p.Pro34Ser"
    assert p34a.props["p_hgvs"] == "NP_000097.3:p.Pro34Ala"
    assert p34s.props["so_terms"] == ["missense_variant", "coding_sequence_variant"]
