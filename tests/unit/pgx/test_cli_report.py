"""Resolution report and the ``python -m modephere.pgx`` guards."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import polars as pl

import modephere.pgx.__main__ as cli
from modephere.pgx.annotation import AnnotationSource, PGxAnnotation
from modephere.pgx.build import DisagreementRow, ResolutionRow, UnresolvedCase
from modephere.pgx.export import (
    ANNOTATION_GENOTYPE_SCHEMA,
    ANNOTATION_ROW_SCHEMA,
    HAPLOTYPE_SCHEMA,
    VARIANT_SCHEMA,
    annotation_genotypes,
    annotation_rows,
    haplotypes,
    variants,
)
from modephere.pgx.inputs import Inputs, PgxLayout
from modephere.pgx.report import UNRESOLVED_LOG, write_report

from .test_write import small_build

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _build_with_rows():
    return small_build().model_copy(
        update={
            "resolution": [
                ResolutionRow(
                    label="CYP2C19*2",
                    kind="star",
                    status="resolved",
                    reason=None,
                    n_annotations=3,
                    n_variants=3,
                    path="haplotype",
                ),
                ResolutionRow(
                    label="HLA-B*58:01",
                    kind="hla",
                    status="unresolved",
                    reason="structural_allele",
                    n_annotations=2,
                    n_variants=0,
                    path=None,
                ),
            ],
            "disagreements": [
                DisagreementRow(
                    issue="allele_not_in_dbsnp",
                    haplotype="X*2",
                    rsid="rs1",
                    pharmvar_spdi="NC_000010.11:1:G:A",
                    dbsnp_spdis=("NC_000010.11:1:G:T",),
                ),
            ],
            "inference_counts": {"linked": 8, "allele_strand_mismatch": 2},
        }
    )


def test_report_files_and_columns(tmp_path: Path) -> None:
    path = write_report(_build_with_rows(), tmp_path)
    df = pl.read_csv(path, separator="\t")
    assert set(df.columns) >= {"label", "kind", "status", "reason", "n_annotations"}
    assert len(df) == 2
    dis = pl.read_csv(tmp_path / "disagreements.tsv", separator="\t")
    assert dis["dbsnp_spdis"].to_list() == ["NC_000010.11:1:G:T"]


def test_report_logs_strand_mismatch_rate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="modephere.pgx.report"):
        write_report(_build_with_rows(), tmp_path)
    assert "strand mismatches 2/10 (0.2000)" in caplog.text
    assert "resolved mentions 3/5" in caplog.text


def _layout(tmp_path: Path) -> PgxLayout:
    return PgxLayout(
        pharmgkb=tmp_path / "pharmgkb",
        pharmvar=tmp_path / "pharmvar",
        name_cache=tmp_path / "drugs" / "cache.json",
        aligned_tsv=tmp_path / "drugs" / "train.tsv",
        ncbi_cache=tmp_path / "ncbi",
        allele_tables=tmp_path / "tables",
        reports=tmp_path / "reports",
        export=tmp_path / "export",
    )


def _inputs(labels: tuple[str, ...] = ()) -> Inputs:
    return Inputs(
        annotations=[
            PGxAnnotation(id="a", source=AnnotationSource.DRUG, variants=list(labels))
        ],
        pharmvar=[],
        tables=[],
        datasets=[],
        drug_cids={},
    )


def _patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inputs: Inputs, connect: Any
) -> None:
    monkeypatch.setattr(cli.PgxLayout, "from_settings", lambda: _layout(tmp_path))
    monkeypatch.setattr(cli, "load_inputs", lambda layout: inputs)
    monkeypatch.setattr(cli.KnowledgeGraph, "connect", classmethod(connect))


def test_ingest_cache_miss_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connected: list[bool] = []
    _patch(
        monkeypatch,
        tmp_path,
        _inputs(("rs4244285",)),
        lambda cls: connected.append(True),
    )
    assert cli.main(["ingest"]) == 1
    assert connected == []


def test_ingest_fails_when_a_needed_allele_table_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connected: list[bool] = []
    _patch(
        monkeypatch, tmp_path, _inputs(("TPMT*3A",)), lambda cls: connected.append(True)
    )
    monkeypatch.setattr(cli, "build_kg", lambda **kw: small_build())
    assert cli.main(["ingest"]) == 1
    assert connected == []


class _FakeGraph:
    def __init__(self) -> None:
        self.replace: list[bool] = []

    def __enter__(self) -> _FakeGraph:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def audit(self) -> dict[str, int]:
        return {"nodes_missing": 0, "rels_missing": 0, "unknown_sources": 0}


def test_ingest_passes_replace_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _FakeGraph()
    _patch(monkeypatch, tmp_path, _inputs(), lambda cls: graph)
    monkeypatch.setattr(cli, "build_kg", lambda **kw: small_build())

    def fake_write(g: Any, build: Any, *, replace: bool) -> Any:
        graph.replace.append(replace)
        from modephere.pgx.write import WriteSummary

        return WriteSummary("run1", 2, 3, 2, int(replace))

    monkeypatch.setattr(cli, "write_build", fake_write)
    assert cli.main(["ingest"]) == 0
    assert cli.main(["ingest", "--replace"]) == 0
    assert graph.replace == [False, True]
    assert (tmp_path / "reports" / "resolution_report.tsv").exists()


class _RowsGraph:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.params: dict[str, Any] = {}

    def run(self, cypher: str, /, **params: Any) -> list[dict[str, Any]]:
        self.params = params
        return self.rows


def test_annotation_rows_keep_full_drug_identity() -> None:
    row = {
        "annotation_id": "1",
        "source": "var_drug_ann",
        "pmid": "123",
        "drugs": [
            {
                "name": "clopidogrel",
                "cid": 60606,
                "chembl_id": "CHEMBL1771",
                "inchikey": "GKTWGGQPFAXNFI-HNNXBMFYSA-N",
            },
            {"name": "no-cid drug", "cid": None, "chembl_id": None, "inchikey": None},
        ],
        "genes": ["CYP2C19"],
        "genotypes": ["CYP2C19*2"],
        "implicated_hgvs": [],
        "alleles": "*2",
        "comparison_alleles": "*1",
        "phenotype_categories": ["Efficacy"],
        "direction": "decreased",
        "significance": "yes",
        "associated": True,
    }
    graph = _RowsGraph([row])
    frame = annotation_rows(graph, sources=["var_drug_ann"])  # type: ignore[arg-type]
    assert frame.schema == pl.Schema(ANNOTATION_ROW_SCHEMA)
    drugs = frame["drugs"][0].to_list()
    assert drugs[0]["inchikey"] == "GKTWGGQPFAXNFI-HNNXBMFYSA-N"
    assert drugs[1]["cid"] is None  # a drug without a CID is kept, not dropped
    assert graph.params == {"sources": ["var_drug_ann"]}


def test_annotation_genotypes_carry_dosage() -> None:
    row = {
        "annotation_id": "1",
        "role": "TESTED",
        "genotype": "CYP2C19*2/*2",
        "strand_assumed": False,
        "locus": "CYP2C19",
        "zygosity": "diploid",
        "status": "resolved",
        "reason": None,
        "hgvs_g38": "NC_000010.11:g.[94781859G>A];[94781859G>A]",
        "ref_copies": None,
        "gene_copies": 2,
        "gene_copies_min": 2,
        "haplotypes": [
            {"name": "CYP2C19*2", "copies": 2, "min_copies": 2, "structure": None}
        ],
        "variants": [
            {"hgvs_g38": "NC_000010.11:g.94781859G>A", "copies": 2, "min_copies": 2}
        ],
    }
    frame = annotation_genotypes(_RowsGraph([row]))  # type: ignore[arg-type]
    assert frame.schema == pl.Schema(ANNOTATION_GENOTYPE_SCHEMA)
    assert frame["variants"][0].to_list()[0]["copies"] == 2


def test_variants_table_carries_consequences() -> None:
    row = {
        "hgvs_g38": "NC_000010.11:g.94942290C>T",
        "spdi_g38": "NC_000010.11:94942289:C:T",
        "genes": ["CYP2C9"],
        "rsids": ["rs1799853"],
        "star_alleles": ["CYP2C9*2"],
        "so_terms": ["missense_variant", "coding_sequence_variant"],
        "p_hgvs": "NP_000762.2:p.Arg144Cys",
        "sources": ["dbsnp"],
        "run_id": "r1",
    }
    frame = variants(_RowsGraph([row]))  # type: ignore[arg-type]
    assert frame.schema == pl.Schema(VARIANT_SCHEMA)
    assert frame["p_hgvs"][0] == "NP_000762.2:p.Arg144Cys"


def test_haplotypes_table_lists_cis_members() -> None:
    row = {
        "name": "CYP2C19*2.001",
        "gene": "CYP2C19",
        "is_reference": False,
        "structure": None,
        "hgvs_g38": "NC_000010.11:g.[94781859G>A;94842866A>G]",
        "hgvs_t2t": None,
        "variants": ["NC_000010.11:g.94781859G>A", "NC_000010.11:g.94842866A>G"],
        "suballele_of": "CYP2C19*2",
        "sources": ["pharmvar"],
        "run_id": "r1",
    }
    frame = haplotypes(_RowsGraph([row]))  # type: ignore[arg-type]
    assert frame.schema == pl.Schema(HAPLOTYPE_SCHEMA)
    assert frame["variants"][0].to_list()[1] == "NC_000010.11:g.94842866A>G"


def test_unresolved_log_lists_every_case(tmp_path: Path) -> None:
    cases = [
        UnresolvedCase(
            category="genotype",
            subject="CYP2D6*1/*36",
            reason="hybrid_allele",
            annotation_id="A1",
            detail="TESTED='*1/*36'",
        ),
        UnresolvedCase(
            category="allele_label",
            subject="NAT1*99",
            reason="not_in_pharmvar_or_tables",
            detail="kind=star; annotations=1",
        ),
    ]
    build = _build_with_rows().model_copy(update={"unresolved": cases})
    (tmp_path / UNRESOLVED_LOG).write_text("stale line from an older run\n")
    write_report(build, tmp_path)
    lines = (tmp_path / UNRESOLVED_LOG).read_text().splitlines()
    assert "stale line" not in "\n".join(lines)  # rewritten per run
    assert "2 unresolved cases" in lines[0]
    assert lines[1].endswith(
        "allele_label | not_in_pharmvar_or_tables | NAT1*99 | annotation=- | "
        "kind=star; annotations=1"
    )
    assert lines[2].endswith(
        "genotype | hybrid_allele | CYP2D6*1/*36 | annotation=A1 | TESTED='*1/*36'"
    )
