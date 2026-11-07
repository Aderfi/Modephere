"""Tests for downloading ClinPGx allele-definition tables."""

from __future__ import annotations

from modephere.pgx.fetch import (
    CLINPGX_TABLE_URL,
    download_allele_tables,
    read_table_index,
)

from .conftest import FIXTURES, FakeResponse, as_transport


class _TableSession:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:  # noqa: ARG002
        self.urls.append(url)
        if url == CLINPGX_TABLE_URL.format(gene="TPMT"):
            return FakeResponse(
                200, content=(FIXTURES / "allele_tables" / "TPMT.xlsx").read_bytes()
            )
        return FakeResponse(404)


def test_download_records_ok_and_not_found(tmp_path):
    session = _TableSession()
    index = download_allele_tables(
        {"TPMT", "NAT1"},
        tmp_path,
        transport=as_transport(session),
        sleep=lambda s: None,
    )
    assert index["TPMT"].status == "ok"
    assert (tmp_path / "TPMT.xlsx").read_bytes() == (
        FIXTURES / "allele_tables" / "TPMT.xlsx"
    ).read_bytes()
    assert len(index["TPMT"].sha256 or "") == 64
    assert index["NAT1"].status == "not_found"
    assert read_table_index(tmp_path).keys() == {"TPMT", "NAT1"}


def test_rerun_downloads_nothing(tmp_path):
    download_allele_tables(
        {"TPMT", "NAT1"},
        tmp_path,
        transport=as_transport(_TableSession()),
        sleep=lambda s: None,
    )
    session = _TableSession()
    download_allele_tables(
        {"TPMT", "NAT1"},
        tmp_path,
        transport=as_transport(session),
        sleep=lambda s: None,
    )
    assert session.urls == []
    assert len((tmp_path / "index.jsonl").read_text().splitlines()) == 2


def test_refresh_retries_tables_that_were_not_ok(tmp_path):
    download_allele_tables(
        {"NAT1"},
        tmp_path,
        transport=as_transport(_TableSession()),
        sleep=lambda s: None,
    )
    session = _TableSession()
    index = download_allele_tables(
        {"NAT1"},
        tmp_path,
        transport=as_transport(session),
        sleep=lambda s: None,
        refresh=True,
    )
    assert len(session.urls) == 1
    assert index["NAT1"].status == "not_found"
    assert read_table_index(tmp_path)["NAT1"].retrieved_at >= index["NAT1"].retrieved_at


def test_cli_fetch_has_refresh_flag():
    import modephere.pgx.__main__ as cli

    args = cli._build_parser().parse_args(["fetch", "--refresh"])
    assert args.refresh
