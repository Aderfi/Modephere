"""Tests for the raw cache and the cached, rate-limited NCBI client."""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from modephere.config import NCBISettings
from modephere.core import DataError, ResourceError
from modephere.library.ingest.raw_cache import CacheEntry, RawCache
from modephere.pgx.cache import CacheMissError
from modephere.pgx.ncbi import NCBIClient

from .conftest import FakeResponse, FakeSession, as_transport


def _entry(key: str) -> CacheEntry:
    return CacheEntry(
        key=key, request="u", status="ok", retrieved_at=datetime.now(UTC), response={}
    )


def _client(tmp_path, session, *, slept=None, settings=None):
    ticks = itertools.count(0, 1000)  # clock always far ahead: no rate-limit sleeps
    return NCBIClient(
        settings or NCBISettings(),
        tmp_path,
        offline=False,
        transport=as_transport(session),
        sleep=(slept.append if slept is not None else lambda s: None),
        clock=lambda: float(next(ticks)),
    )


def _cached_keys(path) -> list[str]:
    return [json.loads(line)["key"] for line in path.read_text().splitlines()]


def test_put_is_idempotent(tmp_path):
    cache = RawCache(tmp_path / "x.jsonl")
    entry = _entry("dbsnp:rs1")
    cache.put(entry)
    cache.put(entry)
    assert len((tmp_path / "x.jsonl").read_text().splitlines()) == 1
    assert RawCache(tmp_path / "x.jsonl").get("dbsnp:rs1") == entry


def test_esummary_fetches_only_missing_and_records_not_found(tmp_path):
    session = FakeSession()
    client = _client(tmp_path, session)
    out = client.esummary_snp([3892097, 999999999999])
    assert out[3892097].status == "ok"
    assert out[3892097].source_version == "Build250306-1408.1"
    assert out[999999999999].status == "not_found"
    calls = session.calls
    client.esummary_snp([3892097, 999999999999])
    assert session.calls == calls  # second call served from cache


def test_merged_rsid_cached_under_requested_id(tmp_path):
    out = _client(tmp_path, FakeSession()).esummary_snp([17879456])
    assert out[17879456].response["snp_id"] == 4244285


def test_batches_of_200(tmp_path):
    session = FakeSession()
    _client(tmp_path, session).esummary_snp(list(range(1, 402)))
    esummary_calls = [u for u in session.urls if "esummary" in u]
    assert len(esummary_calls) == 3


def test_resume_after_failure_does_not_duplicate(tmp_path):
    session = FakeSession(fail_after=2)  # einfo + first batch succeed
    with pytest.raises(ResourceError):
        _client(tmp_path, session).esummary_snp(list(range(1, 401)))
    _client(tmp_path, FakeSession()).esummary_snp(list(range(1, 401)))
    keys = _cached_keys(tmp_path / "dbsnp_esummary.jsonl")
    rs_keys = [k for k in keys if k.startswith("dbsnp:rs")]
    assert len(rs_keys) == len(set(rs_keys)) == 400


def test_retries_429_with_backoff(tmp_path):
    slept: list[float] = []
    session = FakeSession(status_seq=[429, 429, 200])
    entry = _client(tmp_path, session, slept=slept).spdi_canonical(
        "NC_000010.11:94781858:G:A"
    )
    assert entry.status == "ok"
    assert slept == [1, 2]


def test_gives_up_after_four_retries(tmp_path):
    slept: list[float] = []
    session = FakeSession(status_seq=[503] * 5)
    with pytest.raises(ResourceError):
        _client(tmp_path, session, slept=slept).spdi_canonical(
            "NC_000010.11:94781858:G:A"
        )
    assert slept == [1, 2, 4, 8]


def test_bad_request_cached_as_error_not_retried(tmp_path):
    session = FakeSession()
    entry = _client(tmp_path, session).spdi_canonical("NC_000010.11:1:GG:")
    assert entry.status == "error"
    assert session.calls == 1


def test_offline_miss_raises(tmp_path):
    client = NCBIClient(NCBISettings(), tmp_path, offline=True)
    with pytest.raises(CacheMissError) as e:
        client.spdi_canonical("NC_000010.11:94781858:G:A")
    assert e.value.missing == ["spdi_canonical:NC_000010.11:94781858:G:A"]


def test_offline_serves_cache(tmp_path):
    _client(tmp_path, FakeSession()).spdi_canonical("NC_000010.11:94781858:G:A")
    entry = NCBIClient(NCBISettings(), tmp_path, offline=True).spdi_canonical(
        "NC_000010.11:94781858:G:A"
    )
    assert entry.response == {
        "seq_id": "NC_000010.11",
        "position": 94781858,
        "deleted_sequence": "G",
        "inserted_sequence": "A",
    }


def test_contextuals_keeps_ncbi_warnings(tmp_path):
    entry = _client(tmp_path, FakeSession()).hgvs_contextuals(
        "NC_000006.12:g.18149032_18149033insT"
    )
    assert entry.status == "ok"
    assert any("3'" in w for w in entry.warnings)


def test_api_key_never_in_cached_request(tmp_path):
    settings = NCBISettings(api_key=SecretStr("SECRET"))
    session = FakeSession()
    _client(tmp_path, session, settings=settings).esummary_snp([3892097])
    assert any("api_key=SECRET" in u for u in session.urls)
    assert "SECRET" not in (tmp_path / "dbsnp_esummary.jsonl").read_text()


def test_email_only_when_configured(tmp_path):
    session = FakeSession()
    _client(tmp_path, session).esummary_snp([3892097])
    assert all("email=" not in u for u in session.urls)
    assert all("tool=modephere" in u for u in session.urls if "eutils" in u)


# ----- final-review fixes ---------------------------------------------------- #


def test_overwrite_appends_and_reload_keeps_last_write(tmp_path):
    path = tmp_path / "x.jsonl"
    cache = RawCache(path)
    first = _entry("dbsnp:rs1").model_copy(update={"status": "not_found"})
    cache.put(first)
    second = _entry("dbsnp:rs1")
    cache.put(second, overwrite=True)
    assert len(path.read_text().splitlines()) == 2  # history kept, append-only
    reloaded = RawCache(path).get("dbsnp:rs1")
    assert reloaded is not None and reloaded.status == "ok"


def test_truncated_final_line_is_skipped(tmp_path, caplog):
    path = tmp_path / "x.jsonl"
    cache = RawCache(path)
    cache.put(_entry("k1"))
    with path.open("a") as f:
        f.write('{"key": "k2", "request": "u", "sta')  # killed mid-write
    reloaded = RawCache(path)
    assert reloaded.keys() == {"k1"}
    assert "truncated" in caplog.text


def test_corrupt_middle_line_is_an_error(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(
        "garbage\n" + json.dumps(_entry("k1").model_dump(mode="json")) + "\n"
    )
    with pytest.raises(DataError, match="x.jsonl:1"):
        RawCache(path)


def test_refresh_re_requests_failed_entries(tmp_path):
    _client(tmp_path, FakeSession()).esummary_snp([999999999999])
    session = FakeSession()
    ticks = itertools.count(0, 1000)
    client = NCBIClient(
        NCBISettings(),
        tmp_path,
        offline=False,
        transport=as_transport(session),
        sleep=lambda s: None,
        clock=lambda: float(next(ticks)),
        refresh=frozenset({"not_found", "error"}),
    )
    client.esummary_snp([999999999999, 3892097])
    assert any("999999999999" in u for u in session.urls if "esummary" in u)


def test_malformed_esummary_batch_caches_nothing(tmp_path):
    class _ErrorBody(FakeSession):
        def get(self, url, timeout):
            if "esummary" in url:
                self.calls += 1
                return FakeResponse(200, {"esummaryresult": ["Invalid uid"]})
            return super().get(url, timeout)

    with pytest.raises(ResourceError, match="malformed"):
        _client(tmp_path, _ErrorBody()).esummary_snp([3892097])
    keys = RawCache(tmp_path / "dbsnp_esummary.jsonl").keys()
    assert not any(k.startswith("dbsnp:rs") for k in keys)


def test_online_client_re_reads_dbsnp_build_each_run(tmp_path):
    _client(tmp_path, FakeSession()).esummary_snp([3892097])
    later = FakeSession()
    later_build = "Build260101-0000.1"
    original = later.get

    def get(url, timeout):
        if "einfo" in url:
            later.calls += 1
            return FakeResponse(
                200, {"einforesult": {"dbinfo": [{"dbbuild": later_build}]}}
            )
        return original(url, timeout)

    later.get = get  # type: ignore[method-assign]
    out = _client(tmp_path, later).esummary_snp([4244285])
    assert out[4244285].source_version == later_build
    # the earlier record keeps the build it was fetched under
    offline = NCBIClient(NCBISettings(), tmp_path, offline=True)
    assert (
        offline.esummary_snp([3892097])[3892097].source_version == "Build250306-1408.1"
    )
