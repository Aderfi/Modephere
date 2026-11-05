"""Raw cache, PubChem client (fake transport), catalog readers and resolver."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest

from modephere.core import CompoundError, ConfigurationError, DataError
from modephere.library.ingest.catalog import load_catalog, queries_from_values
from modephere.library.ingest.pubchem import PROPERTIES, throttle_factor
from modephere.library.ingest.raw_cache import CacheEntry, RawCache
from modephere.library.ingest.resolve import Resolver
from modephere.types import (
    AmbiguityPolicy,
    CidQuery,
    FailureCategory,
    NameQuery,
    ResolutionSource,
    ResolvedCompound,
    SmilesQuery,
)

from ..conftest import ASPIRIN, ASPIRIN_KEY, FakePubChem, Recorder, make_client

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.config import PubChemSettings

NOW = datetime(2026, 11, 4, tzinfo=UTC)


def _entry(key: str, **kw: object) -> CacheEntry:
    return CacheEntry(key=key, request="POST x", status="ok", retrieved_at=NOW, **kw)


class TestRawCache:
    def test_round_trip_and_last_write_wins(self, tmp_path: Path) -> None:
        cache = RawCache(tmp_path / "c.jsonl")
        cache.put(_entry("a", response={"v": 1}))
        cache.put(_entry("a", response={"v": 2}))  # ignored without overwrite
        cache.put(_entry("a", response={"v": 3}), overwrite=True)
        reloaded = RawCache(tmp_path / "c.jsonl")
        entry = reloaded.get("a")
        assert entry is not None and entry.response == {"v": 3}
        assert len((tmp_path / "c.jsonl").read_text().splitlines()) == 2  # history kept

    def test_truncated_final_line_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        RawCache(path).put(_entry("a"))
        with path.open("a") as f:
            f.write('{"key": "b", "requ')
        cache = RawCache(path)
        assert "a" in cache and "b" not in cache
        cache.put(_entry("c"))  # appends on a fresh line
        assert "c" in RawCache(path)

    def test_corrupt_middle_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        path.write_text("garbage\n" + _entry("a").model_dump_json() + "\n")
        with pytest.raises(DataError, match="corrupt cache line"):
            RawCache(path)


class TestPubChemClient:
    def test_name_lookup_is_cached(
        self,
        tmp_path: Path,
        pubchem_settings: PubChemSettings,
        fake_pubchem: FakePubChem,
    ) -> None:
        client = make_client(tmp_path, pubchem_settings, fake_pubchem)
        assert client.name_to_cids("Aspirin").value == (2244,)
        assert client.name_to_cids("  aspirin ").value == (2244,)  # same key
        assert len(fake_pubchem.requests) == 1
        assert fake_pubchem.requests[0].method == "POST"

    def test_unknown_name_is_cached_not_found(
        self,
        tmp_path: Path,
        pubchem_settings: PubChemSettings,
        fake_pubchem: FakePubChem,
    ) -> None:
        client = make_client(tmp_path, pubchem_settings, fake_pubchem)
        assert client.name_to_cids("unobtainium").status == "not_found"
        offline = make_client(tmp_path, pubchem_settings, offline=True)
        assert offline.name_to_cids("unobtainium").status == "not_found"

    def test_properties_batched_and_per_cid_cached(
        self,
        tmp_path: Path,
        pubchem_settings: PubChemSettings,
        fake_pubchem: FakePubChem,
    ) -> None:
        client = make_client(tmp_path, pubchem_settings, fake_pubchem)
        out = client.properties([2244, 3672, 5793, 999])
        assert len(fake_pubchem.requests) == 2  # batch_size=2
        record = out[2244].value
        assert record is not None and record.inchikey == ASPIRIN_KEY
        assert out[999].status == "not_found"
        client.properties([2244, 999])
        assert len(fake_pubchem.requests) == 2  # all cached now
        path = fake_pubchem.requests[0].url.path
        assert path.endswith(f"/property/{','.join(PROPERTIES)}/JSON")
        assert "IsomericSMILES" not in path  # renamed by PubChem in 2025

    def test_retries_then_succeeds(
        self, tmp_path: Path, pubchem_settings: PubChemSettings
    ) -> None:
        fake, rec = FakePubChem(fail_first=2), Recorder()
        client = make_client(tmp_path, pubchem_settings, fake, recorder=rec)
        assert client.name_to_cids("aspirin").status == "ok"
        assert len(fake.requests) == 3
        assert {1, 2} <= set(rec.sleeps)  # backoff 1s, 2s

    def test_exhausted_retries_are_errors_not_cached(
        self, tmp_path: Path, pubchem_settings: PubChemSettings
    ) -> None:
        fake = FakePubChem(fail_first=100)
        client = make_client(tmp_path, pubchem_settings, fake)
        assert client.name_to_cids("aspirin").status == "error"
        assert client.properties([2244])[2244].status == "error"
        assert "name:aspirin" not in client.names  # next run retries

    def test_throttle_header_slows_requests(
        self, tmp_path: Path, pubchem_settings: PubChemSettings
    ) -> None:
        fake, rec = FakePubChem(throttle="Red"), Recorder()
        client = make_client(tmp_path, pubchem_settings, fake, recorder=rec)
        client.name_to_cids("aspirin")
        client.name_to_cids("ibuprofen")
        assert rec.sleeps[-1] == pytest.approx(
            5.0 / pubchem_settings.max_requests_per_second
        )

    @pytest.mark.parametrize(
        ("header", "factor"),
        [(None, 1.0), ("Request Count status: Green (0%)", 1.0),
         ("Request Count status: Yellow (60%), Service status: Black (99%)", 10.0)],
    )  # fmt: skip
    def test_throttle_factor(self, header: str | None, factor: float) -> None:
        assert throttle_factor(header) == factor

    def test_offline_misses_are_reported(
        self, tmp_path: Path, pubchem_settings: PubChemSettings
    ) -> None:
        client = make_client(tmp_path, pubchem_settings, offline=True)
        assert client.name_to_cids("aspirin").status == "missing"
        assert client.properties([2244])[2244].status == "missing"

    def test_offline_replays_the_cache(
        self,
        tmp_path: Path,
        pubchem_settings: PubChemSettings,
        fake_pubchem: FakePubChem,
    ) -> None:
        make_client(tmp_path, pubchem_settings, fake_pubchem).properties([2244])
        offline = make_client(tmp_path, pubchem_settings, offline=True)
        record = offline.properties([2244])[2244].value
        assert record is not None and record.smiles == ASPIRIN


class TestResolver:
    def _resolver(
        self, tmp_path: Path, settings: PubChemSettings, fake: FakePubChem, **kw: object
    ) -> Resolver:
        return Resolver(make_client(tmp_path, settings, fake), **kw)  # type: ignore[arg-type]

    def test_smiles_is_local_without_client(self) -> None:
        r = Resolver(None).resolve(SmilesQuery(smiles=ASPIRIN))
        assert r.source is ResolutionSource.LOCAL and r.cid is None

    def test_remote_query_without_client(self) -> None:
        with pytest.raises(ConfigurationError):
            Resolver(None).resolve(CidQuery(cid=2244))

    def test_mixed_batch(
        self,
        tmp_path: Path,
        pubchem_settings: PubChemSettings,
        fake_pubchem: FakePubChem,
    ) -> None:
        resolver = self._resolver(tmp_path, pubchem_settings, fake_pubchem)
        out = resolver.resolve_many(
            [
                CidQuery(cid=2244),
                NameQuery(name="ibuprofen"),
                SmilesQuery(smiles="CCO"),
                NameQuery(name="unobtainium"),
                CidQuery(cid=999),
            ]
        )
        first, second, third = (r for r in out[:3] if isinstance(r, ResolvedCompound))
        assert first.title == "Aspirin"
        assert second.cid == 3672
        assert third.source is ResolutionSource.LOCAL
        for failure in out[3:]:
            assert isinstance(failure, CompoundError)
            assert failure.category is FailureCategory.UNRESOLVED_IDENTIFIER

    def test_ambiguous_first_hit(
        self,
        tmp_path: Path,
        pubchem_settings: PubChemSettings,
        fake_pubchem: FakePubChem,
    ) -> None:
        r = self._resolver(tmp_path, pubchem_settings, fake_pubchem).resolve(
            NameQuery(name="glucose")
        )
        assert r.cid == 5793 and r.is_ambiguous
        assert "first" in r.warnings[0]

    def test_ambiguous_fail(
        self,
        tmp_path: Path,
        pubchem_settings: PubChemSettings,
        fake_pubchem: FakePubChem,
    ) -> None:
        resolver = self._resolver(
            tmp_path, pubchem_settings, fake_pubchem, ambiguity=AmbiguityPolicy.FAIL
        )
        with pytest.raises(CompoundError) as exc:
            resolver.resolve(NameQuery(name="glucose"))
        assert exc.value.category is FailureCategory.AMBIGUOUS_IDENTIFIER

    def test_offline_miss_is_cache_miss(
        self, tmp_path: Path, pubchem_settings: PubChemSettings
    ) -> None:
        resolver = Resolver(make_client(tmp_path, pubchem_settings, offline=True))
        [result] = resolver.resolve_many([NameQuery(name="aspirin")])
        assert isinstance(result, CompoundError)
        assert result.category is FailureCategory.CACHE_MISS

    def test_inconsistent_record_is_a_failure(
        self, tmp_path: Path, pubchem_settings: PubChemSettings
    ) -> None:
        fake = FakePubChem(records={2244: {"SMILES": "", "Title": "broken"}})
        [result] = self._resolver(tmp_path, pubchem_settings, fake).resolve_many(
            [CidQuery(cid=2244)]
        )
        assert isinstance(result, CompoundError)


class TestCatalog:
    def test_dev_pgen_tsv(self, tmp_path: Path) -> None:
        path = tmp_path / "drugs.tsv"
        path.write_text(
            "cid\tsmiles\tcmpd_name_cleaned\n"
            f"2244\t{ASPIRIN}\taspirin\n"  # smiles wins: no network needed
            "3672\t\tibuprofen\n"  # cid next
            "\t\tcaffeine\n"  # name last
            "abc\t\t\n"  # invalid cid
            "\t\t\n"  # empty row
        )
        load = load_catalog(path)
        kinds = [q.kind.value for q in load.queries]
        assert kinds == ["smiles", "cid", "name"]
        assert load.queries[0].label == "aspirin"
        assert [f.key for f in load.failures] == ["row:4", "row:5"]

    def test_explicit_kind_value_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "q.csv"
        path.write_text("kind,value,label\ncid,2244,a\nname,aspirin,b\ninchi,x,c\n")
        load = load_catalog(path)
        assert [q.key for q in load.queries] == ["cid:2244", "name:aspirin"]
        assert load.failures[0].key == "row:3"

    def test_parquet(self, tmp_path: Path) -> None:
        path = tmp_path / "q.parquet"
        pl.DataFrame({"cid": [2244, 3672]}).write_parquet(path)
        assert [q.key for q in load_catalog(path).queries] == ["cid:2244", "cid:3672"]

    def test_json_forms(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        a.write_text(json.dumps([{"kind": "cid", "cid": 2244}, {"kind": "bad"}]))
        load = load_catalog(a)
        assert len(load.queries) == 1 and len(load.failures) == 1
        b = tmp_path / "b.json"
        b.write_text(json.dumps({" 2244": ASPIRIN}))  # dev_Pgen mapping
        [q] = load_catalog(b).queries
        assert isinstance(q, SmilesQuery) and q.label == "2244"

    def test_bad_files(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="not found"):
            load_catalog(tmp_path / "missing.tsv")
        (tmp_path / "x.xlsx").write_text("")
        with pytest.raises(DataError, match="unsupported"):
            load_catalog(tmp_path / "x.xlsx")
        (tmp_path / "cols.tsv").write_text("foo\tbar\n1\t2\n")
        with pytest.raises(DataError, match="expected"):
            load_catalog(tmp_path / "cols.tsv")

    def test_values(self) -> None:
        load = queries_from_values(
            smiles=["CCO"], cids=["2244", "x"], names=["aspirin"]
        )
        assert [q.kind.value for q in load.queries] == ["smiles", "cid", "name"]
        assert len(load.failures) == 1
