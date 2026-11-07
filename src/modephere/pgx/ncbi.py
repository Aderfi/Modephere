"""NCBI client: dbSNP ESummary, Gene, Variation Services; cached, rate-limited.

All responses go through :class:`~modephere.library.ingest.raw_cache.RawCache`
first. With ``offline=True`` the client never touches the network and a cache
miss is a :class:`~modephere.pgx.cache.CacheMissError` — that is how ``ingest``
guarantees it builds only from recorded responses. Ported from dev_Pgen's
``src.kg.ncbi`` (same cache keys and files, so its raw cache replays here).

Etiquette (NCBI usage policy): ``tool`` is always sent, ``email`` only when
configured; E-utilities at most 3 requests/s (10 with an API key);
Variation Services at most 1 request/s. 429 and 5xx responses are retried
with exponential backoff; other 4xx responses are cached as ``error``.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from modephere.core import ResourceError
from modephere.library.ingest.raw_cache import CacheEntry, RawCache
from modephere.pgx.cache import CacheMissError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from modephere.config import NCBISettings

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
VARIATION = "https://api.ncbi.nlm.nih.gov/variation/v0"
ESUMMARY_BATCH = 200
BACKOFF_S: tuple[int, ...] = (1, 2, 4, 8)
_RETRY_STATUS = {429, 500, 502, 503, 504}

DBSNP_CACHE = "dbsnp_esummary.jsonl"
SPDI_CACHE = "ncbi_spdi.jsonl"
GENE_CACHE = "ncbi_gene.jsonl"
EINFO_KEY = "dbsnp:einfo"


class NCBIClient:
    def __init__(
        self,
        settings: NCBISettings,
        cache_dir: Path,
        *,
        offline: bool,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        refresh: frozenset[str] = frozenset(),
    ) -> None:
        self._settings = settings
        self._offline = offline
        self._http: httpx.Client | None = None
        if not offline:
            self._http = httpx.Client(
                timeout=settings.timeout_s,
                transport=transport,
                follow_redirects=True,
            )
        self._sleep = sleep
        self._clock = clock
        self._last_request: dict[str, float] = {}
        # Online only: cached entries with these statuses are re-requested.
        self._refresh = frozenset() if offline else refresh
        self._build: str | None = None
        self.dbsnp_cache = RawCache(cache_dir / DBSNP_CACHE)
        self.spdi_cache = RawCache(cache_dir / SPDI_CACHE)
        self.gene_cache = RawCache(cache_dir / GENE_CACHE)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def __enter__(self) -> NCBIClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- transport ------------------------------------------------------ #

    def _min_interval(self, host: str) -> float:
        if host.startswith("eutils"):
            return 0.1 if self._settings.api_key else 1 / 3
        return 1.0

    def _throttle(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            wait = self._min_interval(host) - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._clock()

    def _get(
        self, url: str, secret_params: dict[str, str] | None = None
    ) -> httpx.Response:
        """GET with throttling and retry; raises ResourceError when exhausted."""
        if self._http is None:
            msg = "NCBIClient is offline"
            raise ResourceError(msg)
        full = url
        if secret_params:
            full += ("&" if "?" in url else "?") + urllib.parse.urlencode(secret_params)
        host = urllib.parse.urlparse(url).netloc
        for attempt in range(len(BACKOFF_S) + 1):
            self._throttle(host)
            try:
                response = self._http.get(full)
            except httpx.TransportError as e:
                error: str = f"{type(e).__name__}: {e}"
            else:
                if response.status_code not in _RETRY_STATUS:
                    return response
                error = f"HTTP {response.status_code}"
            if attempt == len(BACKOFF_S):
                break
            delay = BACKOFF_S[attempt]
            logger.warning(
                "NCBI request failed (%s); retrying in %ds: %s", error, delay, url
            )
            self._sleep(delay)
        msg = f"NCBI request failed after {len(BACKOFF_S)} retries: {url} ({error})"
        raise ResourceError(msg)

    def _eutils_url(self, endpoint: str, params: dict[str, str]) -> str:
        public = {**params, "retmode": "json", "tool": self._settings.tool}
        if self._settings.email:
            public["email"] = self._settings.email
        return f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(public)}"

    def _eutils_secret(self) -> dict[str, str] | None:
        key = self._settings.api_key
        return {"api_key": key.get_secret_value()} if key else None

    def _cached(self, cache: RawCache, key: str) -> CacheEntry | None:
        entry = cache.get(key)
        if entry is not None and entry.status in self._refresh:
            return None
        return entry

    def _require_online(self, missing: list[str]) -> None:
        if self._offline and missing:
            raise CacheMissError(missing)

    # ----- dbSNP ---------------------------------------------------------- #

    def dbsnp_build(self) -> str:
        """Current dbSNP build: re-read once per online client, cached for offline use."""
        if self._build is not None:
            return self._build
        if self._offline:
            cached = self.dbsnp_cache.get(EINFO_KEY)
            if cached is None or cached.source_version is None:
                raise CacheMissError([EINFO_KEY])
            self._build = cached.source_version
            return self._build
        url = self._eutils_url("einfo.fcgi", {"db": "snp"})
        payload = self._get(url, self._eutils_secret()).json()
        try:
            build = str(payload["einforesult"]["dbinfo"][0]["dbbuild"])
        except (KeyError, IndexError, TypeError) as e:
            msg = f"malformed dbSNP einfo response: {url}"
            raise ResourceError(msg) from e
        entry = CacheEntry(
            key=EINFO_KEY,
            request=url,
            status="ok",
            retrieved_at=datetime.now(UTC),
            response=payload,
            source_version=build,
        )
        self.dbsnp_cache.put(entry, overwrite=True)
        self._build = build
        return build

    def esummary_snp(self, rsids: Sequence[int]) -> dict[int, CacheEntry]:
        """ESummary records keyed by the *requested* rsID number."""
        keys = {i: f"dbsnp:rs{i}" for i in rsids}
        missing = [
            i for i, k in keys.items() if self._cached(self.dbsnp_cache, k) is None
        ]
        self._require_online([keys[i] for i in missing])
        if missing:
            build = self.dbsnp_build()
            for start in range(0, len(missing), ESUMMARY_BATCH):
                self._fetch_esummary_batch(
                    missing[start : start + ESUMMARY_BATCH], build
                )
        out: dict[int, CacheEntry] = {}
        for i, key in keys.items():
            entry = self.dbsnp_cache.get(key)
            assert entry is not None
            out[i] = entry
        return out

    def _fetch_esummary_batch(self, batch: list[int], build: str) -> None:
        url = self._eutils_url(
            "esummary.fcgi", {"db": "snp", "id": ",".join(map(str, batch))}
        )
        payload = self._get(url, self._eutils_secret()).json()
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or not isinstance(result.get("uids"), list):
            # An E-utilities error body: never record it as per-rsID not_found.
            msg = f"malformed ESummary response (no result.uids): {url}"
            raise ResourceError(msg)
        now = datetime.now(UTC)
        for i in batch:
            record = result.get(str(i))
            found = record is not None and "error" not in record
            self.dbsnp_cache.put(
                CacheEntry(
                    key=f"dbsnp:rs{i}",
                    request=url,
                    status="ok" if found else "not_found",
                    retrieved_at=now,
                    response=record,
                    source_version=build,
                ),
                overwrite=True,
            )
        logger.info("dbSNP ESummary: cached %d records", len(batch))

    # ----- Gene ----------------------------------------------------------- #

    def gene_summary(self, symbol: str) -> CacheEntry:
        """The human NCBI Gene record whose official symbol is ``symbol``.

        ``response`` is ``{"gene_id", "record"}`` (the ESummary document, with
        its ``genomicinfo`` / ``locationhist`` placements); ``not_found`` when
        no current human gene has that symbol.
        """
        key = f"gene:{symbol}"
        cached = self._cached(self.gene_cache, key)
        if cached is not None:
            return cached
        self._require_online([key])
        term = f"{symbol}[sym] AND 9606[taxid]"
        search = self._eutils_url("esearch.fcgi", {"db": "gene", "term": term})
        ids = (
            self._get(search, self._eutils_secret())
            .json()
            .get("esearchresult", {})
            .get("idlist", [])
        )
        record: dict | None = None
        url = search
        if ids:
            url = self._eutils_url("esummary.fcgi", {"db": "gene", "id": ",".join(ids)})
            result = self._get(url, self._eutils_secret()).json().get("result", {})
            matches = [
                result[i]
                for i in result.get("uids", [])
                if isinstance(result.get(i), dict) and result[i].get("name") == symbol
            ]
            record = min(matches, key=lambda r: int(r["uid"])) if matches else None
        entry = CacheEntry(
            key=key,
            request=url,
            status="ok" if record is not None else "not_found",
            retrieved_at=datetime.now(UTC),
            response={"gene_id": int(record["uid"]), "record": record}
            if record is not None
            else None,
            source_version="gene-esummary",
        )
        self.gene_cache.put(entry, overwrite=True)
        return entry

    # ----- Variation Services --------------------------------------------- #

    def _variation(self, key: str, path: str) -> CacheEntry:
        cached = self._cached(self.spdi_cache, key)
        if cached is not None:
            return cached
        self._require_online([key])
        url = f"{VARIATION}/{path}"
        response = self._get(url)
        try:
            payload = response.json()
        except ValueError:  # non-JSON error body (e.g. an HTML 404 page)
            payload = {}
        if not isinstance(payload, dict):
            payload = {"body": payload}
        data = payload.get("data")
        ok = response.status_code == 200 and isinstance(data, dict)
        warnings = [w.get("message", "") for w in (data or {}).get("warnings", [])]
        entry = CacheEntry(
            key=key,
            request=url,
            status="ok" if ok else "error",
            retrieved_at=datetime.now(UTC),
            response=data if ok and isinstance(data, dict) else payload,
            warnings=warnings,
            source_version="variation/v0",
        )
        self.spdi_cache.put(entry, overwrite=True)
        return entry

    def spdi_canonical(self, spdi: str) -> CacheEntry:
        return self._variation(
            f"spdi_canonical:{spdi}", f"spdi/{spdi}/canonical_representative"
        )

    def spdi_hgvs(self, spdi: str) -> CacheEntry:
        return self._variation(f"spdi_hgvs:{spdi}", f"spdi/{spdi}/hgvs")

    def hgvs_contextuals(self, hgvs: str) -> CacheEntry:
        quoted = urllib.parse.quote(hgvs, safe="")
        return self._variation(f"hgvs_contextuals:{hgvs}", f"hgvs/{quoted}/contextuals")


__all__ = ["NCBIClient"]
