"""PubChem PUG-REST client: cached, throttled, batched.

Every answer goes into a :class:`~modephere.library.ingest.raw_cache.RawCache`
first. With ``offline=True`` the client never touches the network: a lookup
whose response was never fetched comes back as ``status="missing"``, which is
how an offline build proves it used only recorded responses.

Etiquette:

* at most ``max_requests_per_second`` (PubChem publishes 5/s), stretched when
  the ``X-Throttling-Control`` header turns yellow / red / black;
* 429 and 5xx responses are retried with exponential backoff;
* names are POSTed (a name may contain ``/``), CIDs are batched
  ``batch_size`` per property request.

PubChem renamed its SMILES properties in 2025: ``IsomericSMILES`` /
``CanonicalSMILES`` are now ``SMILES`` / ``ConnectivitySMILES`` (the old names
silently return nothing), so only the new names are requested.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import ValidationError

from modephere.library.ingest.raw_cache import CacheEntry, RawCache
from modephere.types import FrozenModel, NameQuery
from modephere.types.scalars import InChIKey, NonEmptyStr, PubChemCID, SmilesStr

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from modephere.config import PubChemSettings

logger = logging.getLogger(__name__)

NAME_CACHE = "pubchem_names.jsonl"
PROPERTY_CACHE = "pubchem_properties.jsonl"
PROPERTIES: tuple[str, ...] = (
    "SMILES",
    "ConnectivitySMILES",
    "InChIKey",
    "IUPACName",
    "Title",
)
BACKOFF_S: tuple[int, ...] = (1, 2, 4, 8)
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_SLOWDOWN = {"green": 1.0, "yellow": 2.0, "red": 5.0, "black": 10.0}
_THROTTLE_STATUS = re.compile(r"status:\s*(\w+)", re.IGNORECASE)

type LookupStatus = Literal["ok", "not_found", "missing", "error"]


class PubChemRecord(FrozenModel):
    """The properties of one CID, as PubChem reports them."""

    cid: PubChemCID
    smiles: SmilesStr
    connectivity_smiles: SmilesStr | None = None
    inchikey: InChIKey | None = None
    iupac_name: NonEmptyStr | None = None
    title: NonEmptyStr | None = None


@dataclass(frozen=True, slots=True)
class Lookup[T]:
    """Outcome of one lookup.

    ``missing``: offline and never fetched. ``error``: the request failed
    after every retry (not cached, so the next run tries again).
    """

    status: LookupStatus
    value: T | None = None
    retrieved_at: datetime | None = None
    message: str | None = None


def throttle_factor(header: str | None) -> float:
    """Interval multiplier for an ``X-Throttling-Control`` header (worst status)."""
    if not header:
        return 1.0
    colours = [m.lower() for m in _THROTTLE_STATUS.findall(header)]
    return max((_SLOWDOWN.get(c, 1.0) for c in colours), default=1.0)


def _record(row: dict[str, Any]) -> PubChemRecord:
    return PubChemRecord(
        cid=row["CID"],
        smiles=row.get("SMILES"),
        connectivity_smiles=row.get("ConnectivitySMILES"),
        inchikey=row.get("InChIKey"),
        iupac_name=row.get("IUPACName"),
        title=row.get("Title"),
    )


class PubChemClient:
    """Name -> CIDs and CID -> properties, through the raw cache."""

    def __init__(
        self,
        settings: PubChemSettings,
        cache_dir: Path,
        *,
        offline: bool,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._offline = offline
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self._factor = 1.0
        self._http: httpx.Client | None = None
        if not offline:
            self._http = httpx.Client(
                base_url=settings.base_url,
                timeout=settings.timeout_s,
                headers={"User-Agent": settings.user_agent},
                transport=transport,
            )
        self.names = RawCache(cache_dir / NAME_CACHE)
        self.properties_cache = RawCache(cache_dir / PROPERTY_CACHE)

    @property
    def offline(self) -> bool:
        return self._offline

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def __enter__(self) -> PubChemClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- transport ------------------------------------------------------ #

    def _throttle(self) -> None:
        interval = self._factor / self._settings.max_requests_per_second
        if self._last_request is not None:
            wait = interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _post(self, path: str, data: dict[str, str]) -> httpx.Response:
        """POST with throttling and retry; raises ``httpx.HTTPError`` when exhausted."""
        assert self._http is not None  # online only
        error = ""
        for attempt in range(len(BACKOFF_S) + 1):
            self._throttle()
            try:
                response = self._http.post(path, data=data)
            except httpx.TransportError as e:
                error = f"{type(e).__name__}: {e}"
            else:
                self._factor = throttle_factor(
                    response.headers.get("X-Throttling-Control")
                )
                if self._factor > 1.0:
                    logger.info("PubChem throttling: slowing down x%.0f", self._factor)
                if response.status_code not in _RETRY_STATUS:
                    return response
                error = f"HTTP {response.status_code}"
            if attempt == len(BACKOFF_S):
                break
            delay = BACKOFF_S[attempt]
            logger.warning(
                "PubChem request failed (%s); retry in %ds: %s", error, delay, path
            )
            self._sleep(delay)
        msg = f"PubChem request failed after {len(BACKOFF_S)} retries: {path} ({error})"
        raise httpx.HTTPError(msg)

    # ----- name -> CIDs --------------------------------------------------- #

    def name_to_cids(self, name: str) -> Lookup[tuple[int, ...]]:
        """CIDs PubChem matches to ``name``, in PubChem's order."""
        return self._identifier_cids(NameQuery(name=name).key, "name", name)

    def inchikey_to_cids(self, inchikey: str) -> Lookup[tuple[int, ...]]:
        """CIDs whose standard InChIKey is ``inchikey`` (usually exactly one)."""
        inchikey = inchikey.strip().upper()
        return self._identifier_cids(f"inchikey:{inchikey}", "inchikey", inchikey)

    def _identifier_cids(
        self, key: str, namespace: str, value: str
    ) -> Lookup[tuple[int, ...]]:
        """``/compound/<namespace>/cids`` through the cache (names cache file)."""
        entry = self.names.get(key)
        if entry is None:
            if self._offline:
                return Lookup("missing", message=f"{key} not cached")
            try:
                entry = self._fetch_cids(key, namespace, value)
            except httpx.HTTPError as e:
                return Lookup("error", message=str(e))
            self.names.put(entry)
        if entry.status == "not_found" or entry.response is None:
            return Lookup("not_found", retrieved_at=entry.retrieved_at)
        cids = tuple(int(c) for c in entry.response.get("CID", []) if int(c) > 0)
        if not cids:
            return Lookup("not_found", retrieved_at=entry.retrieved_at)
        return Lookup("ok", cids, entry.retrieved_at)

    def _fetch_cids(self, key: str, namespace: str, value: str) -> CacheEntry:
        path = f"/compound/{namespace}/cids/JSON"
        response = self._post(path, {namespace: value})
        now = datetime.now(tz=UTC)
        request = f"POST {path} {namespace}={value}"
        if response.status_code == httpx.codes.NOT_FOUND:
            return CacheEntry(
                key=key, request=request, status="not_found", retrieved_at=now
            )
        response.raise_for_status()
        body = response.json().get("IdentifierList", {})
        return CacheEntry(
            key=key, request=request, status="ok", retrieved_at=now, response=body
        )

    # ----- CID -> properties ---------------------------------------------- #

    def properties(self, cids: Sequence[int]) -> dict[int, Lookup[PubChemRecord]]:
        """Properties of every CID; uncached CIDs are fetched in batches."""
        wanted = list(dict.fromkeys(cids))
        uncached = [c for c in wanted if f"cid:{c}" not in self.properties_cache]
        failed: dict[int, str] = {}
        if uncached and not self._offline:
            size = self._settings.batch_size
            for start in range(0, len(uncached), size):
                chunk = uncached[start : start + size]
                try:
                    for entry in self._fetch_properties(chunk):
                        self.properties_cache.put(entry)
                except httpx.HTTPError as e:
                    failed |= dict.fromkeys(chunk, str(e))
        return {c: self._property_lookup(c, failed.get(c)) for c in wanted}

    def _property_lookup(self, cid: int, error: str | None) -> Lookup[PubChemRecord]:
        if error is not None:
            return Lookup("error", message=error)
        entry = self.properties_cache.get(f"cid:{cid}")
        if entry is None:
            return Lookup("missing", message=f"cid:{cid} not cached")
        if entry.status == "not_found" or entry.response is None:
            return Lookup("not_found", retrieved_at=entry.retrieved_at)
        try:
            record = _record(entry.response)
        except (ValidationError, KeyError) as e:
            return Lookup("not_found", retrieved_at=entry.retrieved_at, message=str(e))
        return Lookup("ok", record, entry.retrieved_at)

    def _fetch_properties(self, cids: list[int]) -> list[CacheEntry]:
        path = f"/compound/cid/property/{','.join(PROPERTIES)}/JSON"
        joined = ",".join(map(str, cids))
        response = self._post(path, {"cid": joined})
        now = datetime.now(tz=UTC)
        request = f"POST {path} cid={joined}"
        rows: dict[int, dict[str, Any]] = {}
        if response.status_code != httpx.codes.NOT_FOUND:
            response.raise_for_status()
            table = response.json().get("PropertyTable", {}).get("Properties", [])
            rows = {int(r["CID"]): r for r in table}
        return [
            CacheEntry(
                key=f"cid:{c}",
                request=request,
                status="ok" if c in rows else "not_found",
                retrieved_at=now,
                response=rows.get(c),
            )
            for c in cids
        ]


__all__ = ["Lookup", "PubChemClient", "PubChemRecord", "throttle_factor"]
