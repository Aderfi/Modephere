"""Shared helpers for the pgx tests: a fake NCBI server over recorded fixtures.

The fakes keep dev_Pgen's ``get(url, timeout)`` shape; :func:`as_transport`
serves any of them to httpx as a ``MockTransport``.
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import httpx
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "pgx"


def fixture_name(key: str) -> str:
    return key.replace(":", "_").replace("/", "_") + ".json"


def _snv_identity(kind: str, expr: str) -> dict | None:
    """An SNV SPDI is its own canonical form (a 1-base substitution cannot shift)."""
    parts = expr.split(":")
    if kind != "canonical_representative" or len(parts) != 4:
        return None
    seq, pos, deleted, inserted = parts
    if len(deleted) != 1 or len(inserted) != 1:
        return None
    return {
        "data": {
            "seq_id": seq,
            "position": int(pos),
            "deleted_sequence": deleted,
            "inserted_sequence": inserted,
        }
    }


class FakeResponse:
    def __init__(
        self, status_code: int, payload: object = None, content: bytes = b""
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self) -> object:
        return self._payload


class FakeSession:
    """Serves recorded NCBI responses; counts calls; can fail on demand.

    ESummary requests for ids not in the fixture get a synthetic minimal
    record, so batching can be exercised with arbitrary id ranges.
    """

    def __init__(
        self, *, fail_after: int | None = None, status_seq: list[int] | None = None
    ) -> None:
        self.calls = 0
        self.urls: list[str] = []
        self.fail_after = fail_after
        self.status_seq = list(status_seq or [])
        self._esummary = json.loads((FIXTURES / "esummary_snp.json").read_text())[
            "result"
        ]

    def get(self, url: str, timeout: float) -> FakeResponse:  # noqa: ARG002, PLR0911
        self.calls += 1
        self.urls.append(url)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise httpx.ConnectError("simulated outage")
        if self.status_seq:
            status = self.status_seq.pop(0)
            if status != 200:
                return FakeResponse(status, {"error": "rate"})
        parsed = urllib.parse.urlparse(url)
        if parsed.path.endswith("einfo.fcgi"):
            return FakeResponse(
                200,
                {
                    "einforesult": {
                        "dbinfo": [
                            {
                                "dbbuild": "Build250306-1408.1",
                                "lastupdate": "2025/03/11 08:22",
                            }
                        ]
                    }
                },
            )
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("db") == ["gene"]:
            return _gene_response(parsed.path, query)
        if parsed.path.endswith("esummary.fcgi"):
            ids = urllib.parse.parse_qs(parsed.query)["id"][0].split(",")
            result: dict[str, object] = {"uids": ids}
            for i in ids:
                result[i] = self._esummary.get(
                    i, {"uid": i, "snp_id": int(i), "spdi": ""}
                )
            return FakeResponse(200, {"result": result})
        path = urllib.parse.unquote(parsed.path)
        for kind in ("canonical_representative", "hgvs", "contextuals"):
            if path.endswith("/" + kind):
                body = path.split("/variation/v0/")[1].rsplit("/", 1)[0]
                expr = body.split("/", 1)[1]
                folder = {
                    "canonical_representative": "spdi_canonical",
                    "hgvs": "spdi_hgvs",
                    "contextuals": "hgvs_contextuals",
                }[kind]
                file = FIXTURES / folder / fixture_name(expr)
                snv = _snv_identity(kind, expr)
                if not file.exists() and snv is not None:
                    return FakeResponse(200, snv)
                if not file.exists():
                    return FakeResponse(
                        400, {"error": {"message": f"no fixture {file.name}"}}
                    )
                return FakeResponse(200, json.loads(file.read_text()))
        return FakeResponse(404, {})


#: NCBI Gene ESummary records served by FakeSession (CYP2D6 only).
GENE_RECORDS: dict[str, dict[str, object]] = {
    "1565": {
        "uid": "1565",
        "name": "CYP2D6",
        "genomicinfo": [
            {"chraccver": "NC_000022.11", "chrstart": 42130809, "chrstop": 42126498}
        ],
        "locationhist": [
            {
                "assemblyaccver": "GCF_009914755.1",
                "chraccver": "NC_060946.1",
                "chrstart": 42610300,
                "chrstop": 42605989,
            }
        ],
    }
}


def _gene_response(path: str, query: dict[str, list[str]]) -> FakeResponse:
    if path.endswith("esearch.fcgi"):
        symbol = query["term"][0].split("[", 1)[0]
        ids = [uid for uid, r in GENE_RECORDS.items() if r["name"] == symbol]
        return FakeResponse(200, {"esearchresult": {"idlist": ids}})
    ids = query["id"][0].split(",")
    return FakeResponse(
        200, {"result": {"uids": ids, **{i: GENE_RECORDS[i] for i in ids}}}
    )


def as_transport(session: object) -> httpx.MockTransport:
    """Serve a fake ``session.get(url, timeout)`` through httpx."""

    def handler(request: httpx.Request) -> httpx.Response:
        r = session.get(str(request.url), timeout=30)  # type: ignore[attr-defined]
        if r._payload is not None:
            return httpx.Response(r.status_code, json=r._payload)
        return httpx.Response(r.status_code, content=r.content)

    return httpx.MockTransport(handler)


@pytest.fixture
def fake_session() -> FakeSession:
    return FakeSession()
