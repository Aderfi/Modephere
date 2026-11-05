"""Shared library fixtures: a fake PubChem, fast pipeline options, built compounds."""

from __future__ import annotations

import urllib.parse
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from modephere.config import PubChemSettings
from modephere.library.ingest.pubchem import PubChemClient
from modephere.library.pipeline import PipelineOptions, build_compound
from modephere.types import ConformerParams

# the miniature ChEMBL release (tests/unit/chembl/conftest.py)
from ..chembl.conftest import mini_db, staging

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.types import CompoundArtifacts

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
BUTANOL = "CC(O)CC"  # one unassigned stereocentre

RECORDS: dict[int, dict[str, Any]] = {
    2244: {"SMILES": ASPIRIN, "InChIKey": ASPIRIN_KEY, "Title": "Aspirin"},
    3672: {
        "SMILES": IBUPROFEN,
        "InChIKey": "HEFNNWSXXWATRW-UHFFFAOYSA-N",
        "Title": "Ibuprofen",
    },
    5793: {"SMILES": "C(C1C(C(C(C(O1)O)O)O)O)O", "Title": "D-Glucose"},
    79025: {"SMILES": "C(C1C(C(C(C(O1)O)O)O)O)O", "Title": "D-Glucopyranose"},
}
NAMES: dict[str, list[int]] = {
    "aspirin": [2244],
    "ibuprofen": [3672],
    "glucose": [5793, 79025],
}


class FakePubChem:
    """``httpx.MockTransport`` handler speaking the PUG-REST subset we use."""

    def __init__(
        self,
        *,
        fail_first: int = 0,
        status: int = 503,
        throttle: str = "Green",
        records: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.fail_remaining = fail_first
        self.status = status
        self.throttle = throttle
        self.records = RECORDS if records is None else records
        self.requests: list[httpx.Request] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        headers = {
            "X-Throttling-Control": (
                f"Request Count status: {self.throttle} (10%), "
                "Request Time status: Green (0%), Service status: Green (20%)"
            )
        }
        if self.fail_remaining:
            self.fail_remaining -= 1
            return httpx.Response(self.status, headers=headers)
        form = dict(urllib.parse.parse_qsl(request.content.decode()))
        path = request.url.path
        not_found = httpx.Response(404, json={"Fault": {"Code": "PUGREST.NotFound"}})
        if path.endswith("/cids/JSON"):  # name or InChIKey -> CIDs
            cids = self._identifier_cids(path, form)
            if not cids:
                return not_found
            return httpx.Response(
                200, json={"IdentifierList": {"CID": cids}}, headers=headers
            )
        if "/compound/cid/property/" in path:
            cids = [int(c) for c in form["cid"].split(",")]
            rows = [{"CID": c, **self.records[c]} for c in cids if c in self.records]
            if not rows:
                return not_found
            return httpx.Response(
                200, json={"PropertyTable": {"Properties": rows}}, headers=headers
            )
        return httpx.Response(400)

    def _identifier_cids(self, path: str, form: dict[str, str]) -> list[int]:
        if path.endswith("/compound/name/cids/JSON"):
            return NAMES.get(form["name"].strip().casefold(), [])
        if path.endswith("/compound/inchikey/cids/JSON"):
            return [
                c
                for c, rec in self.records.items()
                if rec.get("InChIKey") == form["inchikey"]
            ]
        return []


@pytest.fixture
def fake_pubchem() -> FakePubChem:
    return FakePubChem()


@pytest.fixture
def pubchem_settings() -> PubChemSettings:
    return PubChemSettings(batch_size=2)


class Recorder:
    """Injected ``sleep`` (records waits) and ``clock`` (fixed time)."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    @staticmethod
    def clock() -> float:
        return 0.0


def make_client(
    cache_dir: Path,
    settings: PubChemSettings,
    fake: FakePubChem | None = None,
    *,
    offline: bool = False,
    recorder: Recorder | None = None,
) -> PubChemClient:
    rec = recorder or Recorder()
    return PubChemClient(
        settings,
        cache_dir,
        offline=offline,
        transport=None if fake is None else fake.transport,
        sleep=rec.sleep,
        clock=rec.clock,
    )


#: Few conformers: fast, still exercises enumeration / optimisation / pruning.
FAST = PipelineOptions(
    conformers=ConformerParams(num_conformers=4, min_conformers=1, max_conformers=4)
)


@pytest.fixture(scope="session")
def aspirin() -> CompoundArtifacts:
    return build_compound(ASPIRIN, FAST)


@pytest.fixture(scope="session")
def butanol() -> CompoundArtifacts:
    return build_compound(BUTANOL, FAST)


__all__ = ["mini_db", "staging"]
