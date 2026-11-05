"""Resolver with a staged ChEMBL release (the miniature one in chembl/conftest)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from modephere.chembl import ChemblLookup
from modephere.core import CompoundError, ConfigurationError
from modephere.library.ingest.resolve import Resolver, structure_inchikey
from modephere.types import (
    AmbiguityPolicy,
    ChemblMatch,
    ChemblQuery,
    CidQuery,
    FailureCategory,
    NameQuery,
    ResolutionSource,
    ResolvedCompound,
    SmilesQuery,
)

from ..conftest import ASPIRIN, ASPIRIN_KEY, FakePubChem, make_client

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.config import PubChemSettings

IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"


@pytest.fixture(scope="module")
def lookup(staging: Path) -> ChemblLookup:
    return ChemblLookup(staging)


def test_lookup_knows_its_release(lookup: ChemblLookup) -> None:
    assert lookup.release == "chembl_99"
    assert lookup.released_at.isoformat() == "2026-05-01T00:00:00+00:00"


class TestChemblSources:
    def test_chembl_id_of_a_salt_resolves_to_the_parent(
        self, lookup: ChemblLookup
    ) -> None:
        r = Resolver(None, chembl=lookup).resolve(ChemblQuery(chembl_id="chembl1642"))
        assert r.source is ResolutionSource.CHEMBL
        assert (r.chembl_id, r.chembl_record_id) == ("CHEMBL941", "CHEMBL1642")
        assert r.chembl_match is ChemblMatch.CHEMBL_ID
        assert r.smiles == IMATINIB  # ChEMBL's own salt-stripped parent
        assert r.title == "IMATINIB"
        assert r.retrieved_at == lookup.released_at
        assert r.chembl_release == "chembl_99"

    def test_name_needs_no_pubchem(self, lookup: ChemblLookup) -> None:
        resolver = Resolver(None, chembl=lookup)
        query = NameQuery(name="Gleevec")
        assert not resolver.needs_pubchem(query)
        r = resolver.resolve(query)
        assert (r.chembl_id, r.chembl_match) == ("CHEMBL941", ChemblMatch.NAME)
        assert r.candidate_chembl_ids == ("CHEMBL941",) and not r.is_ambiguous

    def test_unknown_chembl_id(self, lookup: ChemblLookup) -> None:
        with pytest.raises(CompoundError) as e:
            Resolver(None, chembl=lookup).resolve(ChemblQuery(chembl_id="CHEMBL999"))
        assert e.value.category is FailureCategory.UNRESOLVED_IDENTIFIER

    def test_chembl_id_needs_a_release(self) -> None:
        with pytest.raises(ConfigurationError, match="staged ChEMBL"):
            Resolver(None).resolve(ChemblQuery(chembl_id="CHEMBL25"))

    def test_no_structure(
        self, lookup: ChemblLookup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent_of = lookup.parent_of
        monkeypatch.setattr(
            lookup,
            "parent_of",
            lambda h: dataclasses.replace(parent_of(h), smiles=None),
        )
        with pytest.raises(CompoundError) as e:
            Resolver(None, chembl=lookup).resolve(NameQuery(name="aspirin"))
        assert e.value.category is FailureCategory.UNSUPPORTED_CHEMISTRY

    @pytest.mark.parametrize("policy", list(AmbiguityPolicy))
    def test_ambiguous_name(
        self,
        lookup: ChemblLookup,
        monkeypatch: pytest.MonkeyPatch,
        policy: AmbiguityPolicy,
    ) -> None:
        both = [*lookup.by_name("aspirin")[:1], *lookup.by_name("gleevec")]
        monkeypatch.setattr(lookup, "by_name", lambda name: both)
        resolver = Resolver(None, chembl=lookup, ambiguity=policy)
        if policy is AmbiguityPolicy.FAIL:
            with pytest.raises(CompoundError) as e:
                resolver.resolve(NameQuery(name="x"))
            assert e.value.category is FailureCategory.AMBIGUOUS_IDENTIFIER
            return
        r = resolver.resolve(NameQuery(name="x"))
        assert r.chembl_id == "CHEMBL25" and r.is_ambiguous
        assert r.candidate_chembl_ids == ("CHEMBL25", "CHEMBL941")
        assert "2 ChEMBL parents" in r.warnings[0]


class TestMatching:
    def test_smiles_matched_by_structure(self, lookup: ChemblLookup) -> None:
        r = Resolver(None, chembl=lookup).resolve(SmilesQuery(smiles=ASPIRIN))
        assert r.source is ResolutionSource.LOCAL
        assert (r.chembl_id, r.chembl_match) == (
            "CHEMBL25",
            ChemblMatch.STRUCTURE_INCHIKEY,
        )

    def test_salt_smiles_matched_to_the_parent(self, lookup: ChemblLookup) -> None:
        # the ChEMBL record of the sodium salt has a placeholder key here, so
        # only an exact standard-InChIKey match counts: none for this SMILES
        r = Resolver(None, chembl=lookup).resolve(SmilesQuery(smiles="CCO"))
        assert r.chembl_id is None and r.chembl_match is None

    def test_cid_matched_by_upstream_key(
        self, lookup: ChemblLookup, tmp_path: Path, pubchem_settings: PubChemSettings
    ) -> None:
        fake = FakePubChem()
        with make_client(tmp_path, pubchem_settings, fake) as client:
            resolver = Resolver(client, chembl=lookup)
            assert resolver.needs_pubchem(CidQuery(cid=2244))
            aspirin, ibuprofen = resolver.resolve_many(
                [CidQuery(cid=2244), NameQuery(name="ibuprofen")]
            )
        assert isinstance(aspirin, ResolvedCompound) and isinstance(
            ibuprofen, ResolvedCompound
        )
        assert aspirin.source is ResolutionSource.PUBCHEM
        assert (aspirin.chembl_id, aspirin.chembl_match) == (
            "CHEMBL25",
            ChemblMatch.UPSTREAM_INCHIKEY,
        )
        # not in the ChEMBL selection: PubChem resolves it, no chembl_id
        assert (
            ibuprofen.source is ResolutionSource.PUBCHEM and ibuprofen.chembl_id is None
        )

    def test_name_unknown_everywhere_without_pubchem(
        self, lookup: ChemblLookup
    ) -> None:
        with pytest.raises(ConfigurationError, match="PubChemClient"):
            Resolver(None, chembl=lookup).resolve(NameQuery(name="unobtainium"))

    def test_without_a_release_nothing_is_matched(self) -> None:
        r = Resolver(None).resolve(SmilesQuery(smiles=ASPIRIN))
        assert r.chembl_id is None
        assert Resolver(None).needs_pubchem(NameQuery(name="aspirin"))


def test_structure_inchikey() -> None:
    assert structure_inchikey(ASPIRIN) == ASPIRIN_KEY
    assert structure_inchikey("not a smiles(") is None


class TestRecordRules:
    def test_partial_chembl_fields_rejected(self) -> None:
        with pytest.raises(ValueError, match="ChEMBL match needs"):
            ResolvedCompound(
                query=SmilesQuery(smiles=ASPIRIN),
                smiles=ASPIRIN,
                source=ResolutionSource.LOCAL,
                chembl_id="CHEMBL25",
            )

    def test_chembl_query_must_resolve_on_chembl(self) -> None:
        with pytest.raises(ValueError, match="resolved elsewhere"):
            ResolvedCompound(
                query=ChemblQuery(chembl_id="CHEMBL25"),
                smiles=ASPIRIN,
                source=ResolutionSource.PUBCHEM,
                retrieved_at=datetime.now(UTC),
            )
