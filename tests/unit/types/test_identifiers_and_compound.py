"""Queries (discriminated union) and resolution / standardization records."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modephere.types import (
    COMPOUND_QUERY,
    CidQuery,
    IdentifierKind,
    NameQuery,
    ResolutionSource,
    ResolvedCompound,
    SmilesQuery,
    StandardizedCompound,
    parse_query,
)

from .conftest import ASPIRIN, ASPIRIN_KEY, NOW


class TestQueries:
    @pytest.mark.parametrize(
        ("payload", "cls", "key"),
        [
            (
                {"kind": "smiles", "smiles": f" {ASPIRIN} "},
                SmilesQuery,
                f"smiles:{ASPIRIN}",
            ),
            ({"kind": "cid", "cid": " 2244 "}, CidQuery, "cid:2244"),
            ({"kind": "name", "name": "  Aspirin "}, NameQuery, "name:aspirin"),
        ],
    )
    def test_discriminated_parse(self, payload: dict, cls: type, key: str) -> None:
        query = COMPOUND_QUERY.validate_python(payload)
        assert isinstance(query, cls)
        assert query.key == key

    def test_name_key_folds_case_and_spacing(self) -> None:
        a = NameQuery(name="Acetylsalicylic   ACID")
        b = NameQuery(name="acetylsalicylic acid")
        assert a.key == b.key
        assert a.name == "Acetylsalicylic   ACID"  # the original is kept

    def test_parse_query_from_pair(self) -> None:
        q = parse_query("cid", "2244", label="row-7")
        assert q == CidQuery(cid=2244, label="row-7")
        assert parse_query(IdentifierKind.NAME, "aspirin").key == "name:aspirin"

    @pytest.mark.parametrize(
        ("kind", "value"),
        [("inchi", "InChI=1S/..."), ("cid", "abc"), ("cid", 0), ("name", "  ")],
    )
    def test_parse_query_rejects(self, kind: str, value: object) -> None:
        with pytest.raises(ValidationError):
            parse_query(kind, value)

    def test_queries_are_hashable(self) -> None:
        assert len({CidQuery(cid=1), CidQuery(cid=1), CidQuery(cid=2)}) == 2

    def test_json_round_trip(self) -> None:
        q = NameQuery(name="caffeine", label="x")
        assert COMPOUND_QUERY.validate_json(COMPOUND_QUERY.dump_json(q)) == q


class TestResolvedCompound:
    def test_local_needs_smiles_query(self) -> None:
        with pytest.raises(ValidationError, match="resolves locally"):
            ResolvedCompound(
                query=CidQuery(cid=2244),
                smiles=ASPIRIN,
                source=ResolutionSource.LOCAL,
                cid=2244,
            )

    def test_remote_needs_retrieval_time(self) -> None:
        with pytest.raises(ValidationError, match="retrieved_at"):
            ResolvedCompound(
                query=NameQuery(name="aspirin"),
                smiles=ASPIRIN,
                source=ResolutionSource.PUBCHEM,
                cid=2244,
            )

    def test_cid_query_must_keep_its_cid(self) -> None:
        with pytest.raises(ValidationError, match="resolved to CID"):
            ResolvedCompound(
                query=CidQuery(cid=2244),
                smiles=ASPIRIN,
                source=ResolutionSource.PUBCHEM,
                cid=1,
                retrieved_at=NOW,
            )

    def test_ambiguous_name(self) -> None:
        r = ResolvedCompound(
            query=NameQuery(name="glucose"),
            smiles="OCC1OC(O)C(O)C(O)C1O",
            source=ResolutionSource.PUBCHEM,
            cid=5793,
            candidate_cids=(5793, 79025),
            retrieved_at=NOW,
        )
        assert r.is_ambiguous

    @pytest.mark.parametrize(
        ("cid", "candidates", "match"),
        [(1, (5793, 79025), "not among"), (5793, (5793, 5793), "duplicate")],
    )
    def test_candidate_consistency(
        self, cid: int, candidates: tuple[int, ...], match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            ResolvedCompound(
                query=NameQuery(name="glucose"),
                smiles="OCC1OC(O)C(O)C(O)C1O",
                source=ResolutionSource.PUBCHEM,
                cid=cid,
                candidate_cids=candidates,
                retrieved_at=NOW,
            )

    def test_naive_datetime_rejected(self, resolved_pubchem: ResolvedCompound) -> None:
        with pytest.raises(ValidationError):
            ResolvedCompound.model_validate(
                {
                    **resolved_pubchem.model_dump(),
                    "retrieved_at": NOW.replace(tzinfo=None),
                }
            )

    def test_json_round_trip(self, resolved_pubchem: ResolvedCompound) -> None:
        dumped = resolved_pubchem.model_dump_json()
        assert ResolvedCompound.model_validate_json(dumped) == resolved_pubchem


class TestStandardizedCompound:
    def test_key_is_registration_inchikey(
        self, standardized: StandardizedCompound
    ) -> None:
        assert standardized.key == ASPIRIN_KEY

    def test_json_round_trip(self, standardized: StandardizedCompound) -> None:
        dumped = standardized.model_dump_json()
        assert StandardizedCompound.model_validate_json(dumped) == standardized

    def test_negative_fragment_count_rejected(
        self, standardized: StandardizedCompound
    ) -> None:
        with pytest.raises(ValidationError):
            StandardizedCompound.model_validate(
                {**standardized.model_dump(), "fragments_removed": -1}
            )
