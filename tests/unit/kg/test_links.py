"""Library and drug links, against the recording fake driver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from modephere.chembl import ChemblLookup
from modephere.config import PubChemSettings
from modephere.core import DataError
from modephere.kg import KnowledgeGraph
from modephere.kg.links import (
    drugs_from_file,
    drugs_from_graph,
    is_class_or_mixture,
    library_dataset,
    library_rows,
    link_drugs,
    link_library,
    match_drugs,
)
from modephere.library.builder import LibraryBuilder
from modephere.library.config import LibraryBuildConfig
from modephere.library.pipeline import PipelineOptions

from ..library.conftest import ASPIRIN, FakePubChem, make_client
from .test_graph import FakeDriver

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def lookup(staging: Path) -> ChemblLookup:
    return ChemblLookup(staging)


@pytest.fixture(scope="module")
def library(staging: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("lib")
    cfg = LibraryBuildConfig(
        library_root=root / "library",
        cache_dir=root / "cache",
        chembl_staging=staging,
        names=("gleevec",),
        smiles=(ASPIRIN, "CCCCO"),
        pipeline=PipelineOptions(build_3d=False),
    )
    assert LibraryBuilder(cfg).run().built == 3
    return cfg.library_root


class TestLibrary:
    def test_rows(self, library: Path) -> None:
        nodes, edges = library_rows(library)
        assert len(nodes) == 3
        assert {n["props"]["num_conformers"] for n in nodes} == {0}
        ((release, rows),) = edges
        assert release == "chembl_99"
        assert {(r["start"][:4], r["end"]) for r in rows} == {
            ("KTUF", "CHEMBL941"),
            ("BSYN", "CHEMBL25"),
        }
        assert {r["props"]["match"] for r in rows} == {"name", "structure_inchikey"}

    def test_dataset(self, library: Path, tmp_path: Path) -> None:
        ds = library_dataset(library)
        assert ds.id == "modephere_library:library" and ds.uri == str(library.resolve())
        assert library_dataset(library, "mine").id == "mine"
        with pytest.raises(DataError, match="registry"):
            library_dataset(tmp_path)

    def test_link(self, library: Path) -> None:
        driver = FakeDriver()
        summary = link_library(KnowledgeGraph(driver), library)  # type: ignore[arg-type]
        assert (summary.nodes, summary.links, summary.unmatched) == (3, 2, 1)
        rel_calls = [p for q, p in driver.calls if "REPRESENTS" in q]
        assert rel_calls[0]["sources"] == ["modephere_library:library", "chembl_99"]


DRUGS = pl.DataFrame(
    {
        "name": ["aspirin", "gleevec", "ibuprofen", "unobtainium"],
        "cid": [2244, None, 3672, None],
    },
    schema={"name": pl.String, "cid": pl.Int64},
)


class TestDrugs:
    def test_match(self, lookup: ChemblLookup, tmp_path: Path) -> None:
        with make_client(tmp_path, PubChemSettings(), FakePubChem()) as client:
            links, unmatched = match_drugs(DRUGS, lookup, client)
        assert {(link.name, link.chembl_id, link.method) for link in links} == {
            ("aspirin", "CHEMBL25", "cid_inchikey"),
            ("gleevec", "CHEMBL941", "chembl_name"),
        }
        aspirin = next(link for link in links if link.name == "aspirin")
        assert aspirin.evidence.startswith("CID 2244 -> PubChem InChIKey BSYN")
        assert unmatched == ["ibuprofen", "unobtainium"]  # ibuprofen: not in ChEMBL

    def test_without_pubchem_names_still_match(self, lookup: ChemblLookup) -> None:
        links, unmatched = match_drugs(DRUGS, lookup, None)
        assert {link.name for link in links} == {"aspirin", "gleevec"}
        assert all(link.method == "chembl_name" for link in links)
        assert "ibuprofen" in unmatched

    def test_ambiguous_name_is_not_linked(
        self, lookup: ChemblLookup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        both = [*lookup.by_name("aspirin")[:1], *lookup.by_name("gleevec")]
        monkeypatch.setattr(lookup, "by_name", lambda name: both)
        links, unmatched = match_drugs(
            DRUGS.head(2).with_columns(cid=None), lookup, None
        )
        assert links == [] and len(unmatched) == 2

    def test_name_overrides_a_conflicting_cid(
        self, lookup: ChemblLookup, tmp_path: Path
    ) -> None:
        # dev_Pgen-style error: the name's CID is another drug's (aspirin's)
        drugs = pl.DataFrame(
            {"name": ["gleevec"], "cid": [2244]},
            schema={"name": pl.String, "cid": pl.Int64},
        )
        with make_client(tmp_path, PubChemSettings(), FakePubChem()) as client:
            (link,), _ = match_drugs(drugs, lookup, client)
        assert (link.chembl_id, link.method) == ("CHEMBL941", "chembl_name")
        assert "overrides CID 2244" in link.evidence and "(CHEMBL25)" in link.evidence

    def test_classes_and_mixtures_are_not_linked(
        self, lookup: ChemblLookup, tmp_path: Path
    ) -> None:
        names = ["Antithrombotic agents", "aspirin / imatinib", "aspirin + imatinib"]
        drugs = pl.DataFrame(
            {"name": names, "cid": [2244, 2244, None]},
            schema={"name": pl.String, "cid": pl.Int64},
        )
        assert is_class_or_mixture("antithrombotic agents", lookup)
        assert not is_class_or_mixture("(+)-aspirin", lookup)
        with make_client(tmp_path, PubChemSettings(), FakePubChem()) as client:
            links, unmatched = match_drugs(drugs, lookup, client)
        assert links == [] and unmatched == names

    @pytest.mark.parametrize("write_drugs", [True, False])
    def test_link(self, lookup: ChemblLookup, write_drugs: bool) -> None:
        driver = FakeDriver()
        summary = link_drugs(
            KnowledgeGraph(driver),  # type: ignore[arg-type]
            DRUGS,
            lookup,
            None,
            origin="file:drugs.tsv",
            write_drugs=write_drugs,
        )
        assert (summary.links, summary.unmatched) == (2, 2)
        assert summary.nodes == (4 if write_drugs else 0)
        same_as = next(p for q, p in driver.calls if "SAME_AS" in q)
        assert same_as["sources"] == ["drugs:file:drugs.tsv", "chembl_99"]
        # no PubChem client: both links come from ChEMBL names
        assert {r["props"]["method"] for r in same_as["rows"]} == {"chembl_name"}

    def test_from_file(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs_cid.tsv"
        tsv.write_text(
            "cid\tcmpd_name_cleaned\n2244\taspirin\nx\tgleevec\n1\taspirin\n\t \n"
        )
        frame = drugs_from_file(tsv)
        assert frame.rows() == [("aspirin", 2244), ("gleevec", None)]
        csv = tmp_path / "d.csv"
        csv.write_text("name\nimatinib\n")
        assert drugs_from_file(csv).rows() == [("imatinib", None)]
        pq = tmp_path / "d.parquet"
        DRUGS.write_parquet(pq)
        assert drugs_from_file(pq).height == 4
        bad = tmp_path / "bad.tsv"
        bad.write_text("smiles\nC\n")
        with pytest.raises(DataError, match="'name'"):
            drugs_from_file(bad)

    def test_from_graph(self) -> None:
        class Rows(FakeDriver):
            def execute_query(self, query, params, database_):  # noqa: ANN001, ANN202
                rec = type(
                    "R", (), {"data": lambda self: {"name": "aspirin", "cid": 2244}}
                )
                return [rec()], None, None

        frame = drugs_from_graph(KnowledgeGraph(Rows()))  # type: ignore[arg-type]
        assert frame.rows() == [("aspirin", 2244)]
