"""Byte-identical parity with dev_Pgen's drug and pharmacophore graphs (PLAN M3).

Golden outputs come from dev_Pgen itself (``scripts/make_parity_fixtures.py``).
Each SMILES is prepared exactly the way dev_Pgen prepares it (parse, largest
fragment), so a difference can only come from the ported featurizers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from modephere.library.molecules import molecule_graph
from modephere.library.pharmacophore.features import perceive
from modephere.library.pharmacophore.graph2d import pharmacophore_graph
from modephere.library.standardize import RDKIT_VERSION, protonate

FIXTURE = Path(__file__).parents[2] / "fixtures" / "dev_pgen_parity.npz"
_GOLDEN = np.load(FIXTURE)
_META = json.loads(bytes(_GOLDEN["meta"]).decode())
PANEL: list[str] = _META["panel"]

pytestmark = pytest.mark.skipif(
    _META["rdkit"] != RDKIT_VERSION,
    reason=f"golden data from RDKit {_META['rdkit']}, running {RDKIT_VERSION}",
)


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return mol if len(frags) <= 1 else max(frags, key=lambda m: m.GetNumAtoms())


def _drug_mol(smiles: str) -> Chem.Mol:
    """dev_Pgen ``smiles_to_graph(strip_salts=True)`` preparation."""
    return _largest_fragment(Chem.MolFromSmiles(smiles))


def _pharm_mol(smiles: str) -> Chem.Mol:
    """dev_Pgen ``_prepare_mol(strip_salts=True)`` preparation."""
    mol = _largest_fragment(Chem.MolFromSmiles(smiles))
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        Chem.FastFindRings(mol)
    return mol


def _assert_same(ours: np.ndarray, key: str) -> None:
    golden = _GOLDEN[key]
    assert ours.dtype == golden.dtype, key
    assert ours.shape == golden.shape, key
    assert np.array_equal(ours, golden), key


@pytest.mark.parametrize("i", range(len(PANEL)), ids=PANEL)
def test_molecule_graph_matches_dev_pgen(i: int) -> None:
    graph = molecule_graph(_drug_mol(PANEL[i]))
    for attr in ("x", "edge_index", "edge_attr", "global_feats"):
        _assert_same(getattr(graph, attr), f"{i}/drug/{attr}")


@pytest.mark.parametrize("i", range(len(PANEL)), ids=PANEL)
def test_pharmacophore_graph_matches_dev_pgen(i: int) -> None:
    micro = protonate(_pharm_mol(PANEL[i]))
    graph = pharmacophore_graph(micro, perceive(micro))
    for attr in ("x", "edge_index", "edge_attr", "global_feats", "atom2feat"):
        _assert_same(getattr(graph, attr), f"{i}/pharm/{attr}")
    assert graph.num_drug_atoms == int(_GOLDEN[f"{i}/pharm/num_drug_atoms"])
