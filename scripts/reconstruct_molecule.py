"""Rebuild a molecule from its stored graph arrays, check the round trip, draw it.

Usage (from the project root)::

    uv run python scripts/reconstruct_molecule.py warfarin
    uv run python scripts/reconstruct_molecule.py "CC(=O)Oc1ccccc1C(=O)O"
    uv run python scripts/reconstruct_molecule.py CHEMBL941 --conformer 2
    uv run python scripts/reconstruct_molecule.py tamoxifen --no-3d          # 2D only, fast
    uv run python scripts/reconstruct_molecule.py KTUFNOKKBVMGRW-UHFFFAOYSA-N \
        --library data/library                                              # stored copy

What it does:

1. Gets the compound's :class:`CompoundArtifacts`: loaded from a built library
   (``--library``, by InChIKey or after resolving the query) or built in memory
   with the same pipeline the library builder uses.
2. Decodes the molecular graph (``molecule/v1``: 61 atom + 18 bond features,
   ``atomic_numbers``) back into an RDKit molecule, using nothing but the stored
   arrays: element, charge, hydrogens, aromaticity, bond orders, tetrahedral
   chirality, and E/Z double bonds (rebuilt from the stored E/Z labels).
3. Checks the round trip: the rebuilt molecule is featurized again and must
   give the stored arrays back exactly; its SMILES / InChIKey are compared
   with the stored working form.
4. Decodes the pharmacophore graph (``pharmacophore/v1`` + ``atom2feat``) into
   features and checks them against the reconstructed molecule (topological
   distances, aromatic / ring fractions), then the 3D data against the graph
   (bond lengths, feature points at member-atom centroids).
5. Numbers the atoms the IUPAC way (``scripts/_iupac.py``): PubChem's IUPAC
   name, parsed by OPSIN, matched back onto the atoms. Each part of the name
   (parent ring, substituents) has its own locants (1, 2, 4a ...) and is
   tinted in the drawing; ``O@4`` is an O named as a prefix ("4-hydroxy") on
   position 4. Modephere's own atom index (position in the working SMILES,
   used by every stored array) stays in the tables. Needs Java and the OPSIN
   jar in the project root; ``--no-iupac`` skips it.
6. Writes, to ``data/inspect/<name>/`` (``--out`` to change):
   ``molecule.png`` (IUPAC locants, parts tinted),
   ``molecule_indices.png`` (Modephere atom indices),
   ``pharmacophore_features.png`` (one panel per feature),
   ``graphs.png`` (molecular graph, pharmacophore graph and atom->feature
   membership drawn as node-link graphs, needs the ``viz`` extra),
   ``conformers_3d.html`` (3Dmol.js viewer: every conformer with its
   pharmacophore points and directions; opens in any browser) and
   ``conformers.sdf``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from _common import (
    chembl_lookup,
    is_inchikey,
    kv,
    make_query,
    note,
    output_dir,
    pubchem_client,
    section,
    show,
    subsection,
)
from _iupac import IupacNumbering, NumberingUnavailable, iupac_numbering
from rdkit import Chem
from rdkit.Chem import Draw, rdCIPLabeler, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import BondStereo

from modephere.config import get_settings
from modephere.core import CompoundError
from modephere.library import molecules as featurizer
from modephere.library.ingest.resolve import Resolver
from modephere.library.pipeline import PipelineOptions, build_compound
from modephere.library.storage import LibraryStore
from modephere.types import (
    MOLECULE_GRAPH_V1,
    PHARMACOPHORE_GRAPH_V1,
    BuildStage,
    PharmacophoreFamily,
)

if TYPE_CHECKING:
    from modephere.types import CompoundArtifacts, MoleculeGraph, PharmacophoreGraph

# --------------------------------------------------------------------------- #
# Feature layout of molecule/v1, derived from the featurizer's own vocabularies
# (private names on purpose: the slices must follow the featurizer exactly).
# --------------------------------------------------------------------------- #

ELEMENTS = [*featurizer._ELEMENTS, "other"]
HYBRIDIZATIONS = [*(str(h) for h in featurizer._HYBRIDIZATIONS), "other"]
CHIRAL_TAGS = [*featurizer._CHIRAL_TAGS]
CHIRAL_NAMES = ["none", "CW", "CCW", "other"]
BOND_TYPES = [*featurizer._BOND_TYPES]
BOND_NAMES = ["single", "double", "triple", "aromatic", "other"]
BOND_STEREO = [*featurizer._BOND_STEREO]
STEREO_NAMES = ["none", "Z", "E", "cis", "trans", "other"]


def _layout(sizes: list[tuple[str, int]]) -> dict[str, slice]:
    out, start = {}, 0
    for name, n in sizes:
        out[name] = slice(start, start + n)
        start += n
    return out


NODE = _layout(
    [
        ("element", len(ELEMENTS)),
        ("degree", len(featurizer._DEGREE_BINS)),
        ("valence", len(featurizer._VALENCE_BINS)),
        ("charge_scalar", 1),
        ("charge", len(featurizer._CHARGE_BINS) + 1),
        ("hybridization", len(HYBRIDIZATIONS)),
        ("total_hs", len(featurizer._HS_BINS)),
        ("chirality", len(CHIRAL_TAGS) + 1),
        ("aromatic", 1),
        ("in_ring", 1),
        ("ring_size", len(featurizer._RING_SIZES)),
        ("gasteiger", 1),
        ("electronegativity", 1),
        ("donor", 1),
        ("acceptor", 1),
        ("mass", 1),
        ("radicals", 1),
    ]
)
EDGE = _layout(
    [
        ("bond_type", len(BOND_NAMES)),
        ("conjugated", 1),
        ("in_ring", 1),
        ("ring_size", len(featurizer._RING_SIZES)),
        ("stereo", len(STEREO_NAMES)),
    ]
)
assert max(s.stop for s in NODE.values()) == MOLECULE_GRAPH_V1.node_dim
assert max(s.stop for s in EDGE.values()) == MOLECULE_GRAPH_V1.edge_dim

FAMILIES = list(PharmacophoreFamily)
PHARM_NODE = _layout(
    [
        ("family", len(FAMILIES) + 1),
        ("n_atoms", 1),
        ("aromatic_fraction", 1),
        ("ring_fraction", 1),
        ("mean_charge", 1),
    ]
)
assert max(s.stop for s in PHARM_NODE.values()) == PHARMACOPHORE_GRAPH_V1.node_dim
TOPO_DIST_CAP = 30.0  # graph2d.DIST_CAP: last edge feature = distance / 30

FAMILY_COLORS = {
    PharmacophoreFamily.DONOR: "#2ca02c",
    PharmacophoreFamily.ACCEPTOR: "#d62728",
    PharmacophoreFamily.POS_IONIZABLE: "#1f77b4",
    PharmacophoreFamily.NEG_IONIZABLE: "#ff7f0e",
    PharmacophoreFamily.AROMATIC: "#9467bd",
    PharmacophoreFamily.HYDROPHOBE: "#bcbd22",
}
#: Light tints for the parts of an IUPAC name (parent ring, substituents...).
PART_COLORS = [
    "#cfe3f7",
    "#fde2c4",
    "#d6f0d0",
    "#f3d1e6",
    "#e6dcf5",
    "#fff3b0",
    "#d0f0ef",
    "#f5d6d0",
]
ELEMENT_COLORS = {
    "C": "#555555",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "S": "#d4b000",
    "P": "#ff8000",
    "F": "#59c13c",
    "Cl": "#1ff01f",
    "Br": "#a62929",
    "I": "#940094",
}


def _rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _hot(row: np.ndarray, block: slice) -> int:
    """Index of the set bit of a one-hot block; -1 if the block is all zeros."""
    seg = row[block]
    return int(np.argmax(seg)) if seg.max() > 0 else -1


# --------------------------------------------------------------------------- #
# 1. Getting the artifacts
# --------------------------------------------------------------------------- #


def get_artifacts(args: argparse.Namespace) -> tuple[CompoundArtifacts, str]:
    """(artifacts, where they came from)."""
    store = LibraryStore(args.library) if args.library else None
    if store is not None and is_inchikey(args.query):
        if store.exists(args.query):
            return store.load(args.query), f"library {args.library}"
        note(f"{args.query} is not in {args.library}; building it instead")

    query = make_query(args.query, args.kind)
    options = PipelineOptions(build_3d=not args.no_3d)
    chembl = chembl_lookup()
    with pubchem_client(offline=args.offline) as client:
        resolver = Resolver(client, chembl=chembl)
        if store is not None:
            from modephere.library.standardize import standardize

            key = standardize(resolver.resolve(query)).record.inchikey
            if store.exists(key):
                return store.load(key), f"library {args.library}"
            note(f"{key} is not in {args.library}; building it instead")
        start = time.perf_counter()
        art = build_compound(query, options, resolver=resolver)
    return art, f"built in memory in {time.perf_counter() - start:.1f} s"


# --------------------------------------------------------------------------- #
# 2. Graph -> molecule
# --------------------------------------------------------------------------- #


@dataclass
class Rebuilt:
    mol: Chem.Mol
    notes: list[str]


def graph_to_mol(g: MoleculeGraph) -> Rebuilt:
    """Molecule from ``molecule/v1`` arrays only (``x``, ``edge_index``, ``edge_attr``,
    ``atomic_numbers``)."""
    notes: list[str] = []
    rw = Chem.RWMol()
    for i, row in enumerate(g.x):
        z = int(g.atomic_numbers[i])
        atom = Chem.Atom(z)
        elem = _hot(row, NODE["element"])
        if ELEMENTS[elem] == "other":
            notes.append(
                f"atom {i}: element is 'other' in x; atomic_numbers gives "
                f"{atom.GetSymbol()}"
            )
        charge_idx = _hot(row, NODE["charge"])
        if 0 <= charge_idx < len(featurizer._CHARGE_BINS):
            charge = featurizer._CHARGE_BINS[charge_idx]
        else:  # 'other' bucket: the clamped scalar charge/2 is all that is left
            charge = round(float(row[NODE["charge_scalar"]][0]) * 2)
            notes.append(f"atom {i}: charge outside -1..+1, read {charge} from scalar")
        atom.SetFormalCharge(charge)
        hs = _hot(row, NODE["total_hs"])
        if hs < 0:
            notes.append(f"atom {i}: > 4 hydrogens (saturated bin); left to RDKit")
        else:
            atom.SetNumExplicitHs(hs)
            atom.SetNoImplicit(True)
        atom.SetIsAromatic(bool(row[NODE["aromatic"]][0]))
        chiral = _hot(row, NODE["chirality"])
        if 0 <= chiral < len(CHIRAL_TAGS):
            atom.SetChiralTag(CHIRAL_TAGS[chiral])
        atom.SetNumRadicalElectrons(int(row[NODE["radicals"]][0]))
        rw.AddAtom(atom)

    stereo_bonds: dict[int, BondStereo] = {}
    ei, ea = g.edge_index, g.edge_attr
    for e in range(0, ei.shape[1], 2):  # both directions stored, bond order kept
        i, j = int(ei[0, e]), int(ei[1, e])
        btype = _hot(ea[e], EDGE["bond_type"])
        if not 0 <= btype < len(BOND_TYPES):
            notes.append(f"bond {i}-{j}: type 'other'; read as single")
            btype = 0
        rw.AddBond(i, j, BOND_TYPES[btype])
        bond = rw.GetBondBetweenAtoms(i, j)
        if BOND_TYPES[btype] == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)
        stereo = _hot(ea[e], EDGE["stereo"])
        if 0 < stereo < len(BOND_STEREO):
            stereo_bonds[bond.GetIdx()] = BOND_STEREO[stereo]

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    if stereo_bonds:
        mol = _restore_double_bond_stereo(mol, stereo_bonds, notes)
    # as the SMILES parser does: flags possible (also unassigned) stereocentres,
    # which the stereocentre-count descriptor reads
    Chem.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    return Rebuilt(mol, notes)


def _restore_double_bond_stereo(
    mol: Chem.Mol, wanted: dict[int, BondStereo], notes: list[str]
) -> Chem.Mol:
    """E/Z from the stored labels: the graph keeps E/Z but not which neighbours
    they refer to (CIP priority). Set cis on arbitrary neighbours, flip where the
    CIP label disagrees, then let RDKit re-derive E/Z in place."""

    def orient(bond: Chem.Bond, cis: bool) -> None:
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        na = next(n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx())
        nb = next(n.GetIdx() for n in b.GetNeighbors() if n.GetIdx() != a.GetIdx())
        bond.SetStereoAtoms(na, nb)
        bond.SetStereo(BondStereo.STEREOCIS if cis else BondStereo.STEREOTRANS)

    mol = Chem.Mol(mol)
    target = {BondStereo.STEREOZ: "Z", BondStereo.STEREOE: "E"}
    for idx, stereo in wanted.items():
        bond = mol.GetBondWithIdx(idx)
        if stereo in (BondStereo.STEREOCIS, BondStereo.STEREOTRANS):
            notes.append(f"bond {idx}: stored {stereo} without its reference atoms")
            continue
        orient(bond, cis=True)
    rdCIPLabeler.AssignCIPLabels(mol)
    for idx, stereo in wanted.items():
        bond = mol.GetBondWithIdx(idx)
        if stereo in target and bond.GetPropsAsDict().get("_CIPCode") != target[stereo]:
            orient(bond, cis=False)
    Chem.SetDoubleBondNeighborDirections(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


# --------------------------------------------------------------------------- #
# 3. Checks
# --------------------------------------------------------------------------- #


def _block_names(layout: dict[str, slice], columns: np.ndarray) -> list[str]:
    return sorted(
        {name for name, s in layout.items() for c in columns if s.start <= c < s.stop}
    )


def check_molecule(art: CompoundArtifacts, rebuilt: Chem.Mol) -> list[dict[str, Any]]:
    stored = art.molecule
    again = featurizer.molecule_graph(rebuilt)
    working = Chem.MolFromSmiles(art.compound.working_smiles)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(
            {"check": name, "result": "PASS" if ok else "FAIL", "detail": detail}
        )

    add("atom count", again.num_nodes == stored.num_nodes, f"{stored.num_nodes} atoms")
    add(
        "edge_index identical",
        np.array_equal(again.edge_index, stored.edge_index),
        f"{stored.num_edges // 2} bonds",
    )
    diff = ~np.isclose(again.x, stored.x, atol=1e-6)
    add(
        "atom features x identical",
        not diff.any(),
        "differs in: " + ", ".join(_block_names(NODE, np.where(diff.any(0))[0]))
        if diff.any()
        else f"{stored.x.shape[1]} features x {stored.num_nodes} atoms",
    )
    if again.edge_attr.shape == stored.edge_attr.shape:
        ediff = ~np.isclose(again.edge_attr, stored.edge_attr, atol=1e-6)
        add(
            "bond features identical",
            not ediff.any(),
            "differs in: " + ", ".join(_block_names(EDGE, np.where(ediff.any(0))[0]))
            if ediff.any()
            else f"{stored.edge_attr.shape[1]} features",
        )
    gdiff = ~np.isclose(again.global_feats, stored.global_feats, atol=1e-5)
    add(
        "global descriptors identical",
        not gdiff.any(),
        f"{int(gdiff.sum())} of {gdiff.size} differ"
        if gdiff.any()
        else f"{stored.global_feats.size} values",
    )
    smi_rebuilt = Chem.MolToSmiles(rebuilt)
    smi_working = Chem.MolToSmiles(working)
    add("canonical SMILES = working form", smi_rebuilt == smi_working, smi_rebuilt)
    key_rebuilt = Chem.MolToInchiKey(rebuilt)
    add(
        "InChIKey = working form",
        key_rebuilt == Chem.MolToInchiKey(working),
        key_rebuilt,
    )
    elems_in_x = all(ELEMENTS[_hot(r, NODE["element"])] != "other" for r in stored.x)
    checks.append(
        {
            "check": "elements recoverable from x alone",
            "result": "yes" if elems_in_x else "no",
            "detail": "else atomic_numbers is needed"
            if not elems_in_x
            else "all in the 12-element vocabulary",
        }
    )
    return checks


@dataclass
class Feature:
    index: int
    family: PharmacophoreFamily
    atoms: tuple[int, ...]


def decode_pharmacophore(p: PharmacophoreGraph) -> list[Feature]:
    members: dict[int, list[int]] = {}
    for atom, feat in p.atom2feat:
        members.setdefault(int(feat), []).append(int(atom))
    out = []
    for k, row in enumerate(p.x):
        fam = _hot(row, PHARM_NODE["family"])
        out.append(Feature(k, FAMILIES[fam], tuple(sorted(members.get(k, [])))))
    return out


def check_pharmacophore(
    art: CompoundArtifacts, feats: list[Feature], mol: Chem.Mol
) -> list[dict[str, Any]]:
    p = art.pharmacophore
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(
            {"check": name, "result": "PASS" if ok else "FAIL", "detail": detail}
        )

    stored = [
        (f.family, tuple(sorted(f.atom_ids))) for f in art.pharmacophore_2d.features
    ]
    add(
        "graph features = stored 2D feature set",
        stored == [(f.family, f.atoms) for f in feats],
        f"{len(feats)} features",
    )
    n_ok = all(
        math.isclose(
            float(p.x[f.index, PHARM_NODE["n_atoms"]][0]),
            min(len(f.atoms), 6) / 6,
            abs_tol=1e-6,  # stored as float32
        )
        for f in feats
    )
    add("n_atoms column = atom2feat sizes", n_ok)
    arom_ok = all(
        math.isclose(
            float(p.x[f.index, PHARM_NODE["aromatic_fraction"]][0]),
            sum(mol.GetAtomWithIdx(a).GetIsAromatic() for a in f.atoms) / len(f.atoms),
            abs_tol=1e-6,
        )
        for f in feats
        if f.atoms
    )
    add("aromatic fraction matches the rebuilt molecule", arom_ok)
    dist = Chem.GetDistanceMatrix(mol)
    worst = 0.0
    for e in range(p.edge_index.shape[1]):
        i, j = int(p.edge_index[0, e]), int(p.edge_index[1, e])
        stored_d = float(p.edge_attr[e, -1]) * TOPO_DIST_CAP
        real = min(float(dist[np.ix_(feats[i].atoms, feats[j].atoms)].min()), 30.0)
        worst = max(worst, abs(stored_d - real))
    add(
        "topological distances match the rebuilt molecule",
        worst < 1e-3,
        f"max deviation {worst:.4f} bonds",
    )
    return checks


def check_3d(
    art: CompoundArtifacts, mol: Chem.Mol, feats: list[Feature], conformer: int
) -> list[dict[str, Any]]:
    ens = art.conformers
    if ens is None:
        return []
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append(
            {"check": name, "result": "PASS" if ok else "FAIL", "detail": detail}
        )

    add(
        "heavy atoms in conformers = graph atoms",
        ens.num_heavy_atoms == mol.GetNumAtoms(),
        f"{ens.num_heavy_atoms} heavy + {ens.num_atoms - ens.num_heavy_atoms} H",
    )
    pos = ens.positions[conformer]
    lengths = [
        float(np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]))
        for b in mol.GetBonds()
    ]
    add(
        "graph bonds have bond lengths in 3D",
        bool(lengths) and 0.9 < min(lengths) and max(lengths) < 2.0,
        f"{min(lengths):.2f}-{max(lengths):.2f} A" if lengths else "no bonds",
    )
    p3d = art.pharmacophores_3d[conformer]
    worst = 0.0
    for f, point in zip(feats, p3d.points, strict=True):
        centroid = pos[list(f.atoms)].mean(axis=0)
        worst = max(worst, float(np.linalg.norm(centroid - np.asarray(point.position))))
    add(
        "3D points sit at member-atom centroids",
        worst < 1e-3,
        f"max offset {worst:.4f} A",
    )
    return checks


# --------------------------------------------------------------------------- #
# 4. Printing
# --------------------------------------------------------------------------- #


def atom_table(
    g: MoleculeGraph, numbering: IupacNumbering | None = None
) -> pl.DataFrame:
    rows = []
    for i, r in enumerate(g.x):
        charge_idx = _hot(r, NODE["charge"])
        rows.append(
            {
                "idx": i,
                "iupac": numbering.label(i, with_part=True) if numbering else "-",
                "elem": Chem.GetPeriodicTable().GetElementSymbol(
                    int(g.atomic_numbers[i])
                ),
                "deg": _hot(r, NODE["degree"]),
                "chg": featurizer._CHARGE_BINS[charge_idx]
                if 0 <= charge_idx < 3
                else "other",
                "hyb": HYBRIDIZATIONS[_hot(r, NODE["hybridization"])],
                "Hs": _hot(r, NODE["total_hs"]),
                "chiral": CHIRAL_NAMES[_hot(r, NODE["chirality"])],
                "arom": int(r[NODE["aromatic"]][0]),
                "ring": ",".join(
                    str(n)
                    for n, on in zip(
                        featurizer._RING_SIZES, r[NODE["ring_size"]], strict=True
                    )
                    if on
                ),
                "q": round(float(r[NODE["gasteiger"]][0]), 3),
                "don": int(r[NODE["donor"]][0]),
                "acc": int(r[NODE["acceptor"]][0]),
            }
        )
    return pl.DataFrame(rows)


def bond_table(g: MoleculeGraph) -> pl.DataFrame:
    rows = []
    for e in range(0, g.edge_index.shape[1], 2):
        a = g.edge_attr[e]
        rows.append(
            {
                "bond": e // 2,
                "i": int(g.edge_index[0, e]),
                "j": int(g.edge_index[1, e]),
                "type": BOND_NAMES[_hot(a, EDGE["bond_type"])],
                "conj": int(a[EDGE["conjugated"]][0]),
                "ring": ",".join(
                    str(n)
                    for n, on in zip(
                        featurizer._RING_SIZES, a[EDGE["ring_size"]], strict=True
                    )
                    if on
                ),
                "stereo": STEREO_NAMES[_hot(a, EDGE["stereo"])],
            }
        )
    return pl.DataFrame(rows)


def feature_table(p: PharmacophoreGraph, feats: list[Feature]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "feat": f.index,
                "family": str(f.family),
                "atoms": ", ".join(map(str, f.atoms)),
                "aromatic_frac": round(
                    float(p.x[f.index, PHARM_NODE["aromatic_fraction"]][0]), 2
                ),
                "ring_frac": round(
                    float(p.x[f.index, PHARM_NODE["ring_fraction"]][0]), 2
                ),
                "mean_charge": round(
                    float(p.x[f.index, PHARM_NODE["mean_charge"]][0]), 3
                ),
            }
            for f in feats
        ]
    )


def distance_matrix(p: PharmacophoreGraph, feats: list[Feature]) -> pl.DataFrame:
    k = len(feats)
    d = np.zeros((k, k), dtype=int)
    for e in range(p.edge_index.shape[1]):
        i, j = int(p.edge_index[0, e]), int(p.edge_index[1, e])
        d[i, j] = round(float(p.edge_attr[e, -1]) * TOPO_DIST_CAP)
    labels = [f"{f.index}:{str(f.family)[:4]}" for f in feats]
    return pl.DataFrame({"": labels, **{lab: d[:, c] for c, lab in enumerate(labels)}})


def conformer_table(art: CompoundArtifacts) -> pl.DataFrame:
    ens = art.conformers
    assert ens is not None
    energies = ens.energies
    low = float(energies.min()) if energies is not None else 0.0
    return pl.DataFrame(
        [
            {
                "pos": k,
                "conformer_id": c.conformer_id,
                "stereoisomer": c.stereoisomer,
                "E_rel_kcal": round(float(energies[k]) - low, 2)
                if energies is not None
                else None,
                "converged": c.converged,
                "points": len(art.pharmacophores_3d[k].points),
            }
            for k, c in enumerate(ens.conformers)
        ]
    )


def points_table(art: CompoundArtifacts, conformer: int) -> pl.DataFrame:
    p3d = art.pharmacophores_3d[conformer]
    return pl.DataFrame(
        [
            {
                "feat": k,
                "family": str(pt.family),
                "atoms": ", ".join(map(str, pt.atom_ids)),
                "x": round(pt.position[0], 2),
                "y": round(pt.position[1], 2),
                "z": round(pt.position[2], 2),
                "direction": "-"
                if pt.direction is None
                else "({:+.2f}, {:+.2f}, {:+.2f})".format(*pt.direction),
            }
            for k, pt in enumerate(p3d.points)
        ]
    )


# --------------------------------------------------------------------------- #
# 5. Pictures
# --------------------------------------------------------------------------- #


def draw_molecule(
    mol: Chem.Mol, path: Path, title: str, numbering: IupacNumbering | None = None
) -> None:
    """Atom indices, or IUPAC locants with each part of the name tinted."""
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(900, 650)
    opts = d.drawOptions()
    # RDKit's stubs type these attributes as MolDrawOptions, not bool / int
    opts.legendFontSize = 22  # type: ignore[assignment]
    if numbering is None:
        opts.addAtomIndices = True  # type: ignore[assignment]
        d.DrawMolecule(m, legend=title)
    else:
        opts.annotationFontScale = 0.8  # type: ignore[assignment]
        for atom in m.GetAtoms():
            atom.SetProp("atomNote", numbering.label(atom.GetIdx()))
        colors: dict[int, tuple[float, float, float]] = {}
        for part in numbering.parts:
            colors |= dict.fromkeys(part.atoms, _rgb(PART_COLORS[part.number % 8]))
        part_of = {i: loc.part for i, loc in numbering.atoms.items()}
        bonds = {  # bonds inside one part share its tint
            b.GetIdx(): colors[b.GetBeginAtomIdx()]
            for b in m.GetBonds()
            if b.GetBeginAtomIdx() in colors
            and part_of.get(b.GetBeginAtomIdx()) == part_of.get(b.GetEndAtomIdx())
        }
        d.DrawMolecule(
            m,
            highlightAtoms=list(colors),
            highlightAtomColors=colors,
            highlightBonds=list(bonds),
            highlightBondColors=bonds,
            legend=title,
        )
    d.FinishDrawing()
    path.write_bytes(d.GetDrawingText())


def draw_features(
    mol: Chem.Mol,
    feats: list[Feature],
    path: Path,
    numbering: IupacNumbering | None = None,
) -> None:
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    mols, atoms, colors, legends = [], [], [], []
    for f in feats:
        mols.append(m)
        atoms.append(list(f.atoms))
        color = _rgb(FAMILY_COLORS[f.family])
        colors.append(dict.fromkeys(f.atoms, color))
        members = (
            ", ".join(numbering.label(a, with_part=True) for a in f.atoms)
            if numbering
            else ", ".join(map(str, f.atoms))
        )
        legends.append(f"#{f.index} {f.family} [{members}]")
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(4, max(1, len(mols))),
        subImgSize=(360, 300),
        legends=legends,
        highlightAtomLists=atoms,
        highlightAtomColors=colors,
        returnPNG=False,
    )
    img.save(path)


def draw_graphs(
    mol: Chem.Mol,
    art: CompoundArtifacts,
    feats: list[Feature],
    path: Path,
    numbering: IupacNumbering | None = None,
) -> bool:
    """Node-link drawings of the stored graphs; False if matplotlib is missing."""
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR", str(get_settings().paths.cache / "matplotlib")
        )
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return False

    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    conf = m.GetConformer()
    xy = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
            for i in range(m.GetNumAtoms())
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
    _molecular_panel(axes[0], art.molecule, xy, numbering)
    _pharmacophore_panel(axes[1], m, xy, art.pharmacophore, feats)
    handles = [
        Line2D([], [], marker="o", ls="", markersize=10, color=col, label=str(fam))
        for fam, col in FAMILY_COLORS.items()
        if any(f.family == fam for f in feats)
    ]
    axes[1].legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(art.compound.working_smiles, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _molecular_panel(
    ax: Any, g: MoleculeGraph, xy: np.ndarray, numbering: IupacNumbering | None
) -> None:
    """The molecular graph, straight from edge_index / edge_attr."""
    for e in range(0, g.edge_index.shape[1], 2):
        i, j = int(g.edge_index[0, e]), int(g.edge_index[1, e])
        kind = BOND_NAMES[_hot(g.edge_attr[e], EDGE["bond_type"])]
        width = {"single": 1.5, "double": 4.0, "triple": 6.0}.get(kind, 2.5)
        style = "--" if kind == "aromatic" else "-"
        ax.plot(*xy[[i, j]].T, style, color="#999999", lw=width, zorder=1)
    for i, z in enumerate(g.atomic_numbers):
        sym = Chem.GetPeriodicTable().GetElementSymbol(int(z))
        ax.scatter(
            *xy[i],
            s=420,
            color=ELEMENT_COLORS.get(sym, "#bb88dd"),
            zorder=2,
            edgecolors="white",
        )
        ax.annotate(
            f"{sym}{i}",
            xy[i],
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            zorder=3,
            weight="bold",
        )
        if numbering is not None:  # IUPAC locant under the node
            ax.annotate(
                numbering.label(i, with_part=True),
                xy[i] + np.array([0.0, -0.42]),
                ha="center",
                va="center",
                fontsize=6,
                color="#1f4e79",
                zorder=3,
            )
    ax.set_title(
        f"Molecular graph (molecule/v1): {g.num_nodes} nodes, {g.num_edges} directed edges\n"
        "edge width = bond order, dashed = aromatic",
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.axis("off")


def _pharmacophore_panel(
    ax: Any,
    m: Chem.Mol,
    xy: np.ndarray,
    p: PharmacophoreGraph,
    feats: list[Feature],
) -> None:
    """The pharmacophore graph over the skeleton, with atom -> feature membership."""
    for b in m.GetBonds():
        ax.plot(
            *xy[[b.GetBeginAtomIdx(), b.GetEndAtomIdx()]].T,
            "-",
            color="#dddddd",
            lw=1.5,
            zorder=1,
        )
    ax.scatter(*xy.T, s=40, color="#cccccc", zorder=2)
    centres = []
    seen: dict[tuple[float, float], int] = {}
    for f in feats:
        c = xy[list(f.atoms)].mean(axis=0) if f.atoms else np.zeros(2)
        key = (round(c[0], 1), round(c[1], 1))
        n = seen.get(key, 0)
        seen[key] = n + 1
        if n:  # several features on the same atoms: fan them out
            angle = n * 2.1
            c = c + 0.55 * np.array([math.cos(angle), math.sin(angle)])
        centres.append(c)
    centres_a = np.array(centres) if centres else np.zeros((0, 2))
    for e in range(p.edge_index.shape[1]):
        i, j = int(p.edge_index[0, e]), int(p.edge_index[1, e])
        if i >= j:
            continue
        d = round(float(p.edge_attr[e, -1]) * TOPO_DIST_CAP)
        ax.plot(
            *centres_a[[i, j]].T, "-", color="#4a6fa5", lw=0.8, alpha=0.35, zorder=3
        )
        mid = centres_a[[i, j]].mean(axis=0)
        ax.annotate(str(d), mid, fontsize=6, color="#4a6fa5", ha="center", zorder=4)
    for f, c in zip(feats, centres_a, strict=True):
        for a in f.atoms:
            ax.plot(
                *np.array([c, xy[a]]).T,
                ":",
                color=FAMILY_COLORS[f.family],
                lw=1.2,
                zorder=4,
            )
        ax.scatter(
            *c, s=520, color=FAMILY_COLORS[f.family], zorder=5, edgecolors="black"
        )
        ax.annotate(
            str(f.index),
            c,
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            zorder=6,
        )
    ax.set_title(
        f"Pharmacophore graph (pharmacophore/v1): {p.num_nodes} features, complete graph\n"
        "edge label = topological distance (bonds); dotted = atom2feat membership",
        fontsize=11,
    )
    ax.set_aspect("equal")
    ax.axis("off")


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
<style>
 body { font-family: system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }
 #viewer { flex: 1; position: relative; }
 #side { width: 330px; padding: 12px 16px; overflow-y: auto; border-left: 1px solid #ddd;
         font-size: 13px; }
 h1 { font-size: 15px; margin: 4px 0 8px; } code { font-size: 11px; word-break: break-all; }
 .sw { display: inline-block; width: 12px; height: 12px; border-radius: 6px;
       margin-right: 6px; vertical-align: middle; }
 label { display: block; margin: 6px 0; } select { width: 100%; }
 table { border-collapse: collapse; width: 100%; } td { padding: 2px 4px; }
</style></head>
<body><div id="viewer"></div>
<div id="side">
 <h1>__TITLE__</h1><code>__SMILES__</code>
 <label>Conformer <select id="conf"></select></label>
 <label><input type="checkbox" id="pts" checked> pharmacophore points</label>
 <label><input type="checkbox" id="dirs" checked> directions</label>
 <label><input type="checkbox" id="hyd" checked> hydrogens</label>
 <label><input type="checkbox" id="lbl"> atom indices</label>
 <div id="legend"></div><p id="info"></p>
 <p style="color:#777">Atom order: heavy atoms as in the 2D graph, then hydrogens.
 Points sit at the centroid of their member atoms; arrows are donor / acceptor
 directions and aromatic ring normals.</p>
</div>
<script>
const DATA = __DATA__;
const viewer = $3Dmol.createViewer("viewer", {backgroundColor: "white"});
const sel = document.getElementById("conf");
DATA.conformers.forEach((c, k) => {
  const o = document.createElement("option"); o.value = k;
  o.text = `#${k} (id ${c.id}, stereo ${c.stereo})` +
           (c.energy === null ? "" : `  ${c.energy.toFixed(2)} kcal/mol`);
  sel.appendChild(o);
});
document.getElementById("legend").innerHTML = Object.entries(DATA.colors).map(
  ([f, c]) => `<div><span class="sw" style="background:${c}"></span>${f}</div>`).join("");
function draw() {
  const k = +sel.value, c = DATA.conformers[k];
  viewer.clear();
  const showH = document.getElementById("hyd").checked;
  let xyz = "", n = 0;
  c.xyz.forEach((p, i) => { if (showH || DATA.elements[i] !== "H") {
    xyz += `${DATA.elements[i]} ${p[0]} ${p[1]} ${p[2]}\\n`; n++; } });
  const model = viewer.addModel(`${n}\\n${DATA.title}\\n` + xyz, "xyz");
  viewer.setStyle({}, {stick: {radius: 0.15}, sphere: {scale: 0.22}});
  if (document.getElementById("lbl").checked) {
    model.selectedAtoms({}).forEach((a, i) => { if (i < DATA.heavy)
      viewer.addLabel(String(i), {position: a, fontSize: 10, inFront: true,
        backgroundOpacity: 0.4}); });
  }
  if (document.getElementById("pts").checked) c.points.forEach(p => {
    viewer.addSphere({center: {x: p.pos[0], y: p.pos[1], z: p.pos[2]}, radius: 0.55,
                      color: DATA.colors[p.family], opacity: 0.55});
    if (p.dir && document.getElementById("dirs").checked) viewer.addArrow({
      start: {x: p.pos[0], y: p.pos[1], z: p.pos[2]},
      end: {x: p.pos[0] + 1.6 * p.dir[0], y: p.pos[1] + 1.6 * p.dir[1],
            z: p.pos[2] + 1.6 * p.dir[2]},
      radius: 0.07, color: DATA.colors[p.family]});
  });
  document.getElementById("info").textContent =
    `${c.points.length} pharmacophore points in this conformer`;
  viewer.zoomTo(); viewer.render();
}
["conf", "pts", "dirs", "hyd", "lbl"].forEach(id =>
  document.getElementById(id).addEventListener("change", draw));
sel.value = DATA.start; draw();
</script></body></html>
"""


def write_3d_html(art: CompoundArtifacts, title: str, start: int, path: Path) -> None:
    ens = art.conformers
    assert ens is not None
    table = Chem.GetPeriodicTable()
    elements = [table.GetElementSymbol(int(z)) for z in art.molecule.atomic_numbers]
    elements += ["H"] * (ens.num_atoms - ens.num_heavy_atoms)
    energies = ens.energies
    data = {
        "title": title,
        "heavy": ens.num_heavy_atoms,
        "elements": elements,
        "start": start,
        "colors": {str(f): c for f, c in FAMILY_COLORS.items()},
        "conformers": [
            {
                "id": rec.conformer_id,
                "stereo": rec.stereoisomer,
                "energy": None
                if energies is None
                else float(energies[k] - energies.min()),
                "xyz": np.round(ens.positions[k], 4).tolist(),
                "points": [
                    {
                        "family": str(pt.family),
                        "pos": [round(v, 4) for v in pt.position],
                        "dir": None
                        if pt.direction is None
                        else [round(v, 4) for v in pt.direction],
                    }
                    for pt in art.pharmacophores_3d[k].points
                ],
            }
            for k, rec in enumerate(ens.conformers)
        ],
    }
    html = (
        HTML.replace("__TITLE__", title)
        .replace("__SMILES__", art.compound.microspecies_smiles)
        .replace("__DATA__", json.dumps(data))
    )
    path.write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- #


def report_header(art: CompoundArtifacts, name: str, origin: str) -> None:
    rec = art.compound
    section(f"{name}  ({origin})")
    kv(
        {
            "InChIKey (registration)": rec.inchikey,
            "working SMILES (2D graph)": rec.working_smiles,
            "microspecies (3D, features)": rec.microspecies_smiles,
            "ChEMBL parent": rec.resolved.chembl_id,
            "PubChem CID": rec.resolved.cid,
        }
    )
    subsection("Stored arrays")
    arrays: dict[str, Any] = {
        "molecule.x": art.molecule.x,
        "molecule.edge_index": art.molecule.edge_index,
        "molecule.edge_attr": art.molecule.edge_attr,
        "molecule.global_feats": art.molecule.global_feats,
        "molecule.atomic_numbers": art.molecule.atomic_numbers,
        "pharmacophore.x": art.pharmacophore.x,
        "pharmacophore.edge_index": art.pharmacophore.edge_index,
        "pharmacophore.edge_attr": art.pharmacophore.edge_attr,
        "pharmacophore.global_feats": art.pharmacophore.global_feats,
        "pharmacophore.atom2feat": art.pharmacophore.atom2feat,
        "pharm2d_fingerprint (Gobbi)": art.pharm2d_fingerprint,
        "conformers.positions": None
        if art.conformers is None
        else art.conformers.positions,
        "pharm3d_fingerprint": art.pharm3d_fingerprint,
    }
    kv({k: str(v.shape) for k, v in arrays.items() if v is not None})


def report_iupac(
    art: CompoundArtifacts, *, offline: bool, limit: int
) -> IupacNumbering | None:
    """IUPAC locants via PubChem's name and OPSIN; None (with the reason) if not."""
    section("IUPAC numbering")
    working = Chem.MolFromSmiles(art.compound.working_smiles)
    try:
        with pubchem_client(offline=offline) as client:
            numbering = iupac_numbering(art.compound, working, client)
    except NumberingUnavailable as e:
        note(f"not available: {e}")
        note("pictures and tables keep Modephere's atom indices only")
        return None
    kv({"IUPAC name": numbering.name, "from": numbering.source})
    show(
        pl.DataFrame(
            [
                {
                    "part": p.letter,
                    "fragment": p.smiles,
                    "atoms (index=locant)": ", ".join(
                        f"{i}={numbering.label(i)}" for i in p.atoms
                    ),
                }
                for p in numbering.parts
            ]
        ),
        limit,
        title="Parts of the name (each numbered from 1, in name order)",
        max_str=70,
    )
    loose = sorted(i for i, a in numbering.atoms.items() if a.part is None)
    if loose:
        kv(
            {
                "substituent atoms (letter / greek locants)": ", ".join(
                    f"{i}={numbering.label(i)}" for i in loose
                )
            }
        )
    note(
        "Modephere's index is the atom's position in the working SMILES (what the "
        "stored arrays use); the IUPAC locant is its number in the name"
    )
    return numbering


def report_molecule(
    art: CompoundArtifacts, limit: int, numbering: IupacNumbering | None
) -> tuple[Rebuilt, list[dict[str, Any]]]:
    section("Molecular graph (decoded from x / edge_attr)")
    show(
        atom_table(art.molecule, numbering),
        limit,
        title="Atoms (q = Gasteiger charge)",
        max_str=12,
    )
    show(bond_table(art.molecule), limit, title="Bonds")

    section("Reconstruction: graph -> molecule")
    rebuilt = graph_to_mol(art.molecule)
    working = Chem.MolFromSmiles(art.compound.working_smiles)
    kv(
        {
            "rebuilt SMILES": Chem.MolToSmiles(rebuilt.mol),
            "stored working SMILES": Chem.MolToSmiles(working),
        }
    )
    for n in rebuilt.notes:
        note(n)
    return rebuilt, check_molecule(art, rebuilt.mol)


def report_pharmacophore(
    art: CompoundArtifacts, mol: Chem.Mol, limit: int
) -> tuple[list[Feature], list[dict[str, Any]]]:
    section("Pharmacophore graph (decoded from x / atom2feat / edge_attr)")
    feats = decode_pharmacophore(art.pharmacophore)
    show(feature_table(art.pharmacophore, feats), limit, title="Features")
    show(
        distance_matrix(art.pharmacophore, feats),
        limit,
        title="Topological distances between features (bonds, closest member atoms)",
    )
    counts = np.expm1(art.pharmacophore.global_feats[0, : len(FAMILIES)])
    kv(
        {
            "family counts (global_feats, expm1)": dict(
                zip(
                    map(str, FAMILIES), counts.round().astype(int).tolist(), strict=True
                )
            )
        }
    )
    return feats, check_pharmacophore(art, feats, mol)


def report_3d(
    art: CompoundArtifacts,
    mol: Chem.Mol,
    feats: list[Feature],
    conformer: int,
    limit: int,
) -> list[dict[str, Any]]:
    ens = art.conformers
    assert ens is not None
    section("Conformers and 3D pharmacophores")
    kv(
        {
            "conformers": ens.num_conformers,
            "force field": ens.force_field,
            "atoms (heavy + H)": f"{ens.num_heavy_atoms} + "
            f"{ens.num_atoms - ens.num_heavy_atoms}",
            "3D triplet fingerprint bits set": None
            if art.pharm3d_fingerprint is None
            else int(np.count_nonzero(art.pharm3d_fingerprint)),
        }
    )
    show(conformer_table(art), limit, title="Ensemble")
    show(
        points_table(art, conformer), limit, title=f"3D points, conformer #{conformer}"
    )
    return check_3d(art, mol, feats, conformer)


def write_files(
    art: CompoundArtifacts,
    mol: Chem.Mol,
    feats: list[Feature],
    *,
    name: str,
    conformer: int,
    out: Path,
    numbering: IupacNumbering | None,
) -> None:
    section(f"Files in {out}")
    title = f"{name} (rebuilt from graph)"
    if numbering is not None:
        draw_molecule(mol, out / "molecule.png", f"{title}: IUPAC locants", numbering)
        print(f"  {out / 'molecule.png'}  (IUPAC locants, parts tinted)")
    draw_molecule(
        mol,
        out / ("molecule_indices.png" if numbering else "molecule.png"),
        f"{title}: Modephere atom indices",
    )
    print(f"  {out / ('molecule_indices.png' if numbering else 'molecule.png')}")
    draw_features(mol, feats, out / "pharmacophore_features.png", numbering)
    print(f"  {out / 'pharmacophore_features.png'}")
    if draw_graphs(mol, art, feats, out / "graphs.png", numbering):
        print(f"  {out / 'graphs.png'}")
    else:
        note("graphs.png skipped: install the viz extra (uv sync --extra viz)")
    if art.conformers is not None:
        from modephere.library.export import write_sdf

        write_3d_html(art, name, conformer, out / "conformers_3d.html")
        print(f"  {out / 'conformers_3d.html'}  (needs internet for 3Dmol.js)")
        n = write_sdf(art, out / "conformers.sdf")
        print(f"  {out / 'conformers.sdf'}  ({n} conformers)")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    parser.add_argument(
        "query", help="Name, ChEMBL ID, CID, SMILES, or an InChIKey with --library."
    )
    parser.add_argument(
        "--kind", default="auto", choices=["auto", "name", "chembl_id", "cid", "smiles"]
    )
    parser.add_argument(
        "--library", type=Path, help="Read the stored copy from this library."
    )
    parser.add_argument("--out", type=Path, help="Output root (default: data/inspect).")
    parser.add_argument(
        "--conformer", type=int, help="Conformer position (default: lowest energy)."
    )
    parser.add_argument(
        "--no-3d", action="store_true", help="Skip conformers (2D only)."
    )
    parser.add_argument(
        "--offline", action="store_true", help="PubChem from the cache only."
    )
    parser.add_argument(
        "--no-iupac",
        action="store_true",
        help="Skip IUPAC numbering (PubChem name + OPSIN).",
    )
    parser.add_argument("--limit", type=int, default=60, help="Rows per table.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        art, origin = get_artifacts(args)
    except CompoundError as e:
        section(f"{args.query}: the library cannot build this compound")
        kv({"stage": e.stage, "category": e.category, "reason": str(e)})
        if e.stage is BuildStage.CONFORMERS:
            note("the 2D representations may still work: re-run with --no-3d")
        return 2
    rec = art.compound
    name = str(rec.resolved.title or rec.resolved.chembl_id or rec.inchikey)
    conformer = 0
    if art.conformers is not None:
        conformer = art.conformers.lowest if args.conformer is None else args.conformer
        if not 0 <= conformer < art.conformers.num_conformers:
            note(f"--conformer must be 0..{art.conformers.num_conformers - 1}")
            return 1

    report_header(art, name, origin)
    numbering = (
        None
        if args.no_iupac
        else report_iupac(art, offline=args.offline, limit=args.limit)
    )
    rebuilt, checks = report_molecule(art, args.limit, numbering)
    feats, more = report_pharmacophore(art, rebuilt.mol, args.limit)
    checks += more
    if art.conformers is not None:
        checks += report_3d(art, rebuilt.mol, feats, conformer, args.limit)

    section("Checks")
    show(pl.DataFrame(checks), 100, max_str=70)
    failed = sum(c["result"] == "FAIL" for c in checks)
    root = args.out or get_settings().paths.data / "inspect"
    write_files(
        art,
        rebuilt.mol,
        feats,
        name=name,
        conformer=conformer,
        out=output_dir(root, name),
        numbering=numbering,
    )
    print(f"\n{len(checks) - failed} checks passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
