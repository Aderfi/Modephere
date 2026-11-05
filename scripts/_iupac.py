"""IUPAC locants for the atoms of a molecule (inspection scripts only).

Modephere numbers atoms by their position in the working SMILES (0, 1, 2 ...):
that order is what every stored array uses. IUPAC numbering is different: each
ring system or chain of the name carries its own locants (coumarin: O1, C2 ...
C4a ... C8a; a substituent restarts at 1), fixed by the 2013 Recommendations
(P-25 for fused rings, P-31.1.4 for lowest locants). RDKit cannot number atoms
that way, so this module takes the route through a name:

1. the compound's IUPAC name from PubChem (computed by PubChem from the
   structure; found by CID, else by InChIKey, else by name);
2. OPSIN (https://github.com/dan2097/opsin) parses the name back into a
   structure whose atoms carry their locants (CML ``cmlDict:locant`` labels);
3. OPSIN's heavy-atom skeleton is matched onto Modephere's (element and
   connectivity only, so tautomers and charges do not matter), giving every
   atom index its locant(s).

The name's parts (parent ring, substituent chains...) are recovered from the
locants: OPSIN numbers each part from 1, so bonded atoms are joined into one
part only while no locant repeats in it. Parts are listed in name order
(OPSIN's atom order). Heteroatoms named as substituent prefixes (hydroxy O,
the N of an N-substituted amide, chloro) carry a letter locant (O, N, Cl) and
belong to no part.

OPSIN needs Java and its jar: ``$MODEPHERE_OPSIN_JAR``, else ``opsin*.jar`` in
the project root or ``data/tools/``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem

from modephere.config import PROJECT_ROOT

if TYPE_CHECKING:
    from modephere.library.ingest.pubchem import PubChemClient
    from modephere.types import StandardizedCompound

_CML = "{http://www.xml-cml.org/schema}"
_NUMERIC = re.compile(r"^\d+[a-z]?'*$")  # 1, 4a, 2', 3a''
_ELEMENT = re.compile(r"^[A-Z][a-z]?'*$")  # N, O', Cl
_GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "omega": "ω",
}
_POSITIONAL = {"ortho", "meta", "para"}


@dataclass(frozen=True)
class AtomLocant:
    index: int  # Modephere atom index
    locant: str  # the one to show: "4a", "N", "α"
    locants: tuple[str, ...]  # everything OPSIN gave
    part: int | None  # part number in name order; None: substituent heteroatom
    attached_to: tuple[int, ...] = ()  # part-less atoms: the atom(s) they sit on


@dataclass
class Part:
    number: int
    atoms: list[int]  # Modephere indices
    smiles: str = ""

    @property
    def letter(self) -> str:
        return chr(ord("A") + self.number) if self.number < 26 else f"P{self.number}"


@dataclass
class IupacNumbering:
    name: str
    source: str
    atoms: dict[int, AtomLocant] = field(default_factory=dict)
    parts: list[Part] = field(default_factory=list)

    def label(self, index: int, *, with_part: bool = False) -> str:
        """``4a`` (or ``4a(B)``); ``?`` for an atom without a locant."""
        loc = self.atoms.get(index)
        if loc is None:
            return "?"
        if loc.part is None and loc.attached_to:  # 4-hydroxy: "O@4"; bridges: "O@1/2"
            on = "/".join(self.label(i, with_part=with_part) for i in loc.attached_to)
            return f"{loc.locant}@{on}"
        if with_part and loc.part is not None and len(self.parts) > 1:
            return f"{loc.locant}({self.parts[loc.part].letter})"
        return loc.locant


class NumberingUnavailable(Exception):
    """Why no IUPAC numbering could be made (shown to the user)."""


# --------------------------------------------------------------------------- #
# 1. IUPAC name
# --------------------------------------------------------------------------- #


def pubchem_iupac_name(
    rec: StandardizedCompound, working: Chem.Mol, client: PubChemClient
) -> tuple[str, str]:
    """(name, source) from PubChem; the CID must have our skeleton (InChIKey block 1)."""
    candidates: list[tuple[int, str]] = []
    if rec.resolved.cid is not None:
        candidates.append((rec.resolved.cid, "resolved CID"))
    for key in dict.fromkeys([rec.inchikey, Chem.MolToInchiKey(working)]):
        found = client.inchikey_to_cids(key)
        candidates += [(c, f"InChIKey {key}") for c in found.value or ()]
    title = rec.resolved.title
    if not candidates and title:
        found = client.name_to_cids(title)
        candidates += [(c, f"name {title!r}") for c in (found.value or ())[:3]]
    if not candidates:
        msg = "PubChem has no CID for this structure (offline, or not in PubChem)"
        raise NumberingUnavailable(msg)
    skeleton = rec.inchikey.split("-")[0]
    records = client.properties([c for c, _ in candidates])
    for cid, how in candidates:
        record = records[cid].value
        if record is None or not record.iupac_name:
            continue
        if record.inchikey and record.inchikey.split("-")[0] != skeleton:
            continue  # a name-matched CID with another skeleton
        return record.iupac_name, f"PubChem CID {cid} (via {how})"
    msg = "no PubChem record with an IUPAC name and this skeleton"
    raise NumberingUnavailable(msg)


# --------------------------------------------------------------------------- #
# 2. OPSIN
# --------------------------------------------------------------------------- #


def find_opsin_jar() -> Path | None:
    env = os.environ.get("MODEPHERE_OPSIN_JAR")
    if env:
        return Path(env)
    for folder in (PROJECT_ROOT, PROJECT_ROOT / "data" / "tools"):
        jars: list[Path] = sorted(Path(folder).glob("opsin*.jar"))
        if jars:
            return jars[-1]
    return None


def opsin_cml(name: str, jar: Path) -> str:
    java = shutil.which("java")
    if java is None:
        msg = "Java is not installed (OPSIN needs it)"
        raise NumberingUnavailable(msg)
    result = subprocess.run(  # noqa: S603 -- fixed argv, name passed on stdin
        [java, "-jar", str(jar), "-ocml"],
        input=name + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if "<molecule" not in result.stdout:
        msg = f"OPSIN could not parse {name!r}: {result.stderr.strip()[-200:]}"
        raise NumberingUnavailable(msg)
    return result.stdout[result.stdout.index("<?xml") :]


@dataclass
class _OpsinAtom:
    opsin_id: int
    element: str
    locants: tuple[str, ...]


def parse_cml(cml: str) -> tuple[list[_OpsinAtom], list[tuple[int, int]]]:
    """Heavy atoms (in OPSIN order) and the bonds between them (positions)."""
    root = ET.fromstring(cml)  # noqa: S314 -- OPSIN output, local process
    atoms: list[_OpsinAtom] = []
    position: dict[str, int] = {}
    for el in root.iter(f"{_CML}atom"):
        if el.get("elementType") == "H":
            continue
        position[el.get("id", "")] = len(atoms)
        atoms.append(
            _OpsinAtom(
                opsin_id=int(el.get("id", "a0")[1:]),
                element=el.get("elementType", "C"),
                locants=tuple(
                    lab.get("value", "")
                    for lab in el.iter(f"{_CML}label")
                    if lab.get("dictRef") == "cmlDict:locant"
                ),
            )
        )
    bonds = []
    for el in root.iter(f"{_CML}bond"):
        a, b = el.get("atomRefs2", "").split()
        if a in position and b in position:
            bonds.append((position[a], position[b]))
    return atoms, bonds


def _display(locants: tuple[str, ...]) -> str:
    for loc in locants:
        if _NUMERIC.match(loc):
            return loc
    for loc in locants:
        if _ELEMENT.match(loc):
            return loc
    for loc in locants:
        if loc in _GREEK:
            return _GREEK[loc]
    rest = [loc for loc in locants if loc not in _POSITIONAL]
    return rest[0] if rest else "-"


# --------------------------------------------------------------------------- #
# 3. Matching and parts
# --------------------------------------------------------------------------- #


def _skeleton(elements: list[str], bonds: list[tuple[int, int]]) -> Chem.Mol:
    """Elements + connectivity only: every bond single, no charges, no H."""
    rw = Chem.RWMol()
    table = Chem.GetPeriodicTable()
    for el in elements:
        atom = Chem.Atom(table.GetAtomicNumber(el))
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for a, b in bonds:
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def _parts(atoms: list[_OpsinAtom], bonds: list[tuple[int, int]]) -> list[int | None]:
    """Part of each OPSIN atom: union bonded atoms while locants stay unique."""
    numeric = [{loc for loc in a.locants if _NUMERIC.match(loc)} or None for a in atoms]
    parent = list(range(len(atoms)))
    members: dict[int, set[str]] = {
        i: set(n) for i, n in enumerate(numeric) if n is not None
    }

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(
        bonds,
        key=lambda b: (
            abs(atoms[b[0]].opsin_id - atoms[b[1]].opsin_id),
            min(atoms[b[0]].opsin_id, atoms[b[1]].opsin_id),
        ),
    )
    for a, b in order:
        if numeric[a] is None or numeric[b] is None:
            continue
        ra, rb = find(a), find(b)
        if ra == rb or members[ra] & members[rb]:
            continue
        parent[rb] = ra
        members[ra] |= members.pop(rb)
    roots: dict[int, int] = {}
    out: list[int | None] = []
    # number parts in name order: by the smallest OPSIN id they contain
    first_id: dict[int, int] = {}
    for i, atom in enumerate(atoms):
        if numeric[i] is not None:
            r = find(i)
            first_id[r] = min(first_id.get(r, atom.opsin_id), atom.opsin_id)
    for k, r in enumerate(sorted(first_id, key=first_id.__getitem__)):
        roots[r] = k
    for i in range(len(atoms)):
        out.append(roots[find(i)] if numeric[i] is not None else None)
    return out


def number_atoms(mol: Chem.Mol, name: str, source: str, jar: Path) -> IupacNumbering:
    """Locants for every heavy atom of ``mol`` from its IUPAC ``name``."""
    atoms, bonds = parse_cml(opsin_cml(name, jar))
    if len(atoms) != mol.GetNumAtoms():
        msg = (
            f"the name has {len(atoms)} heavy atoms, the molecule "
            f"{mol.GetNumAtoms()} (PubChem's name is for another form)"
        )
        raise NumberingUnavailable(msg)
    theirs = _skeleton([a.element for a in atoms], bonds)
    ours = _skeleton(
        [a.GetSymbol() for a in mol.GetAtoms()],
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
    )
    match = ours.GetSubstructMatch(theirs)
    if len(match) != len(atoms):
        msg = "OPSIN's structure does not have this molecule's skeleton"
        raise NumberingUnavailable(msg)

    part_of = _parts(atoms, bonds)
    numbering = IupacNumbering(name=name, source=source)
    n_parts = max((p for p in part_of if p is not None), default=-1) + 1
    numbering.parts = [Part(k, []) for k in range(n_parts)]
    for pos, ours_idx in enumerate(match):
        a = atoms[pos]
        numbering.atoms[ours_idx] = AtomLocant(
            index=ours_idx,
            locant=_display(a.locants),
            locants=a.locants,
            part=part_of[pos],
        )
        k = part_of[pos]
        if k is not None:
            numbering.parts[k].atoms.append(ours_idx)
    # part-less atoms (hydroxy O, amide N, carboxyl C) point at the atom they sit
    # on: a numbered atom, else (second pass) one that already points somewhere,
    # so "O@α@1" reads "O on the alpha carbon on C1". Never a cycle.
    for numbered_only in (True, False):
        for idx, loc in list(numbering.atoms.items()):
            if loc.part is not None or loc.attached_to:
                continue
            on = [
                n.GetIdx()
                for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if numbering.atoms[n.GetIdx()].part is not None
                or (
                    not numbered_only
                    and bool(numbering.atoms[n.GetIdx()].attached_to)
                    and idx not in numbering.atoms[n.GetIdx()].attached_to
                )
            ]
            if on:
                numbering.atoms[idx] = replace(loc, attached_to=tuple(on))
    for part in numbering.parts:
        part.atoms.sort()
        part.smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=part.atoms)
    return numbering


def iupac_numbering(
    rec: StandardizedCompound, working: Chem.Mol, client: PubChemClient
) -> IupacNumbering:
    """Name from PubChem, locants from OPSIN; raises NumberingUnavailable."""
    jar = find_opsin_jar()
    if jar is None or not jar.exists():
        msg = (
            "OPSIN jar not found: put opsin-cli-*-jar-with-dependencies.jar in the "
            "project root or data/tools/, or set MODEPHERE_OPSIN_JAR"
        )
        raise NumberingUnavailable(msg)
    name, source = pubchem_iupac_name(rec, working, client)
    return number_atoms(working, name, source, jar)
