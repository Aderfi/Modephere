"""Standardization: resolved structure -> registration + working forms (PLAN C2.1).

Steps, in order (each failure is a ``CompoundError`` at stage ``standardize``):

1. parse the SMILES;
2. choose the parent: the largest organic fragment (dev_Pgen behaviour), or the
   ChEMBL salt-list parent (``std`` extra);
3. reject what the force fields and feature definitions cannot handle: any
   element outside H B C N O F Si P S Cl Se Br I (metals; organometallics stay
   bonded to their metal, so they are caught here), no carbon (inorganics),
   more than ``max_heavy_atoms`` heavy atoms;
4. normalize functional groups and neutralize (the **neutral parent**);
5. **registration form**: canonical tautomer of the neutral parent, keeping
   its stereochemistry. Its InChIKey is the library-wide compound key. Used
   for identity only;
6. **working form**: the neutral parent in its input tautomer, with explicit
   (isotopic) hydrogens removed, re-parsed from its canonical SMILES. That
   SMILES fixes the atom order every downstream array uses;
7. **microspecies**: the working form protonated for pH 7.4 by rule
   (:func:`protonate`). Heavy-atom order is unchanged.

RDKit's canonical tautomer is never used for features: it is a canonical
representative, not the dominant tautomer (it turns oxindoles into
hydroxyindoles and deconjugates steroid enones) and would invent donors.
"""

from __future__ import annotations

from dataclasses import dataclass

import rdkit
from pydantic import PositiveInt
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.MolStandardize import rdMolStandardize

from modephere.core import CompoundError, ConfigurationError
from modephere.types import (
    BuildStage,
    FailureCategory,
    FragmentPolicy,
    FrozenModel,
    ProtonationMethod,
    ResolvedCompound,
    StandardizedCompound,
)

#: Elements MMFF94 / UFF and the feature definitions cover (atomic numbers).
ALLOWED_ELEMENTS: frozenset[int] = frozenset(
    {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
)
#: Metals accepted as *dropped* counterions only (Li Na K Rb Cs, Mg Ca Sr Ba).
#: Any other metal in any fragment means the metal is part of the drug
#: (metallocenes, platinum complexes, gadolinium chelates written as ions).
SALT_METALS: frozenset[int] = frozenset({3, 11, 19, 37, 55, 12, 20, 38, 56})
RDKIT_VERSION: str = rdkit.__version__


class StandardizeOptions(FrozenModel):
    """Standardization policy (PLAN Q7-Q9 defaults)."""

    fragment_policy: FragmentPolicy = FragmentPolicy.LARGEST_FRAGMENT
    protonation: ProtonationMethod = ProtonationMethod.RULE_BASED
    max_heavy_atoms: PositiveInt = 100


def _canonical_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """RDKit's canonical tautomer, without its default stereo removal.

    By default the enumerator erases sp3 and double-bond stereo at every atom
    a tautomer can touch (alpha-amino acids, alpha-carbonyl carbons), which
    gives enantiomers one InChIKey and merges them in the library. Kept, the
    key matches the standard InChIKey ChEMBL and PubChem report.
    """
    params = rdMolStandardize.CleanupParameters()
    # RDKit stubs type these flags as CleanupParameters; they are bools.
    params.tautomerRemoveSp3Stereo = False  # type: ignore[assignment]
    params.tautomerRemoveBondStereo = False  # type: ignore[assignment]
    return rdMolStandardize.TautomerEnumerator(params).Canonicalize(mol)


@dataclass(frozen=True)
class StandardizedMol:
    """A standardized record plus the RDKit molecules derived from it.

    ``working`` has the atom order of ``Chem.MolFromSmiles(record.working_smiles)``;
    ``microspecies`` has the same heavy atoms in the same order.
    """

    record: StandardizedCompound
    working: Chem.Mol
    microspecies: Chem.Mol


def _fail(category: FailureCategory, reason: str) -> CompoundError:
    return CompoundError(BuildStage.STANDARDIZE, category, reason)


# --------------------------------------------------------------------------- #
# Microspecies at pH 7.4 (port of dev_Pgen pharmacophore._microspecies)
# --------------------------------------------------------------------------- #

# Rule-based protonation at physiological pH (approximate pKa classes, like the
# defaults of Phase/Epik-style tools). Applied in place on a neutralised copy,
# so heavy-atom indices never change.
_ACID_SMARTS: tuple[tuple[str, str], ...] = (
    # (name, SMARTS whose FIRST atom is the one deprotonated)
    ("carboxylic acid", "[OX2H1]-[CX3]=O"),
    ("sulfonic/sulfinic acid", "[OX2H1]-[SX4,SX3]=O"),
    ("phosphoric/phosphonic acid", "[OX2H1]-[PX4]=O"),
    ("vinylogous acid", "[OX2H1]-[#6]=[#6]-[#6]=O"),  # meloxicam, ascorbic acid
    ("4-hydroxycoumarin/2-pyrone", "[OX2H1]-c1:c:c(=O):o:c:c1"),  # warfarin
    ("tetrazole", "[nX3H1;r5;$(n1nnnc1),$(n1nncn1),$(n1ncnn1),$(n1cnnn1)]"),
    ("acyl sulfonamide", "[NX3H1](-[CX3]=O)-[SX4](=O)=O"),  # incl. sulfonylureas
    ("N-heteroaryl sulfonamide", "[NX3H1](-[SX4](=O)=O)-c:n"),  # sulfa drugs
)
_BASE_SMARTS: tuple[tuple[str, str], ...] = (
    # Amidines / guanidines: protonate the imine N. Acyl-, sulfonyl-, cyano-
    # (cimetidine) and N-aryl imine variants drop the pKa well below 7.4.
    (
        "amidine/guanidine",
        "[NX2;+0;!$(N-[!#6;!#1]);!$(N-C=[O,S]);!$(N-C#N);!$(N-a)]"
        "=[CX3;!$(C-N-C#N);!$(C-N-C=[O,S]);!$(C-N-S(=O)=O)]"
        "-[NX3;+0;!$(N-[!#6;!#1])]",
    ),
    # Aliphatic amines: every neighbour an sp3 carbon (no amide, aniline,
    # enamine, sulfonamide, hydrazine, N-oxide), not next to a CF2/CF3 group.
    (
        "aliphatic amine",
        "[NX3;+0;$(N-[#6]);!$(N~[!#6;!#1]);!$(N-[#6;!X4]);"
        "!$(N-[CX4](F)F);!$(N-[CX4]-[CX4](F)(F)F)]",
    ),
)
_ACID_PATTERNS = tuple((n, Chem.MolFromSmarts(s)) for n, s in _ACID_SMARTS)
_BASE_PATTERNS = tuple((n, Chem.MolFromSmarts(s)) for n, s in _BASE_SMARTS)
_NEAR_CHARGE = 3  # a second base within 3 bonds of a cation stays neutral
_UNCHARGER = rdMolStandardize.Uncharger()


def _set_charge(atom: Chem.Atom, charge: int, h_delta: int) -> None:
    hs = atom.GetTotalNumHs()
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(max(0, hs + h_delta))
    atom.SetFormalCharge(charge)


def protonate(mol: Chem.Mol) -> Chem.Mol:
    """Copy of ``mol`` in its dominant protonation state at pH 7.4.

    Neutralise first (so a SMILES written as a salt or zwitterion gives the same
    result as the neutral one), then deprotonate acids with pKa < ~6.5 and
    protonate bases with pKa > ~8.5. Of bases within ``_NEAR_CHARGE`` bonds of
    an existing cation (piperazine, ethylenediamine) only the first is
    protonated: the second pKa drops by several units. Atom order is unchanged.
    Gasteiger charges are computed on the result.
    """
    neutral = _UNCHARGER.uncharge(Chem.Mol(mol))
    if neutral.GetNumAtoms() != mol.GetNumAtoms():  # never expected; stay safe
        neutral = Chem.Mol(mol)
    rw = Chem.RWMol(neutral)

    acids: set[int] = set()
    centres: set[int] = set()  # one deprotonation per acid centre (P(=O)(OH)2)
    for _, patt in _ACID_PATTERNS:
        for match in rw.GetSubstructMatches(patt):
            atom_idx = match[0]
            centre = match[1] if len(match) > 1 else atom_idx
            if atom_idx in acids or centre in centres:
                continue
            acids.add(atom_idx)
            centres.add(centre)

    candidates: list[tuple[int, int, int]] = []  # (rule rank, heavy degree, idx)
    for rank, (_, patt) in enumerate(_BASE_PATTERNS):
        for match in rw.GetSubstructMatches(patt):
            atom = rw.GetAtomWithIdx(match[0])
            candidates.append((rank, atom.GetDegree(), match[0]))

    for idx in acids:
        _set_charge(rw.GetAtomWithIdx(idx), -1, -1)

    dist = Chem.GetDistanceMatrix(rw)
    cations = [a.GetIdx() for a in rw.GetAtoms() if a.GetFormalCharge() > 0]
    for _, _, idx in sorted(set(candidates)):
        if idx in cations or any(dist[idx, c] <= _NEAR_CHARGE for c in cations):
            continue
        _set_charge(rw.GetAtomWithIdx(idx), +1, +1)
        cations.append(idx)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except ValueError, RuntimeError:
        out = neutral
    rdPartialCharges.ComputeGasteigerCharges(out)
    return out


# --------------------------------------------------------------------------- #
# Parent, checks, forms
# --------------------------------------------------------------------------- #

_LARGEST = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def _parent(mol: Chem.Mol, policy: FragmentPolicy) -> Chem.Mol:
    if policy is FragmentPolicy.LARGEST_FRAGMENT:
        largest: Chem.Mol = _LARGEST.choose(mol)
        return largest
    try:
        from chembl_structure_pipeline import standardizer
    except ImportError as e:
        msg = "FragmentPolicy.CHEMBL_PARENT needs the 'std' extra (chembl_structure_pipeline)"
        raise ConfigurationError(msg) from e
    parent: Chem.Mol = standardizer.get_parent_mol(mol)[0]
    return parent


def _check_input(mol: Chem.Mol) -> None:
    metals = sorted(
        {
            a.GetSymbol()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() not in ALLOWED_ELEMENTS | SALT_METALS
        }
    )
    if metals:
        raise _fail(
            FailureCategory.UNSUPPORTED_CHEMISTRY,
            f"contains {metals}: not a counterion that can be dropped",
        )


def _check_chemistry(mol: Chem.Mol, max_heavy_atoms: int) -> None:
    unsupported = sorted(
        {
            a.GetSymbol()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() not in ALLOWED_ELEMENTS
        }
    )
    if unsupported:
        raise _fail(
            FailureCategory.UNSUPPORTED_CHEMISTRY,
            f"unsupported elements {unsupported} (metals / exotic atoms)",
        )
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        raise _fail(FailureCategory.UNSUPPORTED_CHEMISTRY, "no carbon: inorganic")
    heavy = mol.GetNumHeavyAtoms()
    if heavy > max_heavy_atoms:
        raise _fail(
            FailureCategory.TOO_LARGE,
            f"{heavy} heavy atoms exceeds the cap of {max_heavy_atoms}",
        )


def _without_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    params = Chem.RemoveHsParameters()
    # deuterium: geometry and features unchanged. (RDKit's stub mistypes the flag.)
    params.removeIsotopes = True  # type: ignore[assignment]
    return Chem.RemoveHs(mol, params)


def working_mols(record: StandardizedCompound) -> tuple[Chem.Mol, Chem.Mol]:
    """Rebuild ``(working, microspecies)`` from a stored record (same atom order)."""
    working = Chem.MolFromSmiles(record.working_smiles)
    if working is None:
        msg = f"stored working SMILES no longer parses: {record.working_smiles}"
        raise ConfigurationError(msg)
    return working, protonate(working)


def standardize(
    resolved: ResolvedCompound, options: StandardizeOptions | None = None
) -> StandardizedMol:
    """Standardize one resolved compound; raises ``CompoundError`` on failure."""
    opts = options or StandardizeOptions()
    mol = Chem.MolFromSmiles(resolved.smiles)
    if mol is None:
        raise _fail(
            FailureCategory.INVALID_SMILES, f"RDKit could not parse {resolved.smiles}"
        )
    if mol.GetNumAtoms() == 0:
        raise _fail(
            FailureCategory.EMPTY_MOLECULE, f"parsed to zero atoms: {resolved.smiles}"
        )

    _check_input(mol)
    n_frags = len(Chem.GetMolFrags(mol))
    parent = _parent(mol, opts.fragment_policy)
    _check_chemistry(parent, opts.max_heavy_atoms)

    try:
        neutral = _UNCHARGER.uncharge(rdMolStandardize.Normalize(parent))
        registration = _canonical_tautomer(neutral)
        working_smiles = Chem.MolToSmiles(_without_hydrogens(neutral))
    except (RuntimeError, ValueError) as e:
        raise _fail(FailureCategory.STANDARDIZATION_ERROR, str(e)) from e

    inchikey = Chem.MolToInchiKey(registration)
    working = Chem.MolFromSmiles(working_smiles)
    if not inchikey or working is None:
        raise _fail(
            FailureCategory.STANDARDIZATION_ERROR,
            f"no InChIKey / unparseable working form for {resolved.smiles}",
        )
    microspecies = protonate(working)
    record = StandardizedCompound(
        resolved=resolved,
        inchikey=inchikey,
        registration_smiles=Chem.MolToSmiles(registration),
        working_smiles=working_smiles,
        microspecies_smiles=Chem.MolToSmiles(microspecies),
        fragment_policy=opts.fragment_policy,
        protonation=opts.protonation,
        fragments_removed=n_frags - len(Chem.GetMolFrags(parent)),
        num_heavy_atoms=working.GetNumAtoms(),
        formal_charge=Chem.GetFormalCharge(microspecies),
        toolkit_version=RDKIT_VERSION,
    )
    return StandardizedMol(record=record, working=working, microspecies=microspecies)


__all__ = [
    "ALLOWED_ELEMENTS",
    "RDKIT_VERSION",
    "SALT_METALS",
    "StandardizeOptions",
    "StandardizedMol",
    "protonate",
    "standardize",
    "working_mols",
]
