"""Pharmacophore feature perception (port of dev_Pgen ``_perceive``).

Features are perceived with RDKit's ``BaseFeatures.fdef`` on the pH 7.4
microspecies (:func:`modephere.library.standardize.protonate`) and normalised
to six non-overlapping families, so a chemical group is one feature however
its SMILES was written. Perception runs once per compound; the 2D graph, the
fingerprints and every conformer's 3D pharmacophore use its output.
"""

from __future__ import annotations

import os

from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolChemicalFeatures

from modephere.types import PharmacophoreFamily, PharmacophoreFeature

_FACTORY = rdMolChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)
_FAMILY_RANK = {f.value: f.one_hot_index for f in PharmacophoreFamily}
_ANIONIC_O = Chem.MolFromSmarts("[O-;!$([O-]-[N+]=O)]")

type RawFeature = tuple[str, tuple[int, ...]]


def perceive(microspecies: Chem.Mol) -> tuple[PharmacophoreFeature, ...]:
    """Normalised features of a pH 7.4 microspecies, in canonical order.

    Canonical order: first atom, then family one-hot index, then atom tuple
    (the order :class:`~modephere.types.Pharmacophore2D` stores).
    """
    raw: set[RawFeature] = set()
    for feat in _FACTORY.GetFeaturesForMol(microspecies):
        raw.add((feat.GetFamily(), tuple(sorted(feat.GetAtomIds()))))
    kept = sorted(
        set(_normalise_features(microspecies, raw)),
        key=lambda f: (f[1][0], _FAMILY_RANK.get(f[0], 99), f[1]),
    )
    return tuple(
        PharmacophoreFeature(family=PharmacophoreFamily(fam), atom_ids=ids)
        for fam, ids in kept
    )


def _normalise_features(mol: Chem.Mol, raw: set[RawFeature]) -> list[RawFeature]:
    """Map BaseFeatures.fdef output onto six non-overlapping families.

    * ``Donor`` only on atoms that carry a hydrogen (drops the neutral tertiary
      amine and the H-free aromatic N of the tautomer hedge);
    * ``Acceptor`` as perceived, plus anionic oxygens; ``Aromatic`` one per ring;
    * ``PosIonizable`` / ``NegIonizable`` one per charged atom of the
      microspecies (nitro / N-oxide zwitterions excluded); an oxyanion group
      also takes its acid centre and that centre's terminal O/S;
    * ``Hydrophobe``: a non-aromatic ``LumpedHydrophobe`` group is one feature;
      the remaining single-atom hydrophobes outside aromatic rings and lumped
      groups are merged into bonded chains;
    * ``ZnBinder`` is dropped (its atoms are already donors/acceptors/anions).
    """
    by_family: dict[str, list[tuple[int, ...]]] = {}
    for fam, ids in raw:
        by_family.setdefault(fam, []).append(ids)

    out: list[RawFeature] = []
    out += [
        ("Donor", ids)
        for ids in by_family.get("Donor", [])
        if all(mol.GetAtomWithIdx(i).GetTotalNumHs() > 0 for i in ids)
    ]
    out += [("Acceptor", ids) for ids in by_family.get("Acceptor", [])]
    # BaseFeatures.fdef means to cover anionic O, but that line is spelled
    # ``Atomtype`` and RDKit does not apply it.
    out += [("Acceptor", (m[0],)) for m in mol.GetSubstructMatches(_ANIONIC_O)]
    aromatic = by_family.get("Aromatic", [])
    out += [("Aromatic", ids) for ids in aromatic]
    out += _ionizable_features(mol)

    ring_sets = [frozenset(ids) for ids in aromatic]
    lumped = [
        frozenset(ids)
        for ids in by_family.get("LumpedHydrophobe", [])
        if not any(frozenset(ids) <= r for r in ring_sets)
    ]
    out += [("Hydrophobe", tuple(sorted(g))) for g in lumped]
    covered = ring_sets + lumped
    singles = {
        ids[0]
        for ids in by_family.get("Hydrophobe", [])
        if not any(ids[0] in g for g in covered)
    }
    out += [("Hydrophobe", chain) for chain in _bonded_components(mol, singles)]
    return out


def _ionizable_features(mol: Chem.Mol) -> list[RawFeature]:
    out: list[RawFeature] = []
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if not charge:
            continue
        nbrs = list(atom.GetNeighbors())
        if any(n.GetFormalCharge() * charge < 0 for n in nbrs):
            continue  # nitro, N-oxide: zwitterionic notation, not an ion
        if charge > 0:
            out.append(("PosIonizable", (atom.GetIdx(),)))
            continue
        group = {atom.GetIdx()}
        if atom.GetSymbol() in {"O", "S"}:  # oxyacid: take the centre and its =O
            for centre in nbrs:
                if centre.GetSymbol() in {"C", "S", "P"}:
                    group.add(centre.GetIdx())
                    group |= {
                        t.GetIdx()
                        for t in centre.GetNeighbors()
                        if t.GetSymbol() in {"O", "S"} and t.GetDegree() == 1
                    }
                    break
        out.append(("NegIonizable", tuple(sorted(group))))
    return out


def _bonded_components(mol: Chem.Mol, atoms: set[int]) -> list[tuple[int, ...]]:
    """Connected components of ``atoms`` through bonds between them."""
    left, comps = set(atoms), []
    while left:
        stack, comp = [left.pop()], set()
        while stack:
            i = stack.pop()
            comp.add(i)
            for n in mol.GetAtomWithIdx(i).GetNeighbors():
                if n.GetIdx() in left:
                    left.discard(n.GetIdx())
                    stack.append(n.GetIdx())
        comps.append(tuple(sorted(comp)))
    return comps


__all__ = ["perceive"]
