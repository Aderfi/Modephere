"""Exchange formats: conformer ensembles as SDF (PLAN Q14, optional export)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rdkit import Chem
from rdkit.Geometry import Point3D

from modephere.core import StorageError
from modephere.library.standardize import working_mols

if TYPE_CHECKING:
    from pathlib import Path

    from modephere.types import CompoundArtifacts


def write_sdf(art: CompoundArtifacts, path: Path) -> int:
    """Write every conformer of ``art`` to ``path``; returns the count written.

    The molecule is rebuilt from the stored working SMILES (same atom order),
    protonated and given hydrogens; each record's stereo is re-read from its
    own 3D coordinates, so enumerated stereoisomers are written correctly.
    """
    ensemble = art.conformers
    if ensemble is None:
        msg = f"{art.inchikey} was built 2D-only: no conformers to export"
        raise StorageError(msg)
    _, micro = working_mols(art.compound)
    mol = Chem.AddHs(micro)
    if mol.GetNumAtoms() != ensemble.num_atoms:
        msg = (
            f"{art.inchikey}: rebuilt molecule has {mol.GetNumAtoms()} atoms, "
            f"ensemble {ensemble.num_atoms} (stored with another RDKit?)"
        )
        raise StorageError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as writer:
        for record, xyz in zip(ensemble.conformers, ensemble.positions, strict=True):
            copy = Chem.Mol(mol)
            conf = Chem.Conformer(copy.GetNumAtoms())
            for i, (x, y, z) in enumerate(xyz.tolist()):
                conf.SetAtomPosition(i, Point3D(x, y, z))
            copy.RemoveAllConformers()
            copy.AddConformer(conf, assignId=True)
            Chem.AssignStereochemistryFrom3D(copy)
            copy.SetProp("_Name", f"{art.inchikey}_{record.conformer_id}")
            copy.SetProp("inchikey", art.inchikey)
            copy.SetIntProp("conformer_id", record.conformer_id)
            copy.SetIntProp("stereoisomer", record.stereoisomer)
            copy.SetProp("force_field", str(ensemble.force_field))
            if record.energy is not None:
                copy.SetDoubleProp("energy_kcal_mol", record.energy)
            writer.write(copy)
    return ensemble.num_conformers


__all__ = ["write_sdf"]
