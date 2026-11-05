"""Conformer generation: microspecies -> :class:`~modephere.types.ConformerEnsemble`.

Pipeline (PLAN C2.2), every knob in :class:`~modephere.types.ConformerParams`:

1. stereoisomers: unassigned stereocentres are enumerated (``ENUMERATE``, up
   to ``max_stereoisomers``) or left to the embedder (``RANDOM``);
2. per stereoisomer, add hydrogens and embed ``target_count(rotatable bonds)``
   conformers with ETKDGv3 (fixed seed, all threads, heavy-atom RMSD pruning),
   retrying from random coordinates if none embed;
3. optimise with MMFF94s, or the fallback force field (UFF) when MMFF cannot
   parametrize the molecule (boron, for instance);
4. keep conformers within ``energy_window`` of the global minimum;
5. drop near-duplicates after optimisation: in energy order, a conformer is
   kept only if its symmetry-aware heavy-atom RMSD (``GetBestRMS``) to every
   kept conformer of the same stereoisomer exceeds ``prune_rms_threshold``.

The ensemble is sorted by energy (conformer ``0`` is the lowest) and its atom
order is the microspecies' after ``AddHs``: heavy atoms first, so
``positions[:, :num_heavy_atoms]`` lines up with the 2D graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolAlign, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

from modephere.core import CompoundError
from modephere.types import (
    BuildStage,
    ConformerEnsemble,
    ConformerParams,
    ConformerRecord,
    FailureCategory,
    ForceField,
    StereoPolicy,
)

logger = logging.getLogger(__name__)

_MAX_RMS_MATCHES = 10_000  # cap on symmetry permutations per RMSD
_UNCHARGER = rdMolStandardize.Uncharger()


@dataclass(frozen=True)
class ConformerSet:
    """An ensemble plus the RDKit molecule holding its coordinates.

    ``mol`` has explicit hydrogens and one conformer per ensemble entry, with
    conformer id ``i`` = ensemble position ``i``. Its stereo tags are those of
    the first stereoisomer; use the coordinates (or
    ``Chem.AssignStereochemistryFrom3D``) for the others.
    """

    ensemble: ConformerEnsemble
    mol: Chem.Mol


@dataclass
class _Candidate:
    stereoisomer: int
    mol: Chem.Mol  # heavy+H molecule of that stereoisomer
    conf_id: int
    energy: float | None
    converged: bool


def _fail(category: FailureCategory, reason: str) -> CompoundError:
    return CompoundError(BuildStage.CONFORMERS, category, reason)


def _with_stereo_of(target: Chem.Mol, source: Chem.Mol) -> Chem.Mol:
    """Copy of ``target`` carrying ``source``'s atom and bond stereo (same atoms)."""
    out = Chem.RWMol(target)
    for atom, src in zip(out.GetAtoms(), source.GetAtoms(), strict=True):
        atom.SetChiralTag(src.GetChiralTag())
    for bond, src in zip(out.GetBonds(), source.GetBonds(), strict=True):
        if src.GetStereo() != Chem.BondStereo.STEREONONE:
            bond.SetStereoAtoms(*src.GetStereoAtoms())
            bond.SetStereo(src.GetStereo())
    mol = out.GetMol()
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def _stereoisomers(mol: Chem.Mol, params: ConformerParams) -> list[Chem.Mol]:
    """Stereoisomers of the microspecies, enumerated on its neutral form.

    A protonated amine (N+-H) is a stereocentre of the microspecies only:
    it inverts through deprotonation in water, so enumerating it would double
    every ensemble with interconverting "isomers". The neutral form has the
    same atoms in the same order, so its assigned stereo is copied back.
    """
    if params.stereo_policy is StereoPolicy.RANDOM:
        return [Chem.Mol(mol)]
    neutral = _UNCHARGER.uncharge(Chem.Mol(mol))
    if neutral.GetNumAtoms() != mol.GetNumAtoms():  # never expected; stay safe
        neutral = Chem.Mol(mol)
    options = StereoEnumerationOptions(
        onlyUnassigned=True,
        maxIsomers=params.max_stereoisomers,
        unique=True,
        tryEmbedding=False,
        rand=params.random_seed,
    )
    isomers = [
        _with_stereo_of(mol, iso)
        for iso in EnumerateStereoisomers(neutral, options=options)
    ]
    # Deterministic order whatever the enumerator yields.
    isomers.sort(key=Chem.MolToSmiles)
    return isomers or [Chem.Mol(mol)]


def _embed(mol_h: Chem.Mol, params: ConformerParams, count: int) -> list[int]:
    ps = getattr(rdDistGeom, params.method.value)()
    ps.randomSeed = params.random_seed
    ps.numThreads = params.num_threads
    ps.pruneRmsThresh = params.prune_rms_threshold
    ps.onlyHeavyAtomsForRMS = True
    ps.useSymmetryForPruning = True
    ids = list(rdDistGeom.EmbedMultipleConfs(mol_h, numConfs=count, params=ps))
    if not ids and params.use_random_coords_fallback:
        ps.useRandomCoords = True
        ids = list(rdDistGeom.EmbedMultipleConfs(mol_h, numConfs=count, params=ps))
    return ids


#: Elements whose only UFF type (Rappe et al. 1992) is sp3: Se3+2, Te3+2.
#: RDKit's UFF typer derives the type from the hybridization, so selenium in a
#: conjugated ring (ebselen, selenophenes) asks for "Se2+2", which UFF lacks,
#: and the molecule cannot be parametrized at all.
_UFF_SP3_ONLY = frozenset({34, 52})


def _uff_view(mol_h: Chem.Mol) -> Chem.Mol:
    """``mol_h``, or a copy typed for UFF: sp3-only elements retyped as sp3.

    Only the force field sees the copy; the bonded geometry still comes from the
    ring and its sp2 neighbours (ebselen: C-Se-N 86 degrees, planar ring).
    UFF bonds to Se in such rings come out ~0.07 A short (aromatic bond order).
    """
    retype = [
        a.GetIdx()
        for a in mol_h.GetAtoms()
        if a.GetAtomicNum() in _UFF_SP3_ONLY
        and a.GetHybridization() != Chem.HybridizationType.SP3
    ]
    if not retype:
        return mol_h
    view = Chem.Mol(mol_h)
    for idx in retype:
        view.GetAtomWithIdx(idx).SetHybridization(Chem.HybridizationType.SP3)
    return view


def _resolve_force_field(mol_h: Chem.Mol, params: ConformerParams) -> ForceField:
    order = [params.force_field]
    if params.fallback_force_field is not None:
        order.append(params.fallback_force_field)
    for ff in order:
        if ff is ForceField.NONE:
            return ff
        if ff in (ForceField.MMFF94, ForceField.MMFF94S):
            if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol_h):
                return ff
        elif rdForceFieldHelpers.UFFHasAllMoleculeParams(_uff_view(mol_h)):
            return ff
    raise _fail(
        FailureCategory.FORCEFIELD_FAILED,
        f"no parameters in {[str(f) for f in order]}",
    )


def _optimise(
    mol_h: Chem.Mol, ff: ForceField, params: ConformerParams
) -> list[tuple[bool, float | None]]:
    if ff is ForceField.NONE:
        return [(True, None)] * mol_h.GetNumConformers()
    if ff is ForceField.UFF:
        view = _uff_view(mol_h)
        results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
            view, numThreads=params.num_threads, maxIters=params.max_iterations
        )
        if view is not mol_h:  # optimised on the retyped copy: copy coordinates back
            for conf in view.GetConformers():
                mol_h.GetConformer(conf.GetId()).SetPositions(conf.GetPositions())
    else:
        results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            mol_h,
            numThreads=params.num_threads,
            maxIters=params.max_iterations,
            mmffVariant=ff.value,
        )
    return [(not_converged == 0, float(energy)) for not_converged, energy in results]


def _prune(candidates: list[_Candidate], threshold: float) -> list[_Candidate]:
    """Greedy, energy-ordered RMSD pruning within each stereoisomer."""
    kept: list[_Candidate] = []
    heavy: dict[int, Chem.Mol] = {}
    for cand in candidates:
        if cand.stereoisomer not in heavy:
            heavy[cand.stereoisomer] = Chem.RemoveHs(cand.mol)
        ref_mol = heavy[cand.stereoisomer]
        duplicate = any(
            other.stereoisomer == cand.stereoisomer
            and rdMolAlign.GetBestRMS(
                ref_mol,
                ref_mol,
                prbId=cand.conf_id,
                refId=other.conf_id,
                maxMatches=_MAX_RMS_MATCHES,
            )
            <= threshold
            for other in kept
        )
        if not duplicate:
            kept.append(cand)
    return kept


def generate_conformers(
    microspecies: Chem.Mol, inchikey: str, params: ConformerParams | None = None
) -> ConformerSet:
    """Embed, optimise and prune; raises ``CompoundError`` (stage conformers)."""
    params = params or ConformerParams()
    isomers = _stereoisomers(microspecies, params)
    count = params.target_count(rdMolDescriptors.CalcNumRotatableBonds(microspecies))

    candidates: list[_Candidate] = []
    force_field: ForceField | None = None
    for s, isomer in enumerate(isomers):
        mol_h = Chem.AddHs(isomer)
        ids = _embed(mol_h, params, count)
        if not ids:
            logger.debug("%s: stereoisomer %d did not embed", inchikey, s)
            continue
        ff = _resolve_force_field(mol_h, params)
        if force_field is None:
            force_field = ff
        elif ff is not force_field:  # same constitution: never expected
            msg = f"stereoisomers need different force fields ({force_field}, {ff})"
            raise _fail(FailureCategory.FORCEFIELD_FAILED, msg)
        for conf_id, (converged, energy) in zip(
            ids, _optimise(mol_h, ff, params), strict=True
        ):
            candidates.append(_Candidate(s, mol_h, conf_id, energy, converged))

    if not candidates or force_field is None:
        raise _fail(
            FailureCategory.EMBEDDING_FAILED,
            f"no conformer embedded ({len(isomers)} stereoisomer(s), {count} attempts each)",
        )

    if force_field is not ForceField.NONE:
        candidates.sort(key=lambda c: (c.energy, c.stereoisomer, c.conf_id))
        if params.energy_window is not None:
            floor = candidates[0].energy or 0.0
            candidates = [
                c
                for c in candidates
                if (c.energy or 0.0) - floor <= params.energy_window
            ]
    candidates = _prune(candidates, params.prune_rms_threshold)

    out = Chem.Mol(candidates[0].mol)
    out.RemoveAllConformers()
    positions = []
    for i, cand in enumerate(candidates):
        conf = Chem.Conformer(cand.mol.GetConformer(cand.conf_id))
        conf.SetId(i)
        out.AddConformer(conf, assignId=False)
        positions.append(conf.GetPositions())

    ensemble = ConformerEnsemble(
        inchikey=inchikey,
        num_atoms=out.GetNumAtoms(),
        num_heavy_atoms=microspecies.GetNumAtoms(),
        force_field=force_field,
        params=params,
        conformers=tuple(
            ConformerRecord(
                conformer_id=i,
                stereoisomer=c.stereoisomer,
                energy=c.energy,
                converged=c.converged,
            )
            for i, c in enumerate(candidates)
        ),
        positions=np.asarray(positions, dtype=np.float32),
    )
    return ConformerSet(ensemble=ensemble, mol=out)


__all__ = ["ConformerSet", "generate_conformers"]
