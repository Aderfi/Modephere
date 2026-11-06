"""Standardization, pharmacophore perception, conformers, 3D points, fingerprints."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from modephere.core import CompoundError
from modephere.library import conformers
from modephere.library.conformers import (
    generate_conformers,
    generate_conformers_batch,
    nvmolkit_available,
)
from modephere.library.pharmacophore import (
    perceive,
    place_features,
    tanimoto,
    to_dense,
    triplet_pharm3d,
)
from modephere.library.pipeline import PipelineOptions, build_compound
from modephere.library.standardize import StandardizeOptions, protonate, standardize
from modephere.types import (
    TRIPLET_PHARM3D_V1,
    ConformerBackend,
    ConformerParams,
    FailureCategory,
    ForceField,
    FragmentPolicy,
    PharmacophoreFamily,
    ResolutionSource,
    ResolvedCompound,
    SmilesQuery,
    StereoPolicy,
)

from .conftest import ASPIRIN, ASPIRIN_KEY, FAST

DON, ACC, POS, NEG, ARO, HYD = list(PharmacophoreFamily)


def _std(smiles: str, **options: object):  # noqa: ANN202
    resolved = ResolvedCompound(
        query=SmilesQuery(smiles=smiles), smiles=smiles, source=ResolutionSource.LOCAL
    )
    return standardize(resolved, StandardizeOptions(**options))


def _families(smiles: str) -> list[str]:
    return sorted(
        f.family.value for f in perceive(protonate(Chem.MolFromSmiles(smiles)))
    )


def _fail(smiles: str, **options: object) -> FailureCategory:
    with pytest.raises(CompoundError) as exc:
        _std(smiles, **options)
    return exc.value.category


class TestStandardize:
    def test_salt_is_stripped(self) -> None:
        rec = _std("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]").record
        assert rec.inchikey == ASPIRIN_KEY
        assert rec.working_smiles == ASPIRIN
        assert rec.fragments_removed == 1
        assert rec.microspecies_smiles.endswith("C(=O)[O-]")
        assert rec.formal_charge == -1

    def test_chembl_parent_policy(self) -> None:
        rec = _std(
            "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", fragment_policy="chembl_parent"
        ).record
        assert rec.inchikey == ASPIRIN_KEY
        assert rec.fragment_policy is FragmentPolicy.CHEMBL_PARENT

    def test_zwitterion_is_neutral_working_form(self) -> None:
        rec = _std("[NH3+]CC([O-])=O").record
        assert rec.working_smiles == "NCC(=O)O"
        assert rec.microspecies_smiles == "[NH3+]CC(=O)[O-]"

    def test_working_form_keeps_the_input_tautomer(self) -> None:
        rec = _std("O=C1Cc2ccccc2N1").record  # oxindole
        assert rec.working_smiles == "O=C1Cc2ccccc2N1"
        assert rec.registration_smiles != rec.working_smiles  # canonical tautomer

    def test_tautomers_share_the_registration_key(self) -> None:
        assert (
            _std("Oc1ccccn1").record.inchikey == _std("O=c1cccc[nH]1").record.inchikey
        )

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"),  # L- / D-alanine
            ("C[C@H](C(=O)O)c1ccccc1", "C[C@@H](C(=O)O)c1ccccc1"),  # alpha-carbonyl
            ("C/C=C/C(=O)O", "C/C=C\\C(=O)O"),  # E / Z
        ],
    )
    def test_stereoisomers_keep_distinct_keys(self, left: str, right: str) -> None:
        # The canonical tautomer must not drop stereo next to tautomeric groups,
        # or enantiomers would collapse onto one library key.
        a, b = _std(left).record, _std(right).record
        assert a.inchikey != b.inchikey
        assert a.inchikey == Chem.MolToInchiKey(Chem.MolFromSmiles(left))

    def test_registration_keeps_the_stereo(self) -> None:
        # liothyronine: the key ChEMBL and PubChem give it (standard InChI)
        rec = _std("N[C@@H](Cc1cc(I)c(Oc2ccc(O)c(I)c2)c(I)c1)C(=O)O").record
        assert rec.inchikey == "AUYYCJSJGJYCDS-LBPRGKRZSA-N"
        assert "@" in rec.registration_smiles

    def test_isotopes_only_in_the_key(self) -> None:
        rec = _std("[2H]C([2H])([2H])Oc1ccccc1").record
        assert rec.working_smiles == "COc1ccccc1"
        assert rec.inchikey != _std("COc1ccccc1").record.inchikey
        assert rec.num_heavy_atoms == 8

    def test_atom_order_is_the_working_smiles(self) -> None:
        std = _std("OC(=O)c1ccccc1OC(C)=O")
        reparsed = Chem.MolFromSmiles(std.record.working_smiles)
        assert [a.GetSymbol() for a in std.working.GetAtoms()] == [
            a.GetSymbol() for a in reparsed.GetAtoms()
        ]
        assert [a.GetAtomicNum() for a in std.microspecies.GetAtoms()] == [
            a.GetAtomicNum() for a in std.working.GetAtoms()
        ]

    @pytest.mark.parametrize(
        ("smiles", "category"),
        [
            ("not_smiles", FailureCategory.INVALID_SMILES),
            ("N.N.Cl[Pt]Cl", FailureCategory.UNSUPPORTED_CHEMISTRY),  # cisplatin
            ("[cH-]1cccc1.[cH-]1cccc1.[Fe+2]", FailureCategory.UNSUPPORTED_CHEMISTRY),
            ("O=S(=O)(O)O", FailureCategory.UNSUPPORTED_CHEMISTRY),  # inorganic
            ("C" * 101, FailureCategory.TOO_LARGE),
        ],
    )
    def test_rejections(self, smiles: str, category: FailureCategory) -> None:
        assert _fail(smiles) is category

    def test_heavy_atom_cap_is_configurable(self) -> None:
        assert _fail("CCCCCC", max_heavy_atoms=5) is FailureCategory.TOO_LARGE

    def test_alkali_counterions_are_dropped(self) -> None:
        assert _std("CC(=O)[O-].[K+]").record.fragments_removed == 1


class TestPerception:
    """Chemistry expectations carried over from dev_Pgen's test suite."""

    def test_aspirin(self) -> None:
        assert _families(ASPIRIN) == sorted(
            ["Acceptor"] * 4 + ["Aromatic", "NegIonizable"]
        )

    def test_acid_written_any_way_gives_one_feature_set(self) -> None:
        ref = perceive(protonate(Chem.MolFromSmiles("CC(=O)O")))
        for smiles in ("CC(=O)[O-]", "CC(=O)[O-].[Na+]"):
            mol = Chem.MolFromSmiles(smiles)
            mol = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumAtoms())
            assert perceive(protonate(mol)) == ref

    @pytest.mark.parametrize(
        ("smiles", "family", "count"),
        [
            ("CCN(CC)CC", "PosIonizable", 1),  # protonated tertiary amine
            ("CN1CCNCC1", "PosIonizable", 1),  # piperazine protonates once
            ("CN(C)C(=N)N=C(N)N", "PosIonizable", 1),  # metformin
            ("CN=C(NC#N)NC", "PosIonizable", 0),  # cyanoguanidine stays neutral
            ("c1c[nH]cn1", "Donor", 1),  # imidazole: only the N-H
            ("c1ccc2ccccc2c1", "Aromatic", 2),  # one per fused ring
            ("CC(C)(C)C1CCCCC1", "Hydrophobe", 2),  # lumped groups
        ],
    )
    def test_family_counts(self, smiles: str, family: str, count: int) -> None:
        assert _families(smiles).count(family) == count

    def test_nitro_and_n_oxide_are_not_ions(self) -> None:
        for smiles in ("O=[N+]([O-])c1ccccc1", "[O-][n+]1ccccc1"):
            fams = _families(smiles)
            assert "PosIonizable" not in fams and "NegIonizable" not in fams


class TestConformers:
    def _micro(self, smiles: str) -> Chem.Mol:
        micro: Chem.Mol = protonate(Chem.MolFromSmiles(smiles))
        return micro

    def test_deterministic(self) -> None:
        # RDKit (the default) is reproducible from the seed; nvMolKit is not.
        params = ConformerParams(num_conformers=6)
        assert params.backend is ConformerBackend.RDKIT
        a = generate_conformers(self._micro("CCCCOc1ccccc1"), ASPIRIN_KEY, params)
        b = generate_conformers(self._micro("CCCCOc1ccccc1"), ASPIRIN_KEY, params)
        assert a.ensemble == b.ensemble

    def test_sorted_by_energy_within_window(self) -> None:
        e = generate_conformers(self._micro("CCCCCCO"), ASPIRIN_KEY).ensemble
        energies = e.energies
        assert energies is not None
        assert list(energies) == sorted(energies)
        assert energies.max() - energies.min() <= 10.0 + 1e-6
        assert 1 <= e.num_conformers <= 50

    def test_rmsd_pruning_merges_duplicates(self) -> None:
        rigid = generate_conformers(
            self._micro("c1ccccc1"), ASPIRIN_KEY, ConformerParams(num_conformers=10)
        )
        assert rigid.ensemble.num_conformers == 1  # benzene has one conformation

    def test_uff_fallback(self) -> None:
        e = generate_conformers(self._micro("OB(O)c1ccccc1"), ASPIRIN_KEY).ensemble
        assert e.force_field is ForceField.UFF

    @pytest.mark.parametrize(
        "smiles",
        [
            "O=C1c2ccccc2[Se]N1c1ccccc1",  # ebselen: Se in a conjugated ring
            "c1cc[se]c1",  # selenophene: aromatic Se
        ],
    )
    def test_sp2_selenium_uses_uff(self, smiles: str) -> None:
        # UFF has a single selenium type (Se3+2); RDKit would ask for "Se2+2".
        cset = generate_conformers(self._micro(smiles), ASPIRIN_KEY)
        e = cset.ensemble
        assert e.force_field is ForceField.UFF
        assert all(c.converged for c in e.conformers)
        se = next(a for a in cset.mol.GetAtoms() if a.GetAtomicNum() == 34)
        a, b = (n.GetIdx() for n in se.GetNeighbors())
        pos = e.positions[0]
        u, v = pos[a] - pos[se.GetIdx()], pos[b] - pos[se.GetIdx()]
        angle = np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
        assert 80 < angle < 95  # five-membered ring: ~86 degrees
        # the returned molecule keeps its own (sp2) perception
        assert se.GetHybridization() is Chem.HybridizationType.SP2

    def test_stereoisomers_enumerated(self) -> None:
        e = generate_conformers(self._micro("CC(O)CC"), ASPIRIN_KEY).ensemble
        assert {c.stereoisomer for c in e.conformers} == {0, 1}

    def test_specified_stereo_and_ammonium_not_enumerated(self) -> None:
        morphine = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        e = generate_conformers(self._micro(morphine), ASPIRIN_KEY).ensemble
        assert {c.stereoisomer for c in e.conformers} == {0}

    def test_random_stereo_policy(self) -> None:
        params = ConformerParams(stereo_policy=StereoPolicy.RANDOM)
        e = generate_conformers(self._micro("CC(O)CC"), ASPIRIN_KEY, params).ensemble
        assert {c.stereoisomer for c in e.conformers} == {0}

    def test_heavy_atoms_first(self) -> None:
        cset = generate_conformers(self._micro(ASPIRIN), ASPIRIN_KEY)
        atoms = list(cset.mol.GetAtoms())
        n = cset.ensemble.num_heavy_atoms
        assert all(a.GetAtomicNum() > 1 for a in atoms[:n])
        assert all(a.GetAtomicNum() == 1 for a in atoms[n:])


RDKIT = ConformerParams()
GPU = ConformerParams(backend=ConformerBackend.NVMOLKIT)
needs_gpu = pytest.mark.skipif(
    not nvmolkit_available(), reason="nvMolKit needs a CUDA GPU"
)


def _micro(smiles: str) -> Chem.Mol:
    micro: Chem.Mol = protonate(Chem.MolFromSmiles(smiles))
    return micro


class TestConformerBackends:
    @needs_gpu
    def test_nvmolkit_ensemble(self) -> None:
        cset = generate_conformers(_micro("CC(O)CCOc1ccccc1"), ASPIRIN_KEY, GPU)
        e = cset.ensemble
        assert e.params.backend is ConformerBackend.NVMOLKIT
        assert e.force_field is ForceField.MMFF94S
        assert {c.stereoisomer for c in e.conformers} == {0, 1}
        energies = e.energies
        assert energies is not None
        assert list(energies) == sorted(energies)
        # GPU energies are RDKit's MMFF94s on the returned coordinates
        props = AllChem.MMFFGetMoleculeProperties(cset.mol, mmffVariant="MMFF94s")
        for i, energy in enumerate(energies):
            ff = AllChem.MMFFGetMoleculeForceField(cset.mol, props, confId=i)
            assert ff.CalcEnergy() == pytest.approx(energy, abs=1e-3)

    @needs_gpu
    @pytest.mark.parametrize("smiles", ["OB(O)c1ccccc1", "O=C1c2ccccc2[Se]N1c1ccccc1"])
    def test_nvmolkit_uff_fallback(self, smiles: str) -> None:
        e = generate_conformers(_micro(smiles), ASPIRIN_KEY, GPU).ensemble
        assert e.params.backend is ConformerBackend.NVMOLKIT
        assert e.force_field is ForceField.UFF
        assert all(c.converged for c in e.conformers)

    @needs_gpu
    def test_batch_keeps_order_and_isolates_failures(self) -> None:
        params = GPU.model_copy(update={"fallback_force_field": None})  # no B in MMFF
        out = generate_conformers_batch(
            [
                (_micro("CCCCO"), ASPIRIN_KEY),
                (_micro("OB(O)c1ccccc1"), ASPIRIN_KEY),
                (_micro("c1ccccc1"), ASPIRIN_KEY),
            ],
            params,
        )
        assert isinstance(out[1], CompoundError)
        assert out[1].category is FailureCategory.FORCEFIELD_FAILED
        butanol, benzene = out[0], out[2]
        assert not isinstance(butanol, CompoundError)
        assert not isinstance(benzene, CompoundError)
        assert butanol.ensemble.num_atoms == 15
        assert benzene.ensemble.num_conformers == 1

    def test_no_gpu_falls_back_to_rdkit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(conformers, "nvmolkit_available", lambda: False)
        micro = _micro("CCCCOc1ccccc1")
        e = generate_conformers(micro, ASPIRIN_KEY, GPU).ensemble
        assert e.params.backend is ConformerBackend.RDKIT
        assert e == generate_conformers(micro, ASPIRIN_KEY, RDKIT).ensemble

    def test_nvmolkit_failure_reruns_on_rdkit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*_: object) -> None:
            msg = "simulated nvMolKit failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(conformers, "nvmolkit_available", lambda: True)
        monkeypatch.setattr(conformers, "_run_nvmolkit", broken)
        micro = _micro("CCCCOc1ccccc1")
        e = generate_conformers(micro, ASPIRIN_KEY, GPU).ensemble
        assert e.params.backend is ConformerBackend.RDKIT
        assert e == generate_conformers(micro, ASPIRIN_KEY, RDKIT).ensemble

    def test_over_256_atoms_runs_on_rdkit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(conformers, "nvmolkit_available", lambda: True)
        monkeypatch.setattr(conformers, "_run_nvmolkit", pytest.fail)
        job = conformers._job(_micro("C" * 90), ASPIRIN_KEY, GPU)
        assert job.mols[0].GetNumAtoms() == 272
        assert job.backend is ConformerBackend.RDKIT


def _rotation(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    rot: np.ndarray = q * np.sign(np.linalg.det(q))  # proper rotation
    return rot


class TestPoints3D:
    def test_one_pharmacophore_per_conformer(self, aspirin) -> None:  # noqa: ANN001
        assert len(aspirin.pharmacophores_3d) == aspirin.conformers.num_conformers
        for p in aspirin.pharmacophores_3d:
            assert p.features == aspirin.pharmacophore_2d.features

    def test_directions(self, aspirin) -> None:  # noqa: ANN001
        for point in aspirin.pharmacophores_3d[0].points:
            if point.family is ARO or point.family is ACC:
                assert point.direction is not None
                assert np.linalg.norm(point.direction) == pytest.approx(1.0)
            if point.family is NEG:
                assert point.direction is None

    def test_rigid_motion_equivariance(self) -> None:
        micro = protonate(Chem.MolFromSmiles(ASPIRIN))
        cset = generate_conformers(
            micro, ASPIRIN_KEY, ConformerParams(num_conformers=1)
        )
        features = perceive(micro)
        moved = Chem.Mol(cset.mol)
        rot, shift = _rotation(3), np.array([5.0, -2.0, 1.0])
        conf = moved.GetConformer(0)
        for i, xyz in enumerate(conf.GetPositions()):
            conf.SetAtomPosition(i, (rot @ xyz + shift).tolist())
        kw = {"inchikey": ASPIRIN_KEY, "num_heavy_atoms": micro.GetNumAtoms()}
        [a] = place_features(cset.mol, features, **kw)  # type: ignore[arg-type]
        [b] = place_features(moved, features, **kw)  # type: ignore[arg-type]
        for pa, pb in zip(a.points, b.points, strict=True):
            assert np.allclose(
                rot @ np.array(pa.position) + shift, pb.position, atol=1e-6
            )
            if pa.direction is not None and pb.direction is not None:
                assert np.allclose(
                    rot @ np.array(pa.direction), pb.direction, atol=1e-6
                )
        assert np.array_equal(triplet_pharm3d([a]), triplet_pharm3d([b]))


class TestFingerprints:
    def test_layout_and_order(self, aspirin) -> None:  # noqa: ANN001
        bits = aspirin.pharm3d_fingerprint
        assert bits.dtype == np.int64 and (np.diff(bits) > 0).all()
        assert bits.max() < TRIPLET_PHARM3D_V1.num_bits
        assert to_dense(bits, TRIPLET_PHARM3D_V1.num_bits).sum() == bits.size

    def test_similarity_ranks_analogues(self, aspirin) -> None:  # noqa: ANN001
        salicylic = build_compound("OC(=O)c1ccccc1O", FAST)
        hexane = build_compound("CCCCCCO", FAST)
        a = aspirin.pharm3d_fingerprint
        assert a is not None
        assert salicylic.pharm3d_fingerprint is not None
        assert hexane.pharm3d_fingerprint is not None
        assert tanimoto(a, salicylic.pharm3d_fingerprint) > tanimoto(
            a, hexane.pharm3d_fingerprint
        )
        assert tanimoto(a, a) == 1.0

    def test_2d_only_build(self) -> None:
        art = build_compound(ASPIRIN, PipelineOptions(build_3d=False))
        assert art.conformers is None and art.pharm3d_fingerprint is None
        assert art.pharm2d_fingerprint.size > 0

    def test_featureless_compound_fails(self) -> None:
        with pytest.raises(CompoundError) as exc:
            build_compound("C")
        assert exc.value.category is FailureCategory.NO_PHARMACOPHORE_FEATURES
