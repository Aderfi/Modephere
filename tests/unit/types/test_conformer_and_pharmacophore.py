"""Conformer params / ensembles and 2D / 3D pharmacophore records."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from modephere.types import (
    ConformerEnsemble,
    ConformerParams,
    ForceField,
    Pharmacophore2D,
    Pharmacophore3D,
    PharmacophoreFamily,
    PharmacophoreFeature,
    PharmacophorePoint,
)

from .conftest import ASPIRIN_KEY, make_ensemble

DON, ACC, ARO, HYD, NEG = (
    PharmacophoreFamily.DONOR,
    PharmacophoreFamily.ACCEPTOR,
    PharmacophoreFamily.AROMATIC,
    PharmacophoreFamily.HYDROPHOBE,
    PharmacophoreFamily.NEG_IONIZABLE,
)


class TestConformerParams:
    def test_plan_defaults(self) -> None:
        p = ConformerParams()
        assert (p.min_conformers, p.max_conformers) == (10, 50)
        assert p.prune_rms_threshold == 0.5
        assert p.energy_window == 10.0
        assert p.force_field is ForceField.MMFF94S
        assert p.fallback_force_field is ForceField.UFF
        assert p.max_stereoisomers == 8

    @pytest.mark.parametrize(
        ("rotors", "expected"), [(0, 10), (3, 25), (8, 50), (40, 50)]
    )
    def test_count_scales_with_rotatable_bonds(
        self, rotors: int, expected: int
    ) -> None:
        assert ConformerParams().target_count(rotors) == expected

    def test_fixed_count_overrides_scaling(self) -> None:
        assert ConformerParams(num_conformers=1).target_count(12) == 1

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"min_conformers": 60}, "min_conformers"),
            ({"fallback_force_field": "MMFF94s"}, "repeats"),
            ({"fallback_force_field": "none"}, "not 'none'"),
            ({"force_field": "none"}, "no energies"),
            ({"random_seed": -1}, "greater than or equal"),
            ({"energy_window": 0.0}, "greater than"),
        ],
    )
    def test_rejects(self, overrides: dict, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            ConformerParams(**overrides)

    def test_embed_only_is_consistent(self) -> None:
        p = ConformerParams(
            force_field="none", fallback_force_field=None, energy_window=None
        )
        assert p.force_field is ForceField.NONE


class TestConformerEnsemble:
    def test_valid(self, ensemble: ConformerEnsemble) -> None:
        assert ensemble.num_conformers == 3
        assert ensemble.positions.dtype == np.float32
        assert ensemble.lowest == 0
        energies = ensemble.energies
        assert energies is not None and energies.tolist() == [0.0, 1.5, 4.0]

    def test_shape_must_match(self) -> None:
        with pytest.raises(ValidationError, match="positions shape"):
            make_ensemble(positions=np.zeros((3, 20, 3)))

    def test_heavy_atoms_bounded(self) -> None:
        with pytest.raises(ValidationError, match="num_heavy_atoms"):
            make_ensemble(num_heavy_atoms=30)

    def test_energy_window_enforced(self) -> None:
        with pytest.raises(ValidationError, match="window"):
            make_ensemble((0.0, 12.0))

    def test_every_conformer_needs_energy(self) -> None:
        with pytest.raises(ValidationError, match="needs an energy"):
            make_ensemble((0.0, None))

    def test_force_field_must_come_from_params(self) -> None:
        with pytest.raises(ValidationError, match="neither primary"):
            make_ensemble(force_field="MMFF94")

    def test_fallback_force_field_accepted(self) -> None:
        assert make_ensemble(force_field="UFF").force_field is ForceField.UFF

    def test_embed_only_has_no_energies(self) -> None:
        params = ConformerParams(
            force_field="none", fallback_force_field=None, energy_window=None
        )
        e = make_ensemble((None, None), force_field="none", params=params)
        assert e.energies is None and e.lowest == 0
        with pytest.raises(ValidationError, match="no force field ran"):
            make_ensemble((0.0, None), force_field="none", params=params)

    def test_duplicate_ids_rejected(self, ensemble: ConformerEnsemble) -> None:
        dupes = (ensemble.conformers[0], ensemble.conformers[0], ensemble.conformers[1])
        with pytest.raises(ValidationError, match="duplicate conformer ids"):
            make_ensemble(conformers=dupes)

    def test_json_round_trip(self, ensemble: ConformerEnsemble) -> None:
        dumped = ensemble.model_dump_json()
        assert ConformerEnsemble.model_validate_json(dumped) == ensemble


class TestPharmacophore2D:
    def test_canonical_order(self) -> None:
        # Perception order is arbitrary; storage order is (first atom, family, ids).
        features = [
            PharmacophoreFeature(family=ARO, atom_ids=(4, 5, 6, 7, 8, 9)),
            PharmacophoreFeature(family=ACC, atom_ids=(2,)),
            PharmacophoreFeature(family=NEG, atom_ids=(11, 10, 12)),
            PharmacophoreFeature(family=DON, atom_ids=(2,)),
        ]
        p = Pharmacophore2D(inchikey=ASPIRIN_KEY, num_heavy_atoms=13, features=features)
        assert [(f.family, f.atom_ids) for f in p.features] == [
            (DON, (2,)),
            (ACC, (2,)),
            (ARO, (4, 5, 6, 7, 8, 9)),
            (NEG, (10, 11, 12)),
        ]

    def test_family_counts_complete(self) -> None:
        p = Pharmacophore2D(
            inchikey=ASPIRIN_KEY,
            num_heavy_atoms=13,
            features=[PharmacophoreFeature(family=ACC, atom_ids=(2,))],
        )
        assert p.family_counts[ACC] == 1
        assert set(p.family_counts) == set(PharmacophoreFamily)
        assert sum(p.family_counts.values()) == 1

    def test_empty_is_valid(self) -> None:
        assert Pharmacophore2D(inchikey=ASPIRIN_KEY, num_heavy_atoms=1).features == ()

    def test_duplicate_feature_rejected(self) -> None:
        f = PharmacophoreFeature(family=ACC, atom_ids=(2,))
        with pytest.raises(ValidationError, match="duplicate"):
            Pharmacophore2D(inchikey=ASPIRIN_KEY, num_heavy_atoms=13, features=[f, f])

    def test_atom_range_checked(self) -> None:
        with pytest.raises(ValidationError, match="indexes past"):
            Pharmacophore2D(
                inchikey=ASPIRIN_KEY,
                num_heavy_atoms=13,
                features=[PharmacophoreFeature(family=HYD, atom_ids=(0, 13))],
            )


class TestPharmacophore3D:
    def _points(self) -> list[PharmacophorePoint]:
        return [
            PharmacophorePoint(
                family=ARO,
                atom_ids=(4, 5, 6, 7, 8, 9),
                position=(0.0, 0.0, 0.0),
                direction=(0.0, 0.0, 2.0),
            ),
            PharmacophorePoint(
                family=NEG, atom_ids=(10, 11, 12), position=(2.5, 0.1, 0.0)
            ),
        ]

    def test_direction_normalised(self) -> None:
        aromatic = self._points()[0]
        assert aromatic.direction == (0.0, 0.0, 1.0)

    def test_isotropic_family_has_no_direction(self) -> None:
        with pytest.raises(ValidationError, match="isotropic"):
            PharmacophorePoint(
                family=HYD, atom_ids=(0,), position=(0, 0, 0), direction=(1, 0, 0)
            )

    def test_non_finite_position_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PharmacophorePoint(family=HYD, atom_ids=(0,), position=(float("inf"), 0, 0))

    def test_projects_onto_2d(self) -> None:
        p3 = Pharmacophore3D(
            inchikey=ASPIRIN_KEY,
            conformer_id=0,
            num_heavy_atoms=13,
            points=self._points(),
        )
        p2 = p3.to_2d()
        assert p2.features == p3.features
        assert all(type(f) is PharmacophoreFeature for f in p2.features)
        assert p3.family_counts == p2.family_counts

    def test_points_passed_to_2d_lose_geometry(self) -> None:
        p2 = Pharmacophore2D(
            inchikey=ASPIRIN_KEY, num_heavy_atoms=13, features=self._points()
        )
        assert all(type(f) is PharmacophoreFeature for f in p2.features)

    def test_same_features_across_conformers(self) -> None:
        a = Pharmacophore3D(
            inchikey=ASPIRIN_KEY,
            conformer_id=0,
            num_heavy_atoms=13,
            points=self._points(),
        )
        moved = [
            p.model_copy(update={"position": (p.position[0] + 1.0, *p.position[1:])})
            for p in self._points()
        ]
        b = Pharmacophore3D(
            inchikey=ASPIRIN_KEY, conformer_id=1, num_heavy_atoms=13, points=moved
        )
        assert a.features == b.features
        assert a.points != b.points
