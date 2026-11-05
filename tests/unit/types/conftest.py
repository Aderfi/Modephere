"""Valid baseline objects; each test perturbs one field to hit one validator."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from modephere.types import (
    CidQuery,
    ConformerEnsemble,
    ConformerParams,
    ConformerRecord,
    FragmentPolicy,
    ProtonationMethod,
    ResolutionSource,
    ResolvedCompound,
    SmilesQuery,
    StandardizedCompound,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
ASPIRIN_ANION = "CC(=O)Oc1ccccc1C(=O)[O-]"
CAFFEINE_KEY = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
def resolved_local() -> ResolvedCompound:
    return ResolvedCompound(
        query=SmilesQuery(smiles=ASPIRIN), smiles=ASPIRIN, source=ResolutionSource.LOCAL
    )


@pytest.fixture
def resolved_pubchem() -> ResolvedCompound:
    return ResolvedCompound(
        query=CidQuery(cid=2244, label="row-1"),
        smiles=ASPIRIN,
        source=ResolutionSource.PUBCHEM,
        cid=2244,
        inchikey=ASPIRIN_KEY,
        title="Aspirin",
        retrieved_at=NOW,
    )


@pytest.fixture
def standardized(resolved_pubchem: ResolvedCompound) -> StandardizedCompound:
    return StandardizedCompound(
        resolved=resolved_pubchem,
        inchikey=ASPIRIN_KEY,
        registration_smiles=ASPIRIN,
        working_smiles=ASPIRIN,
        microspecies_smiles=ASPIRIN_ANION,
        fragment_policy=FragmentPolicy.LARGEST_FRAGMENT,
        protonation=ProtonationMethod.RULE_BASED,
        fragments_removed=0,
        num_heavy_atoms=13,
        formal_charge=-1,
        toolkit_version="2025.09.6",
    )


def make_ensemble(
    energies: tuple[float | None, ...] = (0.0, 1.5, 4.0),
    *,
    num_atoms: int = 21,
    num_heavy_atoms: int = 13,
    **overrides: object,
) -> ConformerEnsemble:
    rng = np.random.default_rng(0)
    fields: dict[str, object] = {
        "inchikey": ASPIRIN_KEY,
        "num_atoms": num_atoms,
        "num_heavy_atoms": num_heavy_atoms,
        "force_field": "MMFF94s",
        "params": ConformerParams(),
        "conformers": tuple(
            ConformerRecord(conformer_id=i, energy=e, converged=True)
            for i, e in enumerate(energies)
        ),
        "positions": rng.normal(size=(len(energies), num_atoms, 3)),
    }
    fields.update(overrides)
    return ConformerEnsemble(**fields)


@pytest.fixture
def ensemble() -> ConformerEnsemble:
    return make_ensemble()
