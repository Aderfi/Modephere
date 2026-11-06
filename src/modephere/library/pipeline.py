"""Per-compound pipeline: one standardized compound -> :class:`CompoundArtifacts`.

The builder runs this for every compound of a catalog; it is also the
shortest path from a structure to every representation, without a library on
disk::

    >>> from modephere.library import build_compound
    >>> art = build_compound("CC(=O)Oc1ccccc1C(=O)O")      # aspirin, offline
    >>> art.molecule.x.shape, art.pharmacophore.num_nodes, art.conformers.num_conformers

Stages after standardization (each failure is a ``CompoundError``):

* molecule graph of the working form (``molecule/v1``);
* perception on the pH 7.4 microspecies, then the pharmacophore graph
  (``pharmacophore/v1``), the 2D feature set and the Gobbi fingerprint. A
  compound without any feature fails (``no_pharmacophore_features``), as in
  dev_Pgen;
* unless 2D-only: conformers, 3D feature points per conformer and the 3D
  triplet fingerprint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from modephere.core import CompoundError
from modephere.library.conformers import ConformerSet, generate_conformers
from modephere.library.molecules import molecule_graph
from modephere.library.pharmacophore.features import perceive
from modephere.library.pharmacophore.fingerprints import gobbi_pharm2d, triplet_pharm3d
from modephere.library.pharmacophore.graph2d import pharmacophore_graph
from modephere.library.pharmacophore.points3d import place_features
from modephere.library.standardize import StandardizeOptions, standardize
from modephere.types import (
    BuildStage,
    CompoundArtifacts,
    ConformerParams,
    FailureCategory,
    FrozenModel,
    Pharmacophore2D,
    SmilesQuery,
)

if TYPE_CHECKING:
    from modephere.library.ingest.resolve import Resolver
    from modephere.library.molecules import FeatureSaturation
    from modephere.library.standardize import StandardizedMol
    from modephere.types import CompoundQuery


class PipelineOptions(FrozenModel):
    """What to build for each compound."""

    standardize: StandardizeOptions = Field(default_factory=StandardizeOptions)
    conformers: ConformerParams = Field(default_factory=ConformerParams)
    build_3d: bool = Field(default=True, description="False: 2D-only library.")


def build_artifacts(
    std: StandardizedMol,
    options: PipelineOptions | None = None,
    *,
    saturation: FeatureSaturation | None = None,
    conformers: ConformerSet | CompoundError | None = None,
) -> CompoundArtifacts:
    """Every representation of one standardized compound.

    ``conformers``: the compound's result from
    :func:`~modephere.library.conformers.generate_conformers_batch` when the
    caller batched the conformer stage (a ``CompoundError`` is raised here, after
    the earlier stages, as if it had happened here); None generates them.
    """
    opts = options or PipelineOptions()
    record = std.record
    molecule = molecule_graph(std.working, saturation=saturation)

    features = perceive(std.microspecies)
    if not features:
        raise CompoundError(
            BuildStage.PHARMACOPHORE,
            FailureCategory.NO_PHARMACOPHORE_FEATURES,
            f"no pharmacophore feature in {record.working_smiles}",
        )
    p2d = Pharmacophore2D(
        inchikey=record.inchikey,
        num_heavy_atoms=record.num_heavy_atoms,
        features=features,
    )

    ensemble = p3d = fp3d = None
    if opts.build_3d:
        if isinstance(conformers, CompoundError):
            raise conformers
        cset = conformers or generate_conformers(
            std.microspecies, record.inchikey, opts.conformers
        )
        ensemble = cset.ensemble
        p3d = place_features(
            cset.mol,
            p2d.features,
            inchikey=record.inchikey,
            num_heavy_atoms=record.num_heavy_atoms,
        )
        fp3d = triplet_pharm3d(p3d)

    return CompoundArtifacts(
        compound=record,
        molecule=molecule,
        pharmacophore=pharmacophore_graph(std.microspecies, p2d.features),
        pharmacophore_2d=p2d,
        pharm2d_fingerprint=gobbi_pharm2d(std.microspecies),
        conformers=ensemble,
        pharmacophores_3d=p3d or (),
        pharm3d_fingerprint=fp3d,
    )


def build_compound(
    query: CompoundQuery | str,
    options: PipelineOptions | None = None,
    *,
    resolver: Resolver | None = None,
) -> CompoundArtifacts:
    """Resolve, standardize and build one compound (a bare string is a SMILES).

    CID / name queries need a ``resolver`` with a PubChem client.
    """
    from modephere.library.ingest.resolve import Resolver as _Resolver

    if isinstance(query, str):
        query = SmilesQuery(smiles=query)
    resolved = (resolver or _Resolver(None)).resolve(query)
    opts = options or PipelineOptions()
    return build_artifacts(standardize(resolved, opts.standardize), opts)


__all__ = ["PipelineOptions", "build_artifacts", "build_compound"]
