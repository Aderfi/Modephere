"""Modephere types: Pydantic schemas, annotated scalars and table schemas.

The leaf of the import graph: every other layer imports from here, and this
package imports no other Modephere module. RDKit is never imported here (it
belongs to the standardization layer).

The torch / PyG contracts in :mod:`modephere.types.tensors` need the ``ml``
extra and are loaded lazily on first access, so ``import modephere.types``
never imports torch::

    >>> from modephere.types import CidQuery, ConformerParams
    >>> from modephere.types import MoleculeTensors   # imports torch now
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from modephere.types.arrays import Float32Array, Float64Array, Int64Array, UInt8Array
from modephere.types.artifacts import CompoundArtifacts
from modephere.types.base import ArrayModel, FrozenModel, ModephereModel
from modephere.types.compound import ResolvedCompound, StandardizedCompound
from modephere.types.conformer import (
    ConformerEnsemble,
    ConformerParams,
    ConformerRecord,
)
from modephere.types.enums import (
    AmbiguityPolicy,
    BuildStage,
    ChemblMatch,
    ConformerBackend,
    EmbeddingMethod,
    FailureCategory,
    ForceField,
    FragmentPolicy,
    IdentifierKind,
    PharmacophoreFamily,
    ProtonationMethod,
    ResolutionSource,
    StereoPolicy,
)
from modephere.types.failures import FailureRecord
from modephere.types.frames import (
    COMPOUNDS_SCHEMA,
    CONFORMERS_SCHEMA,
    FAILURES_SCHEMA,
    FrameSchemaError,
    check_frame,
    compounds_frame,
    conformers_frame,
    failures_frame,
)
from modephere.types.graph import GraphArrays, MoleculeGraph, PharmacophoreGraph
from modephere.types.identifiers import (
    COMPOUND_QUERY,
    ChemblQuery,
    CidQuery,
    CompoundQuery,
    NameQuery,
    SmilesQuery,
    parse_query,
)
from modephere.types.layouts import (
    GOBBI_PHARM2D_V1,
    MOLECULE_GRAPH_V1,
    PHARMACOPHORE_GRAPH_V1,
    TRIPLET_PHARM3D_V1,
    FingerprintLayout,
    GraphLayout,
)
from modephere.types.pharmacophore import (
    Pharmacophore2D,
    Pharmacophore3D,
    PharmacophoreFeature,
    PharmacophorePoint,
)
from modephere.types.scalars import (
    Angstrom,
    AtomIds,
    AtomIndex,
    ChemblID,
    InChIKey,
    KcalPerMol,
    NonEmptyStr,
    PositiveKcalPerMol,
    PubChemCID,
    Seed,
    SmilesStr,
    UnitVec3,
    Vec3,
)

if TYPE_CHECKING:
    from modephere.types.tensors import (
        ConformerTensors,
        GraphTensors,
        HeteroTensors,
        MoleculeTensors,
        PharmacophoreData,
        PharmacophoreTensors,
    )

_LAZY_TENSORS = frozenset(
    {
        "ConformerTensors",
        "GraphTensors",
        "HeteroTensors",
        "MoleculeTensors",
        "PharmacophoreData",
        "PharmacophoreTensors",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_TENSORS:
        return getattr(importlib.import_module("modephere.types.tensors"), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "GOBBI_PHARM2D_V1",
    "TRIPLET_PHARM3D_V1",
    "CompoundArtifacts",
    "FingerprintLayout",
    "GraphArrays",
    "MoleculeGraph",
    "PharmacophoreGraph",
    "COMPOUNDS_SCHEMA",
    "COMPOUND_QUERY",
    "CONFORMERS_SCHEMA",
    "FAILURES_SCHEMA",
    "MOLECULE_GRAPH_V1",
    "PHARMACOPHORE_GRAPH_V1",
    "AmbiguityPolicy",
    "Angstrom",
    "ArrayModel",
    "AtomIds",
    "AtomIndex",
    "BuildStage",
    "ChemblID",
    "ChemblMatch",
    "ChemblQuery",
    "CidQuery",
    "CompoundQuery",
    "ConformerEnsemble",
    "ConformerParams",
    "ConformerBackend",
    "ConformerRecord",
    "ConformerTensors",
    "EmbeddingMethod",
    "FailureCategory",
    "FailureRecord",
    "Float32Array",
    "Float64Array",
    "ForceField",
    "FragmentPolicy",
    "FrameSchemaError",
    "FrozenModel",
    "GraphLayout",
    "GraphTensors",
    "HeteroTensors",
    "IdentifierKind",
    "InChIKey",
    "Int64Array",
    "KcalPerMol",
    "ModephereModel",
    "MoleculeTensors",
    "NameQuery",
    "NonEmptyStr",
    "Pharmacophore2D",
    "Pharmacophore3D",
    "PharmacophoreData",
    "PharmacophoreFamily",
    "PharmacophoreFeature",
    "PharmacophorePoint",
    "PharmacophoreTensors",
    "PositiveKcalPerMol",
    "ProtonationMethod",
    "PubChemCID",
    "ResolutionSource",
    "ResolvedCompound",
    "Seed",
    "SmilesQuery",
    "SmilesStr",
    "StandardizedCompound",
    "StereoPolicy",
    "UInt8Array",
    "UnitVec3",
    "Vec3",
    "check_frame",
    "compounds_frame",
    "conformers_frame",
    "failures_frame",
    "parse_query",
]
