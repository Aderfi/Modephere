"""Closed vocabularies of the library, as ``StrEnum``.

Member *order* is part of the contract where an enum defines a feature layout
(:class:`PharmacophoreFamily` is the pharmacophore one-hot, identical to
dev_Pgen's ``_FAMILIES``): append new members, never reorder.
"""

from __future__ import annotations

from enum import StrEnum

# --------------------------------------------------------------------------- #
# Ingestion and resolution
# --------------------------------------------------------------------------- #


class IdentifierKind(StrEnum):
    """What a :class:`~modephere.types.identifiers.CompoundQuery` carries."""

    SMILES = "smiles"
    CID = "cid"
    NAME = "name"
    CHEMBL = "chembl_id"


class ChemblMatch(StrEnum):
    """How a resolved compound was tied to a ChEMBL molecule."""

    CHEMBL_ID = "chembl_id"  # the query was a ChEMBL ID
    NAME = "name"  # preferred name or synonym in the ChEMBL release
    UPSTREAM_INCHIKEY = "upstream_inchikey"  # InChIKey PubChem reported
    STRUCTURE_INCHIKEY = "structure_inchikey"  # standard InChIKey of the SMILES


class ResolutionSource(StrEnum):
    """Where a resolved structure came from."""

    LOCAL = "local"  # parsed from the SMILES given, no network
    PUBCHEM = "pubchem"
    CHEMBL = "chembl"


class AmbiguityPolicy(StrEnum):
    """What to do when a name maps to several compounds (PubChem CIDs or
    ChEMBL parents)."""

    FIRST_HIT = "first_hit"  # PubChem's first CID, logged as a warning (default)
    FAIL = "fail"


# --------------------------------------------------------------------------- #
# Standardization
# --------------------------------------------------------------------------- #


class FragmentPolicy(StrEnum):
    """How the parent of a multi-fragment input (salt, solvate) is chosen."""

    LARGEST_FRAGMENT = "largest_fragment"  # dev_Pgen behaviour (default)
    CHEMBL_PARENT = "chembl_parent"  # chembl_structure_pipeline salt list


class ProtonationMethod(StrEnum):
    """How the pH 7.4 microspecies is assigned."""

    RULE_BASED = "rule_based"  # port of dev_Pgen's _microspecies (default)


# --------------------------------------------------------------------------- #
# Conformers
# --------------------------------------------------------------------------- #


class EmbeddingMethod(StrEnum):
    """RDKit distance-geometry parameter set (``rdDistGeom.<value>()``)."""

    ETKDG_V3 = "ETKDGv3"
    SR_ETKDG_V3 = "srETKDGv3"  # small-ring torsion preferences


class ForceField(StrEnum):
    """Force field used to optimise conformers."""

    MMFF94 = "MMFF94"
    MMFF94S = "MMFF94s"
    UFF = "UFF"
    NONE = "none"  # embed only; conformers carry no energy


class StereoPolicy(StrEnum):
    """What to do with unassigned stereocentres before embedding."""

    ENUMERATE = "enumerate"  # embed each stereoisomer, capped (default)
    RANDOM = "random"  # let the embedder pick one


# --------------------------------------------------------------------------- #
# Pharmacophores
# --------------------------------------------------------------------------- #


class PharmacophoreFamily(StrEnum):
    """The six non-overlapping families (CATS / ErG / vendor vocabulary).

    Values are RDKit ``BaseFeatures.fdef`` family names. Excluded volume is
    deliberately absent: it describes the receptor, which a ligand-only build
    does not have.
    """

    DONOR = "Donor"
    ACCEPTOR = "Acceptor"
    POS_IONIZABLE = "PosIonizable"
    NEG_IONIZABLE = "NegIonizable"
    AROMATIC = "Aromatic"
    HYDROPHOBE = "Hydrophobe"

    @property
    def one_hot_index(self) -> int:
        """Position in the one-hot layout."""
        return _FAMILY_INDEX[self]

    @property
    def is_directional(self) -> bool:
        """Whether a 3D point of this family may carry a direction vector.

        Donor/acceptor directions point along the H-bond; an aromatic direction
        is the ring normal. Ions and hydrophobes are isotropic.
        """
        return self in _DIRECTIONAL


_FAMILY_INDEX = {f: i for i, f in enumerate(PharmacophoreFamily)}
_DIRECTIONAL = frozenset(
    {
        PharmacophoreFamily.DONOR,
        PharmacophoreFamily.ACCEPTOR,
        PharmacophoreFamily.AROMATIC,
    }
)

# --------------------------------------------------------------------------- #
# Build reporting
# --------------------------------------------------------------------------- #


class BuildStage(StrEnum):
    """Pipeline stage, in execution order."""

    RESOLVE = "resolve"
    STANDARDIZE = "standardize"
    CONFORMERS = "conformers"
    PHARMACOPHORE = "pharmacophore"
    FEATURIZE = "featurize"
    SAVE = "save"


class FailureCategory(StrEnum):
    """The nature of a per-compound failure (the failure counter's key)."""

    INVALID_QUERY = "invalid_query"
    UNRESOLVED_IDENTIFIER = "unresolved_identifier"
    AMBIGUOUS_IDENTIFIER = "ambiguous_identifier"  # only under AmbiguityPolicy.FAIL
    CACHE_MISS = "cache_miss"  # offline build, response never fetched
    UPSTREAM_ERROR = "upstream_error"
    INVALID_SMILES = "invalid_smiles"
    EMPTY_MOLECULE = "empty_molecule"
    UNSUPPORTED_CHEMISTRY = "unsupported_chemistry"  # metals, inorganics
    TOO_LARGE = "too_large"  # above the heavy-atom cap
    STANDARDIZATION_ERROR = "standardization_error"
    EMBEDDING_FAILED = "embedding_failed"
    FORCEFIELD_FAILED = "forcefield_failed"
    NO_PHARMACOPHORE_FEATURES = "no_pharmacophore_features"
    SAVE_ERROR = "save_error"


__all__ = [
    "AmbiguityPolicy",
    "BuildStage",
    "ChemblMatch",
    "EmbeddingMethod",
    "FailureCategory",
    "ForceField",
    "FragmentPolicy",
    "IdentifierKind",
    "PharmacophoreFamily",
    "ProtonationMethod",
    "ResolutionSource",
    "StereoPolicy",
]
