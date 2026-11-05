"""Compound records: a query after resolution, and after standardization.

Two stages, two models (PLAN C2.1):

* :class:`ResolvedCompound` — the structure a source returned for a query,
  with provenance. Nothing has been changed by RDKit yet.
* :class:`StandardizedCompound` — the same compound in its two working
  representations:

  - the **registration form** (neutral parent, canonical tautomer). It is used
    only for identity: its InChIKey is the library-wide compound key;
  - the **working form** (neutral parent, input tautomer) and its pH 7.4
    **microspecies**. These feed conformers and pharmacophore perception. The
    canonical tautomer is never used there: it is a canonical representative,
    not the dominant tautomer, and invents donors (dev_Pgen pharmacophore.py).
"""

from __future__ import annotations

from typing import Self

from pydantic import AwareDatetime, Field, NonNegativeInt, PositiveInt, model_validator

from modephere.types.base import FrozenModel
from modephere.types.enums import (
    ChemblMatch,
    FragmentPolicy,
    ProtonationMethod,
    ResolutionSource,
)
from modephere.types.identifiers import (
    ChemblQuery,
    CidQuery,
    CompoundQuery,
    SmilesQuery,
)
from modephere.types.scalars import (
    ChemblID,
    InChIKey,
    NonEmptyStr,
    PubChemCID,
    SmilesStr,
)


class ResolvedCompound(FrozenModel):
    """A query resolved to a structure, before any standardization."""

    query: CompoundQuery
    smiles: SmilesStr = Field(
        ..., description="Structure as given (SMILES query) or returned upstream."
    )
    source: ResolutionSource
    cid: PubChemCID | None = None
    inchikey: InChIKey | None = Field(
        default=None, description="InChIKey reported upstream, if any."
    )
    title: NonEmptyStr | None = Field(
        default=None, description="Preferred name (PubChem Title)."
    )
    candidate_cids: tuple[PubChemCID, ...] = Field(
        default=(),
        description="Every CID a name lookup returned, in PubChem's order.",
    )
    retrieved_at: AwareDatetime | None = Field(
        default=None,
        description="When the upstream response was fetched (ChEMBL: release date).",
    )
    chembl_id: ChemblID | None = Field(
        default=None,
        description="ChEMBL *parent* molecule: the key of its annotations and "
        "bioactivity, whatever form the query named.",
    )
    chembl_record_id: ChemblID | None = Field(
        default=None,
        description="ChEMBL molecule the match was made on (salt, parent).",
    )
    chembl_match: ChemblMatch | None = None
    chembl_release: NonEmptyStr | None = Field(
        default=None, description="Staged ChEMBL release, e.g. chembl_37."
    )
    candidate_chembl_ids: tuple[ChemblID, ...] = Field(
        default=(),
        description="Every ChEMBL parent a name matched, best first.",
    )
    warnings: tuple[NonEmptyStr, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        return len(self.candidate_cids) > 1 or len(self.candidate_chembl_ids) > 1

    @model_validator(mode="after")
    def _check_provenance(self) -> Self:
        if self.source is ResolutionSource.LOCAL:
            if not isinstance(self.query, SmilesQuery):
                msg = f"only a SMILES query resolves locally, got {self.query.kind}"
                raise ValueError(msg)
        elif self.retrieved_at is None:
            msg = f"a {self.source} resolution must record retrieved_at"
            raise ValueError(msg)
        chembl = (
            self.chembl_id,
            self.chembl_record_id,
            self.chembl_match,
            self.chembl_release,
        )
        if any(v is not None for v in chembl) and any(v is None for v in chembl):
            msg = "a ChEMBL match needs chembl_id, record, match kind and release"
            raise ValueError(msg)
        if self.source is ResolutionSource.CHEMBL and self.chembl_id is None:
            msg = "a ChEMBL resolution must name its ChEMBL molecule"
            raise ValueError(msg)
        if isinstance(self.query, ChemblQuery) and (
            self.source is not ResolutionSource.CHEMBL
            or self.chembl_record_id != self.query.chembl_id
        ):
            msg = f"ChEMBL query {self.query.chembl_id} resolved elsewhere"
            raise ValueError(msg)
        if (
            self.candidate_chembl_ids
            and self.chembl_id not in self.candidate_chembl_ids
        ):
            msg = f"ChEMBL parent {self.chembl_id} is not among {self.candidate_chembl_ids}"
            raise ValueError(msg)
        if isinstance(self.query, CidQuery) and self.cid != self.query.cid:
            msg = f"CID query {self.query.cid} resolved to CID {self.cid}"
            raise ValueError(msg)
        if len(set(self.candidate_cids)) != len(self.candidate_cids):
            msg = f"duplicate candidate CIDs: {self.candidate_cids}"
            raise ValueError(msg)
        if self.candidate_cids and self.cid not in self.candidate_cids:
            msg = f"chosen CID {self.cid} is not among {self.candidate_cids}"
            raise ValueError(msg)
        return self


class StandardizedCompound(FrozenModel):
    """A resolved compound in its registration and working representations."""

    resolved: ResolvedCompound
    inchikey: InChIKey = Field(
        ..., description="Registration InChIKey: the library-wide compound key."
    )
    registration_smiles: SmilesStr = Field(
        ..., description="Neutral parent, canonical tautomer (identity only)."
    )
    working_smiles: SmilesStr = Field(
        ..., description="Neutral parent, input tautomer (3D and features)."
    )
    microspecies_smiles: SmilesStr = Field(
        ..., description="Working form protonated for pH 7.4."
    )
    fragment_policy: FragmentPolicy
    protonation: ProtonationMethod
    fragments_removed: NonNegativeInt = Field(
        ..., description="Counterions / solvent fragments dropped."
    )
    num_heavy_atoms: PositiveInt
    formal_charge: int = Field(..., description="Net charge of the microspecies.")
    toolkit_version: NonEmptyStr = Field(
        ...,
        description="RDKit version that standardized it: canonical SMILES change "
        "between releases, so it is recorded with every record.",
    )

    @property
    def key(self) -> str:
        return self.inchikey


__all__ = ["ResolvedCompound", "StandardizedCompound"]
