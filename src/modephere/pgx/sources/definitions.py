"""Haplotype definitions — the shape both PharmVar and ClinPGx readers produce.

A haplotype is the set of variants that sit together on one chromosome
(*cis*). Each defining variant is kept as it appears in its source — either a
raw SPDI built from the source's coordinates or the source's own HGVS — and is
canonicalized later through NCBI, never here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DefinitionRow(BaseModel):
    """One defining variant, exactly as its source states it."""

    model_config = ConfigDict(frozen=True)

    spdi_input: str | None = None
    hgvs_input: str | None = None
    rsid: str | None = None
    evidence: str = Field(..., min_length=1, description="The raw source row or cell.")

    @model_validator(mode="after")
    def _exactly_one_input(self) -> DefinitionRow:
        if (self.spdi_input is None) == (self.hgvs_input is None):
            msg = "exactly one of spdi_input / hgvs_input must be set"
            raise ValueError(msg)
        return self


class HaplotypeDefinition(BaseModel):
    """A named allele and its defining variants."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Full name, e.g. 'CYP2C19*2.001' or 'TPMT*3A'.")
    gene: str
    core_name: str = Field(..., description="'CYP2C19*2' for 'CYP2C19*2.001'.")
    is_reference: bool = False
    rows: tuple[DefinitionRow, ...] = ()
    dataset_id: str
    unresolved_reason: str | None = None


__all__ = ["DefinitionRow", "HaplotypeDefinition"]
