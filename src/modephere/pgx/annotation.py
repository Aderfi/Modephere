"""PharmGKB variant annotation — one row of ``var_{drug,pheno,fa}_ann.tsv``.

These files are the source of the knowledge graph (see ``src/kg/``). Each row
links one or more variants/haplotypes, one or more genes and zero or more
drugs through a single literature-backed finding. ``PGxAnnotation`` is the
validated boundary between the raw TSV and the Neo4j loader.
"""

from __future__ import annotations

import csv
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Mapping


class AnnotationSource(StrEnum):
    """Which PharmGKB variant-annotation file a row came from."""

    DRUG = "var_drug_ann"
    PHENO = "var_pheno_ann"
    FUNCTIONAL = "var_fa_ann"


#: Closed vocabulary of PharmGKB ``Phenotype Category`` values.
PHENOTYPE_CATEGORIES: tuple[str, ...] = (
    "Dosage",
    "Efficacy",
    "Metabolism/PK",
    "Other",
    "PD",
    "Toxicity",
)

#: Closed vocabulary of PharmGKB ``Direction of effect`` values.
DIRECTIONS: tuple[str, ...] = ("increased", "decreased")


def split_pharmgkb_list(value: str | None) -> list[str]:
    """Split a PharmGKB comma list, honouring double-quoted items.

    ``'"interferon alfa-2b, recombinant", ribavirin'`` yields two items, not
    three. Empty items are dropped; order and duplicates are preserved.

    Fields that are CSV-escaped as a whole (wrapped in quotes, inner quotes
    doubled) reach us raw because the TSVs are read with quoting disabled;
    they are unescaped first.
    """
    if not value or not value.strip():
        return []
    value = value.strip()
    if len(value) > 1 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1].replace('""', '"')
    row = next(csv.reader([value], skipinitialspace=True))
    return [item.strip() for item in row if item.strip()]


def normalize_drug_name(name: str) -> str:
    """Canonical drug key: lowercase, single spaces, no surrounding quotes."""
    return " ".join(name.strip().strip('"').lower().split())


class PGxAnnotation(BaseModel):
    """A single PharmGKB variant annotation, normalized for graph loading."""

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    id: str = Field(..., min_length=1, description="Variant Annotation ID.")
    source: AnnotationSource
    variants: list[str] = Field(default_factory=list)
    genes: list[str] = Field(default_factory=list)
    drugs: list[str] = Field(default_factory=list, description="Normalized drug names.")
    pmid: str | None = None
    phenotype_categories: list[str] = Field(default_factory=list)
    direction: str | None = None
    significance: str | None = None
    associated: bool | None = Field(
        default=None, description="False for 'Not associated with' findings."
    )
    alleles: str | None = Field(
        default=None, description="Raw 'Alleles' field (genotype/allele studied)."
    )
    comparison_alleles: str | None = Field(
        default=None, description="Raw 'Comparison Allele(s) or Genotype(s)' field."
    )

    @field_validator("phenotype_categories")
    @classmethod
    def _known_categories(cls, value: list[str]) -> list[str]:
        unknown = [c for c in value if c not in PHENOTYPE_CATEGORIES]
        if unknown:
            msg = f"unknown phenotype categories: {unknown}"
            raise ValueError(msg)
        return value

    @field_validator("direction")
    @classmethod
    def _known_direction(cls, value: str | None) -> str | None:
        if value is not None and value not in DIRECTIONS:
            msg = f"unknown direction of effect: {value!r}"
            raise ValueError(msg)
        return value

    @classmethod
    def from_pharmgkb_row(
        cls, row: Mapping[str, str | None], source: AnnotationSource
    ) -> PGxAnnotation:
        """Build from a raw TSV row keyed by the PharmGKB column headers."""

        def _get(key: str) -> str | None:
            value = row.get(key)
            if value is None or not str(value).strip():
                return None
            return str(value).strip()

        assoc = _get("Is/Is Not associated")
        return cls(
            id=_get("Variant Annotation ID") or "",
            source=source,
            variants=split_pharmgkb_list(_get("Variant/Haplotypes")),
            genes=split_pharmgkb_list(_get("Gene")),
            drugs=[
                normalize_drug_name(d) for d in split_pharmgkb_list(_get("Drug(s)"))
            ],
            pmid=_get("PMID"),
            phenotype_categories=split_pharmgkb_list(_get("Phenotype Category")),
            direction=_get("Direction of effect"),
            significance=(_get("Significance") or "").lower() or None,
            associated=None
            if assoc is None
            else assoc.lower().startswith("associated"),
            alleles=_get("Alleles"),
            comparison_alleles=_get("Comparison Allele(s) or Genotype(s)"),
        )


__all__ = [
    "DIRECTIONS",
    "PHENOTYPE_CATEGORIES",
    "AnnotationSource",
    "PGxAnnotation",
    "normalize_drug_name",
    "split_pharmgkb_list",
]
