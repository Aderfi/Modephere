"""Classify PharmGKB ``Variant/Haplotypes`` labels.

The label itself is kept verbatim; classification only decides which
resolver may handle it and, for labels that can never resolve, why not:

* ``rsid`` — dbSNP RefSNP.
* ``star`` — a star allele defined by small variants (PharmVar / ClinPGx).
* ``cnv`` — copy-number alleles, each with its :class:`Structure`:
  whole-gene deletions (CYP2D6\\*5, CYP2A6\\*4, and the ``*0`` allele behind
  GSTM1/GSTT1 ``null``), duplications of a star allele (``*2xN``, ``*1x2``),
  and CYP2D6-2D7 / CYP2A6-2A7 hybrids. Hybrids stay unresolved
  (``hybrid_allele``): splitting them into small variants means guessing.
  ``GSTM1 non-null`` is a genotype class (at least one copy), not an allele.
* ``hla`` — HLA alleles, named by IPD-IMGT/HLA (``HLA-B*58:01``). They are
  defined by their whole sequence, not by changes against GRCh38, so they are
  identified by that name alone (``Structure.HLA_TYPED``).
* ``phenotype`` — metabolizer / acetylator / deficiency labels: the result of
  a diplotype, not a variant.
* ``other`` — named variants this phase has no source for (G6PD Canton,
  SLC6A4 HTTLPR repeat forms, …).
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class AlleleKind(StrEnum):
    RSID = "rsid"
    STAR = "star"
    HLA = "hla"
    CNV = "cnv"
    PHENOTYPE = "phenotype"
    OTHER = "other"


class Structure(StrEnum):
    """What a structural allele is, when it is not a set of small variants."""

    GENE_DELETION = "gene_deletion"
    DUPLICATION = "duplication"
    HYBRID = "hybrid"
    HLA_TYPED = "hla_typed"


#: The allele name of a whole-gene deletion written as ``<GENE> null``.
NULL_ALLELE = "*0"

_RSID = re.compile(r"^rs(\d+)$")
_STAR = re.compile(r"^[A-Z][A-Z0-9-]*\*\S+$")
_DUPLICATION = re.compile(r"^(?P<base>[A-Z][A-Z0-9-]*\*[^\sx]+)x(?P<n>N|\d+)$")
_HLA = re.compile(r"^HLA-[A-Z0-9]+\*\d{2,3}(?::\d{2,3}){0,3}[NLSCAQ]?$")
_NULL_GENOTYPE = re.compile(r"^(?P<gene>\S+) (?P<state>null|non-null)$", re.IGNORECASE)
# PharmVar structural-variation catalogue: deletions and hybrids with a
# neighbouring pseudogene/paralog are not small-variant sets.
_GENE_DELETIONS: dict[str, frozenset[str]] = {
    "CYP2D6": frozenset({"*5"}),
    "CYP2A6": frozenset({"*4"}),
    "GSTM1": frozenset({NULL_ALLELE}),
    "GSTT1": frozenset({NULL_ALLELE}),
}
_HYBRIDS: dict[str, frozenset[str]] = {
    "CYP2D6": frozenset({"*13", "*36", "*61", "*63", "*68"}),
    "CYP2A6": frozenset({"*12", "*34"}),
}
_PHENOTYPE = re.compile(
    r"metaboli[sz]er|acetylator|deficien|activity|function|phenotype|genotype",
    re.IGNORECASE,
)


class ClassifiedLabel(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    kind: AlleleKind
    rsid: int | None = None
    #: The haplotype the label names: the star allele, the deletion allele
    #: (``GSTM1*0`` for ``GSTM1 null``), the HLA allele, or -- for a
    #: duplication -- the allele that is duplicated (``CYP2D6*2`` for ``*2xN``).
    haplotype_name: str | None = None
    structure: Structure | None = None
    #: Copies of the gene this allele puts on one chromosome: 0 for a
    #: deletion, 2 for ``x2``, None for ``xN`` (at least 2), 1 otherwise.
    copies: int | None = 1
    reason: str | None = None


def _structural_star(label: str, text: str) -> ClassifiedLabel | None:
    gene, allele = text.split("*", 1)
    if m := _DUPLICATION.match(text):
        base = classify_label(m["base"])
        if base.kind is not AlleleKind.STAR:
            return ClassifiedLabel(
                label=label, kind=AlleleKind.CNV, reason="duplicated_structural_allele"
            )
        return ClassifiedLabel(
            label=label,
            kind=AlleleKind.CNV,
            haplotype_name=m["base"],
            structure=Structure.DUPLICATION,
            copies=None if m["n"] == "N" else int(m["n"]),
        )
    if f"*{allele}" in _GENE_DELETIONS.get(gene, frozenset()):
        return ClassifiedLabel(
            label=label,
            kind=AlleleKind.CNV,
            haplotype_name=text,
            structure=Structure.GENE_DELETION,
            copies=0,
        )
    if f"*{allele}" in _HYBRIDS.get(gene, frozenset()):
        return ClassifiedLabel(
            label=label,
            kind=AlleleKind.CNV,
            structure=Structure.HYBRID,
            reason="hybrid_allele",
        )
    return None


def _named_hla(label: str, text: str) -> ClassifiedLabel:
    if _HLA.match(text):
        return ClassifiedLabel(
            label=label,
            kind=AlleleKind.HLA,
            haplotype_name=text,
            structure=Structure.HLA_TYPED,
        )
    return ClassifiedLabel(
        label=label, kind=AlleleKind.HLA, reason="unrecognized_hla_name"
    )


def _null_genotype(label: str, m: re.Match[str]) -> ClassifiedLabel:
    if m["state"].lower() == "non-null":
        return ClassifiedLabel(
            label=label, kind=AlleleKind.CNV, reason="genotype_class"
        )
    return ClassifiedLabel(
        label=label,
        kind=AlleleKind.CNV,
        haplotype_name=f"{m['gene']}{NULL_ALLELE}",
        structure=Structure.GENE_DELETION,
        copies=0,
    )


def _named(label: str, text: str) -> ClassifiedLabel:
    """A label that is neither an rsID, an HLA allele nor a null genotype."""
    if _STAR.match(text):
        structural = _structural_star(label, text)
        if structural is not None:
            return structural
        return ClassifiedLabel(label=label, kind=AlleleKind.STAR, haplotype_name=text)
    if _PHENOTYPE.search(text):
        return ClassifiedLabel(
            label=label, kind=AlleleKind.PHENOTYPE, reason="not_a_variant"
        )
    return ClassifiedLabel(
        label=label, kind=AlleleKind.OTHER, reason="not_in_pharmvar_or_tables"
    )


def classify_label(label: str) -> ClassifiedLabel:
    text = label.strip()
    if m := _RSID.match(text):
        return ClassifiedLabel(label=label, kind=AlleleKind.RSID, rsid=int(m.group(1)))
    if text.startswith("HLA-"):
        return _named_hla(label, text)
    if m := _NULL_GENOTYPE.match(text):
        return _null_genotype(label, m)
    return _named(label, text)


def null_genotype_gene(label: str) -> str | None:
    """``'GSTM1'`` for ``'GSTM1 null'`` / ``'GSTM1 non-null'``."""
    m = _NULL_GENOTYPE.match(label.strip())
    return m["gene"] if m else None


__all__ = [
    "NULL_ALLELE",
    "AlleleKind",
    "ClassifiedLabel",
    "Structure",
    "classify_label",
    "null_genotype_gene",
]
