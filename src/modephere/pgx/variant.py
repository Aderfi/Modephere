"""Canonical variant identity for the knowledge graph.

In the graph a variant is identified by its genomic **HGVS** on a GRCh38
RefSeq chromosome accession, ``Variant {hgvs_g38: "NC_000010.11:g.94781859G>A"}``:
the expression names the sequence, position and change, so it needs no rsID,
gene or star allele to stand on its own. Every assembly-dependent property
carries the assembly suffix (``hgvs_g38``, ``spdi_g38``, ``pos_g38`` …) so
another assembly can be added next to it (``hgvs_g37``, ``hgvs_t2t``) without
replacing anything.

The HGVS is derived from the NCBI **SPDI** (``NC_000010.11:94781858:G:A``),
which is canonical only after NCBI's ``canonical_representative`` service has
normalized it (fully justified); indels get their HGVS from NCBI too. This
module never normalizes indels itself. SPDI positions are 0-based interbase by
definition; ``CanonicalVariant.pos`` is the project's 1-based coordinate.

Several variants together are written in HGVS allele syntax: a *cis* set
(a star allele) as ``NC_000010.11:g.[94762804C>T;94781859G>A]``, a genotype
as ``NC_000010.11:g.[94781859G>A];[94781859=]`` — ``[=]`` is the reference
allele, ``[?]`` an allele the source did not report.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

# GRCh38 primary-assembly chromosome accessions (GRCh38.p14; chromosome
# accessions do not change across patches). RefSeq versions differ per
# chromosome, so a version threshold is wrong: chr19 is NC_000019.10 here but
# NC_000019.9 in GRCh37. Verified 2026-10-18 against NCBI nuccore titles.
GRCH38_CHROMOSOMES: frozenset[str] = frozenset(
    {
        "NC_000001.11",
        "NC_000002.12",
        "NC_000003.12",
        "NC_000004.12",
        "NC_000005.10",
        "NC_000006.12",
        "NC_000007.14",
        "NC_000008.11",
        "NC_000009.12",
        "NC_000010.11",
        "NC_000011.10",
        "NC_000012.12",
        "NC_000013.11",
        "NC_000014.9",
        "NC_000015.10",
        "NC_000016.10",
        "NC_000017.11",
        "NC_000018.10",
        "NC_000019.10",
        "NC_000020.11",
        "NC_000021.9",
        "NC_000022.11",
        "NC_000023.11",
        "NC_000024.10",
        "NC_012920.1",
    }
)
_SPDI = re.compile(
    r"^(?P<seq>[A-Z]{2}_\d+\.\d+):(?P<pos>\d+):(?P<del>[ACGTN]*):(?P<ins>[ACGTN]*)$"
)


def is_grch38_chromosome(accession: str) -> bool:
    """True for a GRCh38 primary-assembly RefSeq chromosome accession."""
    return accession in GRCH38_CHROMOSOMES


def format_spdi(seq_id: str, pos_0based: int, deleted: str, inserted: str) -> str:
    return f"{seq_id}:{pos_0based}:{deleted}:{inserted}"


def parse_spdi(spdi: str) -> tuple[str, int, str, str]:
    """Split an SPDI string into ``(seq_id, pos_0based, deleted, inserted)``."""
    match = _SPDI.match(spdi)
    if match is None:
        msg = f"not a valid SPDI expression: {spdi!r}"
        raise ValueError(msg)
    return match["seq"], int(match["pos"]), match["del"], match["ins"]


def snv_hgvs(seq_id: str, pos_1based: int, ref: str, alt: str) -> str:
    """Genomic (``g.``) or mitochondrial (``m.``) HGVS for a substitution."""
    return f"{seq_id}:{coordinate_type(seq_id)}.{pos_1based}{ref}>{alt}"


MITOCHONDRION = "NC_012920.1"
_HGVS = re.compile(r"^(?P<seq>[A-Z]{2}_\d+\.\d+):(?P<type>[gm])\.(?P<change>.+)$")


def coordinate_type(seq_id: str) -> str:
    """HGVS coordinate prefix: ``m`` on the mitochondrial genome, else ``g``."""
    return "m" if seq_id == MITOCHONDRION else "g"


def hgvs_change(hgvs_g: str) -> tuple[str, str]:
    """``'NC_000010.11:g.94781859G>A'`` -> ``('NC_000010.11', '94781859G>A')``."""
    m = _HGVS.match(hgvs_g)
    if m is None or m["type"] != coordinate_type(m["seq"]):
        msg = f"not a genomic HGVS expression: {hgvs_g!r}"
        raise ValueError(msg)
    return m["seq"], m["change"]


def cis_hgvs(seq_id: str, changes: Sequence[str]) -> str:
    """One allele (chromosome copy) carrying ``changes``; none = the reference."""
    t = coordinate_type(seq_id)
    if not changes:
        return f"{seq_id}:{t}.="
    if len(changes) == 1:
        return f"{seq_id}:{t}.{changes[0]}"
    return f"{seq_id}:{t}.[{';'.join(changes)}]"


def diploid_hgvs(
    seq_id: str, first: Sequence[str] | None, second: Sequence[str] | None
) -> str:
    """Both alleles of a genotype; ``()`` is the reference, ``None`` unknown."""

    def side(changes: Sequence[str] | None) -> str:
        return "?" if changes is None else (";".join(changes) or "=")

    return f"{seq_id}:{coordinate_type(seq_id)}.[{side(first)}];[{side(second)}]"


class CanonicalVariant(BaseModel):
    """One allele-specific change on GRCh38, keyed by canonical SPDI."""

    model_config = ConfigDict(frozen=True)

    spdi: str
    seq_id: str
    pos: int = Field(..., ge=1, description="1-based position (SPDI position + 1).")
    ref: str = Field(..., description="Deleted sequence (SPDI 'D'); may be empty.")
    alt: str = Field(..., description="Inserted sequence (SPDI 'I'); may be empty.")
    hgvs_g: str = Field(..., min_length=1)
    assembly: Literal["GRCh38"] = "GRCh38"
    vrs_id: None = Field(
        default=None, description="Reserved for GA4GH VRS (pangenome phase)."
    )

    @field_validator("seq_id")
    @classmethod
    def _grch38_only(cls, value: str) -> str:
        if not is_grch38_chromosome(value):
            msg = f"{value!r} is not a GRCh38 chromosome accession"
            raise ValueError(msg)
        return value

    @classmethod
    def from_spdi(
        cls, seq_id: str, pos_0based: int, deleted: str, inserted: str, hgvs_g: str
    ) -> CanonicalVariant:
        return cls(
            spdi=format_spdi(seq_id, pos_0based, deleted, inserted),
            seq_id=seq_id,
            pos=pos_0based + 1,
            ref=deleted,
            alt=inserted,
            hgvs_g=hgvs_g,
        )

    @property
    def is_snv(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1

    @property
    def change(self) -> str:
        """The HGVS change without its sequence, e.g. ``94781859G>A``."""
        return hgvs_change(self.hgvs_g)[1]

    @property
    def reference_change(self) -> str | None:
        """HGVS for the reference allele over this variant's span (``94781859=``).

        ``None`` for an insertion without reference context (empty SPDI ``D``).
        """
        if not self.ref:
            return None
        end = self.pos + len(self.ref) - 1
        return f"{self.pos}=" if end == self.pos else f"{self.pos}_{end}="

    def graph_props(self) -> dict[str, object]:
        """Node properties besides the ``hgvs_g38`` key, all GRCh38-specific."""
        return {
            "spdi_g38": self.spdi,
            "seq_id_g38": self.seq_id,
            "pos_g38": self.pos,
            "ref_g38": self.ref,
            "alt_g38": self.alt,
        }


__all__ = [
    "GRCH38_CHROMOSOMES",
    "CanonicalVariant",
    "cis_hgvs",
    "diploid_hgvs",
    "format_spdi",
    "hgvs_change",
    "is_grch38_chromosome",
    "parse_spdi",
    "snv_hgvs",
]
