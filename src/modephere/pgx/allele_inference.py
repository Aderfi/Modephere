"""Link an annotation about an rsID to the one alt allele it actually studied.

An rsID names a site, often with several alternative alleles. PharmGKB
annotations report the alleles or genotypes they compared (``CT`` vs ``CC``).
When those bases are pure nucleotides, all belong to the site's GRCh38
reference + alt alleles on the forward strand, and exactly one alt appears,
that alt is the variant the annotation is about. Otherwise nothing is
inferred and the reason is recorded.

Bases outside {ref} ∪ alts are counted as ``allele_strand_mismatch``: that
count is the empirical check that PharmGKB alleles are on the forward strand.
The check is blind where the reverse complement of the report is also
consistent with the site (palindromic C/G, A/T sites; four-allele sites). If
that flipped reading would implicate a different variant, nothing is linked
(``strand_ambiguous``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from modephere.pgx.variant import CanonicalVariant

if TYPE_CHECKING:
    from collections.abc import Sequence

METHOD = "annotation_alleles_forward_strand"
_TOKEN = re.compile(r"^[ACGT]{1,2}$")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


class Inference(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: CanonicalVariant | None
    reason: str | None
    evidence: str


def _bases(field: str | None) -> set[str] | None:
    """Bases named by an allele/genotype field, or None if not pure nucleotides."""
    if field is None:
        return set()
    tokens = [t for t in re.split(r"[+\s]+", field.strip()) if t]
    if not tokens or not all(_TOKEN.match(t) for t in tokens):
        return None
    return set("".join(tokens))


def _single_snv_ref(site_variants: Sequence[CanonicalVariant]) -> str | None:
    """The shared reference base when the site is SNVs at one position."""
    if not site_variants or not all(v.is_snv for v in site_variants):
        return None
    if len({(v.seq_id, v.pos, v.ref) for v in site_variants}) != 1:
        return None
    return site_variants[0].ref


def _decide(
    bases: set[str], site_variants: Sequence[CanonicalVariant]
) -> tuple[str | None, CanonicalVariant | None]:
    ref = _single_snv_ref(site_variants)
    if ref is None:
        return "not_snv_site", None
    by_alt = {v.alt: v for v in site_variants}
    if not bases <= {ref, *by_alt}:
        return "allele_strand_mismatch", None
    alts = bases - {ref}
    if len(alts) != 1:
        return ("no_alt_observed" if not alts else "multiple_alts_observed"), None
    return None, by_alt[alts.pop()]


def infer_variant(
    alleles: str | None,
    comparison: str | None,
    site_variants: Sequence[CanonicalVariant],
) -> Inference:
    evidence = f"Alleles={alleles!r}; Comparison={comparison!r}"
    studied, compared = _bases(alleles), _bases(comparison)
    if studied is None or compared is None or not (studied | compared):
        return Inference(
            variant=None, reason="not_nucleotide_alleles", evidence=evidence
        )
    bases = studied | compared
    reason, variant = _decide(bases, site_variants)
    if variant is not None and _strand_flip_changes_answer(
        bases, variant, site_variants
    ):
        return Inference(variant=None, reason="strand_ambiguous", evidence=evidence)
    return Inference(variant=variant, reason=reason, evidence=evidence)


def _strand_flip_changes_answer(
    bases: set[str],
    variant: CanonicalVariant,
    site_variants: Sequence[CanonicalVariant],
) -> bool:
    """True when the reverse-complemented report is also consistent with the
    site but implicates a different variant (or none) — the forward-strand
    assumption then decides the answer and cannot be checked."""
    flipped = {b.translate(_COMPLEMENT) for b in bases}
    flip_reason, flip_variant = _decide(flipped, site_variants)
    # A flipped reading naming two alts and omitting the reference is not a
    # plausible report (no group would carry the reference allele).
    plausible = flip_reason in {None, "no_alt_observed"}
    return plausible and flip_variant != variant


__all__ = ["METHOD", "Inference", "infer_variant"]
