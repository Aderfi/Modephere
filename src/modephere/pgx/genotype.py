"""Genotypes a PharmGKB annotation compares: diplotypes and site genotypes.

An outcome often depends on the *pair* of alleles a patient carries:
``CYP2C9*1/*22`` and ``*22/*22`` differ, as do ``CT`` and ``TT`` at an rsID.
PharmGKB writes them in the ``Alleles`` field (what was tested) and the
``Comparison Allele(s) or Genotype(s)`` field (the reference group)::

    *1/*22 + *22/*22      two diplotypes; '+' means either qualifies
    *3                    carriers of *3 (the other allele is not reported)
    *1/*5, *1/*2xN        a whole-gene deletion, a duplication
    null, non-null        GSTM1 / GSTT1: both copies deleted, at least one copy
    CT, AA + AG           site genotypes; 'T' alone is a carrier of T
    G/del, CTT/CTT, TC/TCCTC   indel genotypes

This module parses those fields and resolves each genotype against the
haplotypes / dbSNP site the build already resolved. It is pure: the build
turns the result into ``(:Genotype)`` nodes. A genotype is resolved only when
every allele in it is; nothing is guessed.

Copies. ``copies`` is the exact number of copies of an allele the genotype
carries, None when the source does not say (a carrier-only report, ``xN``);
``min_copies`` is always the lower bound. Whole-gene deletions are an allele
of their own (``CYP2D6*5``, ``GSTM1*0``) that adds no copy of the gene:
``gene_copies`` (exact) / ``gene_copies_min`` count the gene copies of a
diploid genotype -- ``*4/*5`` has 1, ``*1/*2xN`` at least 3.

Star loci: each allele must be a resolved haplotype, a named deletion, a
duplication of a resolved haplotype, or an HLA allele (named by IPD-IMGT/HLA,
no HGVS). Hybrids stay unresolved.

SNV sites read bases on the forward strand (a name such as
``rs1799853:C/T`` is always forward-strand); a base outside the GRCh38
reference + alternatives is ``allele_strand_mismatch``. Whether an annotation
reported on the forward strand cannot be checked when its bases
reverse-complemented are also a plausible report (:func:`strand_ambiguous`);
the build then links its genotypes on the forward-strand assumption and marks
the link ``strand_assumed``.

Indel sites are read per annotation (:func:`read_indel_site`), because the
two fields share one notation. NCBI writes every allele of a site over the
same span with its full repeat context (``TCCTC`` / ``TC``); PharmGKB uses
either those sequences or ``X/del`` -- *X present / X absent*. A sequence is
matched to the site allele it spells, ``X/del`` to the pair of site alleles
that differ by exactly X (preferring X present in the reference, as in
``CTT/del`` for F508del); compact tokens (``TT``, ``GG``) are split only into
tokens the annotation itself uses. Indel genotype names spell each allele by
its HGVS change, ``=`` for the reference (``rs113993960:117559592_117559594del/=``),
so one genotype has one name however it was written.
"""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from modephere.pgx.labels import (
    NULL_ALLELE,
    AlleleKind,
    ClassifiedLabel,
    Structure,
    classify_label,
    null_genotype_gene,
)
from modephere.pgx.variant import diploid_hgvs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from modephere.pgx.variant import CanonicalVariant

METHOD = "pharmgkb_genotype_notation"

_STAR_SIDE = re.compile(r"^(?P<gene>[A-Z][A-Z0-9-]*)?(?P<allele>\*\S+)$")
_SITE_SIDE = re.compile(r"^(?:[ACGT]+|del)$")
_NULL_SIDE = re.compile(r"^(?:null|non-null)$", re.IGNORECASE)
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_DEL = "del"


class Role(StrEnum):
    """Which field of the annotation a genotype came from."""

    TESTED = "TESTED"
    COMPARED_TO = "COMPARED_TO"


class Locus(BaseModel):
    """The one gene or dbSNP site an annotation is about.

    ``star``: star / structural / HLA alleles of one gene; ``rsid``: one dbSNP
    site; ``null``: a GSTM1/GSTT1-style null / non-null genotype.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["star", "rsid", "null"]
    name: str  # 'CYP2C9', 'rs1799853', 'GSTM1'
    rsid: int | None = None


class ParsedGenotype(BaseModel):
    """One genotype as written: 2 sides = diploid, 1 = carrier or compact."""

    model_config = ConfigDict(frozen=True)

    sides: tuple[str, ...]
    slashed: bool  # written as 'a/b' rather than a compact token ('CT', '*3')


class Copy(BaseModel):
    """How many copies of one allele a genotype carries."""

    model_config = ConfigDict(frozen=True)

    allele: str  # a Haplotype name or a Variant hgvs_g38
    copies: int | None
    min_copies: int


class Genotype(BaseModel):
    """A resolved (or explicitly unresolved) genotype at one locus.

    ``copies`` lists the alleles it carries -- ``Haplotype`` names at a star
    locus, ``Variant`` HGVS at an rsID locus; at an rsID locus the reference
    copies are ``ref_copies`` (they are no variant).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    locus: str
    zygosity: Literal["diploid", "carrier"]
    status: Literal["resolved", "unresolved"]
    reason: str | None = None
    target: Literal["Haplotype", "Variant"]
    copies: tuple[Copy, ...] = ()
    ref_copies: int | None = None
    gene_copies: int | None = None
    gene_copies_min: int | None = None
    hgvs_g38: str | None = None


# --------------------------------------------------------------------------- #
# Locus + parsing
# --------------------------------------------------------------------------- #


def _rsid_locus(labels: Sequence[ClassifiedLabel]) -> Locus | str | None:
    rsids = {c.rsid for c in labels if c.kind is AlleleKind.RSID}
    if not rsids:
        return None
    if len(rsids) > 1 or len(rsids) != len({c.label for c in labels}):
        return "multi_locus"
    (rsid,) = rsids
    assert rsid is not None
    return Locus(kind="rsid", name=f"rs{rsid}", rsid=rsid)


def _null_locus(labels: Sequence[ClassifiedLabel]) -> Locus | str | None:
    genes = {null_genotype_gene(c.label) for c in labels}
    if None in genes:
        return None
    if len(genes) > 1:
        return "multi_locus"
    (gene,) = genes
    assert gene is not None
    return Locus(kind="null", name=gene)


def annotation_locus(labels: Sequence[ClassifiedLabel]) -> Locus | str:
    """The single locus the annotation's labels name, or why there is none."""
    if not labels:
        return "no_labels"
    locus = _rsid_locus(labels) or _null_locus(labels)
    if locus is not None:
        return locus
    genes = {c.label.strip().split("*", 1)[0] for c in labels if "*" in c.label}
    if len(genes) != 1 or any("*" not in c.label for c in labels):
        return "multi_locus" if len(genes) > 1 else "not_a_genotype_locus"
    return Locus(kind="star", name=genes.pop())


def _star_side(side: str, gene: str) -> str | None:
    """'*22' -> 'CYP2C9*22'; a side naming another gene is not parseable."""
    m = _STAR_SIDE.match(side)
    if m is None or (m["gene"] is not None and m["gene"] != gene):
        return None
    return f"{gene}{m['allele']}"


def _sides(raw: list[str], locus: Locus) -> tuple[str, ...] | None:
    if locus.kind == "star":
        names = [_star_side(s, locus.name) for s in raw]
        return None if any(n is None for n in names) else tuple(n or "" for n in names)
    pattern = _SITE_SIDE if locus.kind == "rsid" else _NULL_SIDE
    if not all(pattern.match(s) for s in raw):
        return None
    return tuple(s.lower() if locus.kind == "null" else s for s in raw)


def parse_field(field: str | None, locus: Locus) -> list[ParsedGenotype] | None:
    """``[]`` for an empty field; ``None`` when the notation is not understood."""
    if field is None or not field.strip():
        return []
    out: list[ParsedGenotype] = []
    for alternative in field.split("+"):
        text = alternative.strip()
        slashed = "/" in text
        raw = [s.strip() for s in text.split("/")] if slashed else [text]
        if len(raw) > 2 or not all(raw):  # noqa: PLR2004 -- diploid
            return None
        sides = _sides(raw, locus)
        if sides is None:
            return None
        out.append(ParsedGenotype(sides=sides, slashed=slashed))
    return out


# --------------------------------------------------------------------------- #
# Star loci
# --------------------------------------------------------------------------- #


def _natural(name: str) -> list[object]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


#: Haplotype name -> (seq id, cis changes); () for a reference haplotype.
HaplotypeHgvs = tuple[str, tuple[str, ...]]


class _StarAllele(BaseModel):
    haplotype: str
    copies: int | None  # haplotype copies on this chromosome (None = xN)
    structure: Structure | None


def _star_allele(
    name: str, haplotype_reason: Callable[[str], str | None]
) -> _StarAllele | str:
    c = classify_label(name)
    if c.kind is AlleleKind.STAR:
        allele = _StarAllele(haplotype=name, copies=1, structure=None)
    elif c.haplotype_name is not None and c.structure in {
        Structure.DUPLICATION,
        Structure.GENE_DELETION,
        Structure.HLA_TYPED,
    }:
        dup = c.structure is Structure.DUPLICATION
        allele = _StarAllele(
            haplotype=c.haplotype_name,
            copies=c.copies if dup else 1,
            structure=c.structure,
        )
    else:
        return c.reason or "not_a_star_allele"
    return haplotype_reason(allele.haplotype) or allele


def _gene_copies(alleles: Sequence[_StarAllele]) -> tuple[int | None, int | None]:
    """(exact, lower bound) gene copies of a diploid genotype; HLA: (None, None)."""
    if any(a.structure is Structure.HLA_TYPED for a in alleles):
        return None, None
    per = [0 if a.structure is Structure.GENE_DELETION else a.copies for a in alleles]
    if any(n is None for n in per):
        return None, sum(2 if n is None else n for n in per)
    exact = sum(n for n in per if n is not None)
    return exact, exact


def resolve_star(
    parsed: ParsedGenotype,
    gene: str,
    *,
    haplotype_reason: Callable[[str], str | None],
    haplotype_hgvs: Mapping[str, HaplotypeHgvs],
) -> Genotype:
    """``haplotype_reason(name)`` is None when the haplotype resolved."""
    sides = sorted(parsed.sides, key=_natural)
    carrier = len(sides) == 1
    base = {
        "name": gene + "/".join(s.removeprefix(gene) for s in sides),
        "locus": gene,
        "zygosity": "carrier" if carrier else "diploid",
        "target": "Haplotype",
    }
    resolved = [_star_allele(s, haplotype_reason) for s in sides]
    reasons = [r for r in resolved if isinstance(r, str)]
    if reasons:
        return Genotype(**base, status="unresolved", reason=reasons[0])
    alleles = [a for a in resolved if not isinstance(a, str)]
    exact, lower = _gene_copies(alleles) if not carrier else (None, None)
    return Genotype(
        **base,
        status="resolved",
        copies=_star_copies(alleles, carrier=carrier),
        gene_copies=exact,
        gene_copies_min=lower,
        hgvs_g38=_star_hgvs(alleles, haplotype_hgvs),
    )


def _star_copies(alleles: Sequence[_StarAllele], *, carrier: bool) -> tuple[Copy, ...]:
    by_haplotype: dict[str, list[_StarAllele]] = {}
    for a in alleles:
        by_haplotype.setdefault(a.haplotype, []).append(a)
    out: list[Copy] = []
    for haplotype in sorted(by_haplotype, key=_natural):
        group = by_haplotype[haplotype]
        known = (
            None
            if any(a.copies is None for a in group)
            else sum(a.copies or 0 for a in group)
        )
        duplicated = any(a.structure is Structure.DUPLICATION for a in group)
        lower = sum(2 if a.copies is None else a.copies for a in group)
        out.append(
            Copy(
                allele=haplotype,
                copies=None if carrier else known,
                min_copies=lower if duplicated or not carrier else 1,
            )
        )
    return tuple(out)


def _star_hgvs(
    alleles: Sequence[_StarAllele], hgvs: Mapping[str, HaplotypeHgvs]
) -> str | None:
    """Diploid HGVS; none for duplications (one chromosome, several copies)."""
    if any(a.structure is not None and a.copies != 1 for a in alleles) or any(
        a.haplotype not in hgvs for a in alleles
    ):
        return None
    seqs = {hgvs[a.haplotype][0] for a in alleles}
    if len(seqs) != 1:
        return None
    changes = [hgvs[a.haplotype][1] for a in alleles]
    return diploid_hgvs(
        seqs.pop(), changes[0], changes[1] if len(changes) > 1 else None
    )


def resolve_null(
    parsed: ParsedGenotype,
    gene: str,
    *,
    haplotype_reason: Callable[[str], str | None],
    haplotype_hgvs: Mapping[str, HaplotypeHgvs],
) -> Genotype:
    """GSTM1-style genotypes: ``null`` deletes the gene, ``non-null`` keeps a copy.

    A lone ``null`` is the null genotype (both copies deleted, ``GENE*0/*0``);
    a lone ``non-null`` has at least one copy; ``non-null/null`` has one
    deleted copy and at least one present.
    """
    deletion = f"{gene}{NULL_ALLELE}"
    sides = parsed.sides
    if set(sides) == {"null"}:
        both = ParsedGenotype(sides=(deletion, deletion), slashed=True)
        return resolve_star(
            both, gene, haplotype_reason=haplotype_reason, haplotype_hgvs=haplotype_hgvs
        )
    carrier = sides == ("non-null",)
    n_null = sides.count("null")
    reason = haplotype_reason(deletion) if n_null else None
    base = {
        "name": f"{gene} {'/'.join(sorted(sides, reverse=True))}",
        "locus": gene,
        "zygosity": "carrier" if carrier else "diploid",
        "target": "Haplotype",
    }
    if reason is not None:
        return Genotype(**base, status="unresolved", reason=reason)
    return Genotype(
        **base,
        status="resolved",
        copies=(Copy(allele=deletion, copies=n_null, min_copies=n_null),)
        if n_null
        else (),
        gene_copies_min=sides.count("non-null"),
    )


# --------------------------------------------------------------------------- #
# rsID loci
# --------------------------------------------------------------------------- #


def is_snv_site(site: Sequence[CanonicalVariant]) -> bool:
    """Every alternative a single base at one position."""
    return _snv_ref(site) is not None


def _snv_ref(site: Sequence[CanonicalVariant]) -> str | None:
    if not site or not all(v.is_snv for v in site):
        return None
    if len({(v.seq_id, v.pos, v.ref) for v in site}) != 1:
        return None
    return site[0].ref


def _expand_snv(parsed: ParsedGenotype) -> tuple[str, ...] | str:
    """Bases of an SNV genotype ('CT' -> C, T), or why they cannot be read."""
    if parsed.slashed:
        if any(len(s) != 1 for s in parsed.sides):
            return "not_snv_alleles"
        return parsed.sides
    (token,) = parsed.sides
    if token == _DEL or len(token) > 2:  # noqa: PLR2004 -- 'CT'
        return "not_snv_alleles"
    return tuple(token)


def strand_ambiguous(
    genotypes: Sequence[ParsedGenotype], site: Sequence[CanonicalVariant]
) -> bool:
    """True when the annotation's bases, read on the reverse strand, are also a
    plausible report at this SNV site -- the forward-strand assumption then
    decides what was tested and cannot be checked (as in allele inference)."""
    ref = _snv_ref(site)
    if ref is None:
        return False
    bases: set[str] = set()
    for p in genotypes:
        tokens = _expand_snv(p)
        if not isinstance(tokens, str):
            bases.update(tokens)
    if not bases:
        return False
    allowed = {ref, *(v.alt for v in site)}
    flipped = {b.translate(_COMPLEMENT) for b in bases}
    return flipped <= allowed and len(flipped - {ref}) <= 1


#: Site allele -> the variant it is, None for the reference.
_Alleles = dict[str, "CanonicalVariant | None"]


def _site_genotype(
    sides: Sequence[str] | str,
    mapping: Mapping[str, CanonicalVariant | None] | str,
    *,
    raw: Sequence[str],
    rsid: str,
    site: Sequence[CanonicalVariant],
    spell: Callable[[CanonicalVariant | None], str],
) -> Genotype:
    """A site genotype from its allele tokens and their reading."""
    carrier = len(raw) == 1 if isinstance(sides, str) else len(sides) == 1
    unresolved = sides if isinstance(sides, str) else mapping
    if isinstance(unresolved, str):
        return Genotype(
            name=f"{rsid}:{'/'.join(sorted(raw))}",
            locus=rsid,
            zygosity="carrier" if carrier else "diploid",
            status="unresolved",
            reason=unresolved,
            target="Variant",
        )
    assert not isinstance(sides, str) and not isinstance(mapping, str)
    alleles = [mapping[t] for t in sides]
    counts = Counter(v.hgvs_g for v in alleles if v is not None)
    return Genotype(
        name=f"{rsid}:{'/'.join(sorted(spell(v) for v in alleles))}",
        locus=rsid,
        zygosity="carrier" if carrier else "diploid",
        status="resolved",
        target="Variant",
        copies=tuple(
            Copy(
                allele=h,
                copies=None if carrier else n,
                min_copies=n,
            )
            for h, n in sorted(counts.items())
        ),
        ref_copies=None if carrier else sum(v is None for v in alleles),
        hgvs_g38=_site_hgvs(alleles, site, carrier),
    )


def resolve_snv(
    parsed: ParsedGenotype, rsid: str, site: Sequence[CanonicalVariant]
) -> Genotype:
    """A genotype at an SNV site (every alternative a single base, one position)."""
    ref = _snv_ref(site)
    assert ref is not None
    tokens = _expand_snv(parsed)
    by_alt = {v.alt: v for v in site}
    mapping: _Alleles | str = (
        "allele_strand_mismatch"
        if not isinstance(tokens, str) and not set(tokens) <= {ref, *by_alt}
        else {ref: None, **by_alt}
    )
    return _site_genotype(
        tokens,
        mapping,
        raw=tokens if not isinstance(tokens, str) else parsed.sides,
        rsid=rsid,
        site=site,
        spell=lambda v: ref if v is None else v.alt,
    )


def unresolved_site(parsed: ParsedGenotype, rsid: str, reason: str) -> Genotype:
    """A genotype at an rsID that did not resolve: kept as written."""
    return _site_genotype(
        reason, reason, raw=parsed.sides, rsid=rsid, site=(), spell=lambda v: ""
    )


# ----- indel / multi-base sites -------------------------------------------- #


def _inserted(shorter: str, longer: str, unit: str) -> bool:
    return any(
        longer == shorter[:i] + unit + shorter[i:] for i in range(len(shorter) + 1)
    )


def _presence_pair(
    unit: str | None, ref: str, seqs: Sequence[str]
) -> tuple[str, str] | None:
    """The (shorter, longer) site alleles that ``unit`` present / absent tells apart.

    One of them is the reference; with ``unit`` None (a lone ``del``) any
    length difference qualifies. Prefers the reading where the reference
    carries ``unit`` (a deletion) over an insertion; None unless unique.
    """
    pairs = [
        (s, longer)
        for s in seqs
        for longer in seqs
        if len(longer) > len(s)
        and ref in {s, longer}
        and (unit is None or _inserted(s, longer, unit))
    ]
    preferred = [p for p in pairs if p[1] == ref] or [p for p in pairs if p[0] == ref]
    return preferred[0] if len(preferred) == 1 else None


def _split(token: str, vocab: set[str]) -> tuple[str, str] | None:
    splits = [
        (token[:i], token[i:])
        for i in range(1, len(token))
        if token[:i] in vocab and token[i:] in vocab
    ]
    return splits[0] if len(splits) == 1 else None


def _expand_indel(
    parsed: ParsedGenotype, vocab: set[str], seqs: set[str]
) -> tuple[str, ...]:
    """Tokens of one genotype; compact ones split only into known tokens."""
    if parsed.slashed:
        return parsed.sides
    (token,) = parsed.sides
    if token == _DEL or token in vocab:
        return (token,)
    split = _split(token, vocab) or (None if token in seqs else _split(token, seqs))
    return split or (token,)


def _spelled_pair(unit: str | None, seqs: Sequence[str]) -> tuple[str, str] | None:
    """``X/del`` with X spelled as a whole site allele: del is the one allele
    shorter than X (``TTAAAGTTA/del`` at a site whose only shorter allele is
    ``TTA``)."""
    if unit is None or unit not in seqs:
        return None
    shorter = [s for s in seqs if len(s) < len(unit)]
    return (shorter[0], unit) if len(shorter) == 1 else None


def _read_tokens(
    tokens: set[str], ref: str, alleles: _Alleles, vocab: set[str]
) -> _Alleles | str:
    """Token -> site allele for one annotation's indel notation."""
    seqs = sorted(alleles, key=len)
    units = tokens - {_DEL}
    if _DEL in tokens:
        if len(units) > 1:
            return "indel_notation_unmatched"
        unit = next(iter(units), None)
        pair = _presence_pair(unit, ref, seqs) or (
            _spelled_pair(unit, seqs) if unit in vocab else None
        )
        if pair is None:
            return "indel_notation_unmatched"
        by_presence: _Alleles = {_DEL: alleles[pair[0]]}
        if unit is not None:
            by_presence[unit] = alleles[pair[1]]
        return by_presence
    reading: _Alleles = {}
    for unit in units:
        exact = unit if unit in alleles else None
        pair = _presence_pair(unit, ref, seqs)
        present = pair[1] if pair else None
        if exact is None and present is None:
            return "indel_notation_unmatched"
        if exact is not None and present is not None and exact != present:
            return "ambiguous_indel_notation"
        reading[unit] = alleles[exact or present or ""]
    return reading


def read_indel_site(
    genotypes: Sequence[ParsedGenotype], site: Sequence[CanonicalVariant]
) -> tuple[list[tuple[str, ...]], _Alleles] | str:
    """Read every genotype of one annotation at a non-SNV site together.

    Returns each genotype's tokens and the token -> allele reading, or why the
    notation does not fit the site's alleles.
    """
    if len({(v.seq_id, v.pos, v.ref) for v in site}) != 1:
        return "site_alleles_not_aligned"
    ref = site[0].ref
    alleles: _Alleles = {ref: None, **{v.alt: v for v in site}}
    vocab = {s for p in genotypes if p.slashed for s in p.sides}
    expanded = [_expand_indel(p, vocab, set(alleles)) for p in genotypes]
    reading = _read_tokens(
        {t for sides in expanded for t in sides}, ref, alleles, vocab
    )
    if isinstance(reading, str):
        return reading
    return expanded, reading


def resolve_indel(
    parsed: ParsedGenotype,
    tokens: tuple[str, ...],
    reading: Mapping[str, CanonicalVariant | None] | str,
    rsid: str,
    site: Sequence[CanonicalVariant],
) -> Genotype:
    return _site_genotype(
        tokens,
        reading,
        raw=parsed.sides,
        rsid=rsid,
        site=site,
        spell=lambda v: "=" if v is None else v.change,
    )


def _site_hgvs(
    alleles: Sequence[CanonicalVariant | None],
    site: Sequence[CanonicalVariant],
    carrier: bool,
) -> str | None:
    reference = site[0].reference_change
    sides: list[tuple[str, ...]] = []
    for v in alleles:
        if v is not None:
            sides.append((v.change,))
        elif reference is None:
            return None
        else:
            sides.append((reference,))
    sides.sort(key=lambda s: s[0].endswith("="))  # alternative first, ref second
    return diploid_hgvs(site[0].seq_id, sides[0], None if carrier else sides[1])


__all__ = [
    "METHOD",
    "Copy",
    "Genotype",
    "HaplotypeHgvs",
    "Locus",
    "ParsedGenotype",
    "Role",
    "annotation_locus",
    "is_snv_site",
    "parse_field",
    "read_indel_site",
    "resolve_indel",
    "resolve_null",
    "resolve_snv",
    "resolve_star",
    "strand_ambiguous",
    "unresolved_site",
]
