"""Compound queries: what the caller asks the library to build.

A query is one of four shapes, discriminated by ``kind``::

    >>> COMPOUND_QUERY.validate_python({"kind": "cid", "cid": "2244"})
    CidQuery(label=None, kind=<IdentifierKind.CID: 'cid'>, cid=2244)

The kind is always explicit. Guessing it from the value is unsafe: ``"CCO"`` is
a valid SMILES and could also be a name, and a bare number could be a CID or a
name.

``query.key`` is the request's cache and manifest key (``"cid:2244"``,
``"name:aspirin"``). It identifies the *request*, not the molecule: two
queries for one compound have different keys and meet at the InChIKey after
standardization.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from modephere.types.base import FrozenModel
from modephere.types.enums import IdentifierKind
from modephere.types.scalars import ChemblID, NonEmptyStr, PubChemCID, SmilesStr


class _QueryBase(FrozenModel):
    label: NonEmptyStr | None = Field(
        default=None,
        description="Caller's own row identifier, carried through the build.",
    )


class SmilesQuery(_QueryBase):
    """A structure given directly as SMILES."""

    kind: Literal[IdentifierKind.SMILES] = IdentifierKind.SMILES
    smiles: SmilesStr

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.smiles}"


class CidQuery(_QueryBase):
    """A PubChem Compound ID."""

    kind: Literal[IdentifierKind.CID] = IdentifierKind.CID
    cid: PubChemCID

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.cid}"


class NameQuery(_QueryBase):
    """A chemical or drug name (ChEMBL names first, then PubChem)."""

    kind: Literal[IdentifierKind.NAME] = IdentifierKind.NAME
    name: NonEmptyStr

    @property
    def key(self) -> str:
        # PubChem name lookup is case-insensitive; one key per spelling.
        return f"{self.kind}:{' '.join(self.name.casefold().split())}"


class ChemblQuery(_QueryBase):
    """A ChEMBL molecule ID (resolved offline from the staged release)."""

    kind: Literal[IdentifierKind.CHEMBL] = IdentifierKind.CHEMBL
    chembl_id: ChemblID

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.chembl_id}"


type CompoundQuery = Annotated[
    SmilesQuery | CidQuery | NameQuery | ChemblQuery, Field(discriminator="kind")
]

#: Validator for raw query input (dicts, catalog rows, JSON).
COMPOUND_QUERY: TypeAdapter[CompoundQuery] = TypeAdapter(CompoundQuery)

_VALUE_FIELD: dict[IdentifierKind, str] = {
    IdentifierKind.SMILES: "smiles",
    IdentifierKind.CID: "cid",
    IdentifierKind.NAME: "name",
    IdentifierKind.CHEMBL: "chembl_id",
}


def parse_query(
    kind: IdentifierKind | str, value: object, *, label: str | None = None
) -> CompoundQuery:
    """Build a query from a ``(kind, value)`` pair, e.g. a two-column catalog row.

    Raises :class:`pydantic.ValidationError` for an unknown kind or bad value.
    """
    payload: dict[str, object] = {"kind": kind, "label": label}
    try:
        payload[_VALUE_FIELD[IdentifierKind(kind)]] = value
    except ValueError:
        pass  # unknown kind: the discriminator reports it
    return COMPOUND_QUERY.validate_python(payload)


__all__ = [
    "COMPOUND_QUERY",
    "ChemblQuery",
    "CidQuery",
    "CompoundQuery",
    "NameQuery",
    "SmilesQuery",
    "parse_query",
]
