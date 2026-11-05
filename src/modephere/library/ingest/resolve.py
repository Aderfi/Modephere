"""Resolver: SMILES | CID | name | ChEMBL ID -> :class:`~modephere.types.ResolvedCompound`.

* SMILES queries resolve locally; no upstream is asked.
* ChEMBL ID queries resolve offline from the staged ChEMBL release.
* Name queries try the ChEMBL release first (preferred names and synonyms;
  curated, offline), then PubChem's ``name -> CIDs``. With several matches
  the :class:`~modephere.types.AmbiguityPolicy` decides (default: the best
  hit, recorded as a warning on the record).
* CIDs (given or chosen) fetch their PubChem properties in batches.

Every resolution is then tied to a ChEMBL *parent* when the release knows the
compound: by the ID or name it was resolved from, else by the InChIKey
PubChem reported, else by the standard InChIKey of the structure. The parent
is where ChEMBL's annotations and bioactivities live, so a salt name such as
"imatinib mesylate" maps to imatinib (CHEMBL941).

Per-query failures are returned as :class:`~modephere.core.CompoundError`, so
one unresolvable name never stops a batch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from rdkit import Chem, rdBase

from modephere.core import CompoundError, ConfigurationError
from modephere.types import (
    AmbiguityPolicy,
    BuildStage,
    ChemblMatch,
    ChemblQuery,
    CidQuery,
    FailureCategory,
    NameQuery,
    ResolutionSource,
    ResolvedCompound,
    SmilesQuery,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from modephere.chembl.lookup import ChemblHit, ChemblLookup
    from modephere.library.ingest.pubchem import Lookup, PubChemClient
    from modephere.types import CompoundQuery

logger = logging.getLogger(__name__)

_LOOKUP_FAILURE = {
    "not_found": FailureCategory.UNRESOLVED_IDENTIFIER,
    "missing": FailureCategory.CACHE_MISS,
    "error": FailureCategory.UPSTREAM_ERROR,
}

type Resolution = ResolvedCompound | CompoundError


def _fail(category: FailureCategory, reason: str) -> CompoundError:
    return CompoundError(BuildStage.RESOLVE, category, reason)


def _lookup_failure(query: CompoundQuery, lookup: Lookup[object]) -> CompoundError:
    category = _LOOKUP_FAILURE[lookup.status]
    detail = f": {lookup.message}" if lookup.message else ""
    return _fail(category, f"{query.key} {lookup.status} on PubChem{detail}")


def structure_inchikey(smiles: str) -> str | None:
    """Standard InChIKey of a SMILES as written (``None`` if RDKit cannot)."""
    with rdBase.BlockLogs():  # scoped: leaves the global RDKit log setup alone
        mol = Chem.MolFromSmiles(smiles)
        return (Chem.MolToInchiKey(mol) or None) if mol is not None else None


class Resolver:
    """Resolve queries, batching every PubChem property request.

    ``chembl`` is the staged release's :class:`~modephere.chembl.ChemblLookup`;
    without it ChEMBL ID queries fail, names go straight to PubChem and no
    compound gets a ``chembl_id``.
    """

    def __init__(
        self,
        client: PubChemClient | None,
        *,
        ambiguity: AmbiguityPolicy = AmbiguityPolicy.FIRST_HIT,
        chembl: ChemblLookup | None = None,
    ) -> None:
        self._client = client
        self._ambiguity = ambiguity
        self._chembl = chembl

    def resolve(self, query: CompoundQuery) -> ResolvedCompound:
        """Resolve one query; raises :class:`~modephere.core.CompoundError`."""
        result = self.resolve_many([query])[0]
        if isinstance(result, CompoundError):
            raise result
        return result

    def needs_pubchem(self, query: CompoundQuery) -> bool:
        """Whether resolving ``query`` asks PubChem (``fetch`` skips the rest)."""
        if isinstance(query, CidQuery):
            return True
        if isinstance(query, NameQuery):
            return self._chembl is None or not self._chembl.by_name(query.name)
        return False

    def resolve_many(self, queries: Sequence[CompoundQuery]) -> list[Resolution]:
        """Resolve in input order; each item is a compound or its failure."""
        results: dict[int, Resolution] = {}
        chosen: dict[int, tuple[int, tuple[int, ...], tuple[str, ...]]] = {}

        for i, q in enumerate(queries):
            if isinstance(q, SmilesQuery):
                results[i] = self._build(
                    {"query": q, "smiles": q.smiles, "source": ResolutionSource.LOCAL}
                )
            elif isinstance(q, ChemblQuery):
                results[i] = self._from_chembl_id(q)
            elif isinstance(q, CidQuery):
                chosen[i] = (q.cid, (), ())
            else:
                by_chembl = self._from_chembl_name(q)
                if by_chembl is not None:
                    results[i] = by_chembl
                    continue
                outcome = self._choose_cid(q)
                if isinstance(outcome, CompoundError):
                    results[i] = outcome
                else:
                    chosen[i] = outcome

        if chosen:
            client = self._require_client()
            lookups = client.properties([cid for cid, _, _ in chosen.values()])
            for i, (cid, candidates, warnings) in chosen.items():
                lookup = lookups[cid]
                if lookup.status != "ok" or lookup.value is None:
                    results[i] = _lookup_failure(queries[i], lookup)
                    continue
                record = lookup.value
                results[i] = self._build(
                    {
                        "query": queries[i],
                        "smiles": record.smiles,
                        "source": ResolutionSource.PUBCHEM,
                        "cid": record.cid,
                        "inchikey": record.inchikey,
                        "title": record.title,
                        "candidate_cids": candidates,
                        "retrieved_at": lookup.retrieved_at,
                        "warnings": warnings,
                    }
                )
        return [results[i] for i in range(len(queries))]

    # ----- construction ---------------------------------------------------- #

    def _build(self, fields: dict[str, Any]) -> Resolution:
        """Validate a resolution, tying it to a ChEMBL parent when possible."""
        if "chembl_id" not in fields:
            fields |= self._chembl_by_inchikey(fields)
        try:
            return ResolvedCompound(**fields)
        except ValidationError as e:  # e.g. PubChem answered another CID
            query: CompoundQuery = fields["query"]
            return _fail(
                FailureCategory.UNRESOLVED_IDENTIFIER,
                f"{query.key}: inconsistent upstream record: {e.errors()[0]['msg']}",
            )

    def _chembl_fields(self, hit: ChemblHit, match: ChemblMatch) -> dict[str, Any]:
        assert self._chembl is not None
        return {
            "chembl_id": hit.parent_chembl_id,
            "chembl_record_id": hit.chembl_id,
            "chembl_match": match,
            "chembl_release": self._chembl.release,
        }

    def _chembl_by_inchikey(self, fields: dict[str, Any]) -> dict[str, Any]:
        if self._chembl is None:
            return {}
        keys = [
            (fields.get("inchikey"), ChemblMatch.UPSTREAM_INCHIKEY),
            (structure_inchikey(fields["smiles"]), ChemblMatch.STRUCTURE_INCHIKEY),
        ]
        for key, match in keys:
            if key and (hits := self._chembl.by_inchikey(key)):
                return self._chembl_fields(hits[0], match)
        return {}

    def _chembl_resolution(
        self,
        query: CompoundQuery,
        hit: ChemblHit,
        match: ChemblMatch,
        *,
        candidates: tuple[str, ...] = (),
        warnings: tuple[str, ...] = (),
    ) -> Resolution:
        assert self._chembl is not None
        parent = self._chembl.parent_of(hit)
        if parent.smiles is None:
            return _fail(
                FailureCategory.UNSUPPORTED_CHEMISTRY,
                f"{query.key}: ChEMBL {parent.chembl_id} has no small-molecule "
                "structure (biologic or mixture)",
            )
        return self._build(
            {
                "query": query,
                "smiles": parent.smiles,
                "source": ResolutionSource.CHEMBL,
                "inchikey": parent.inchikey,
                "title": parent.pref_name or hit.pref_name,
                "retrieved_at": self._chembl.released_at,
                "candidate_chembl_ids": candidates,
                "warnings": warnings,
                **self._chembl_fields(hit, match),
            }
        )

    # ----- sources ---------------------------------------------------------- #

    def _from_chembl_id(self, query: ChemblQuery) -> Resolution:
        if self._chembl is None:
            msg = "ChEMBL ID queries need a staged ChEMBL release (python -m modephere.chembl extract)"
            raise ConfigurationError(msg)
        hit = self._chembl.by_chembl_id(query.chembl_id)
        if hit is None:
            return _fail(
                FailureCategory.UNRESOLVED_IDENTIFIER,
                f"{query.key} is not in the staged {self._chembl.release} selection",
            )
        return self._chembl_resolution(query, hit, ChemblMatch.CHEMBL_ID)

    def _from_chembl_name(self, query: NameQuery) -> Resolution | None:
        """A ChEMBL resolution, a failure, or ``None`` to try PubChem."""
        if self._chembl is None:
            return None
        hits = self._chembl.by_name(query.name)
        if not hits:
            return None
        parents = tuple(dict.fromkeys(h.parent_chembl_id for h in hits))
        warnings: tuple[str, ...] = ()
        if len(parents) > 1:
            if self._ambiguity is AmbiguityPolicy.FAIL:
                return _fail(
                    FailureCategory.AMBIGUOUS_IDENTIFIER,
                    f"{query.key} names {len(parents)} ChEMBL parents: {parents[:10]}",
                )
            warning = f"name matches {len(parents)} ChEMBL parents; took {parents[0]}"
            logger.warning("%s: %s", query.key, warning)
            warnings = (warning,)
        return self._chembl_resolution(
            query, hits[0], ChemblMatch.NAME, candidates=parents, warnings=warnings
        )

    def _require_client(self) -> PubChemClient:
        if self._client is None:
            msg = "CID queries and names unknown to ChEMBL need a PubChemClient"
            raise ConfigurationError(msg)
        return self._client

    def _choose_cid(
        self, query: NameQuery
    ) -> tuple[int, tuple[int, ...], tuple[str, ...]] | CompoundError:
        lookup = self._require_client().name_to_cids(query.name)
        if lookup.status != "ok" or not lookup.value:
            return _lookup_failure(query, lookup)
        cids = lookup.value
        if len(cids) == 1:
            return cids[0], cids, ()
        if self._ambiguity is AmbiguityPolicy.FAIL:
            return _fail(
                FailureCategory.AMBIGUOUS_IDENTIFIER,
                f"{query.key} matches {len(cids)} CIDs: {cids[:10]}",
            )
        warning = f"name matches {len(cids)} CIDs; took PubChem's first ({cids[0]})"
        logger.warning("%s: %s", query.key, warning)
        return cids[0], cids, (warning,)


__all__ = ["Resolution", "Resolver", "structure_inchikey"]
