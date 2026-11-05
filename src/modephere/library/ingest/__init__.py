"""Ingestion: catalogs -> queries -> resolved structures (PubChem or local).

Every network response is recorded in a raw cache first, so a build can run
fully offline from it (``python -m modephere.library fetch`` fills it).
"""

from modephere.library.ingest.catalog import (
    CatalogLoad,
    load_catalog,
    queries_from_values,
)
from modephere.library.ingest.pubchem import Lookup, PubChemClient, PubChemRecord
from modephere.library.ingest.raw_cache import CacheEntry, RawCache
from modephere.library.ingest.resolve import Resolution, Resolver

__all__ = [
    "CacheEntry",
    "CatalogLoad",
    "Lookup",
    "PubChemClient",
    "PubChemRecord",
    "RawCache",
    "Resolution",
    "Resolver",
    "load_catalog",
    "queries_from_values",
]
