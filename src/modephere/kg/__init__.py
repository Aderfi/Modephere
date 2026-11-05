"""Neo4j knowledge graph: ChEMBL drugs, targets, genes, diseases, bioactivity.

Needs the ``kg`` extra (``uv sync --extra kg``) and a running Neo4j
(``docker compose -f docker-compose.kg.yml up -d``). The model is in
:mod:`modephere.kg.schema`; provenance follows dev_Pgen's ``src.kg``.
"""

from modephere.kg.chembl import GraphRows, LoadSummary, chembl_rows, load_chembl
from modephere.kg.graph import KnowledgeGraph, RelSpec
from modephere.kg.provenance import Dataset, IngestRun, Stamp
from modephere.kg.schema import NODE_KEYS, REL_TYPES

__all__ = [
    "NODE_KEYS",
    "REL_TYPES",
    "Dataset",
    "GraphRows",
    "IngestRun",
    "KnowledgeGraph",
    "LoadSummary",
    "RelSpec",
    "Stamp",
    "chembl_rows",
    "load_chembl",
]
