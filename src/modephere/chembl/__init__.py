"""ChEMBL: local SQLite release -> staging Parquet (and from there the graph).

* :mod:`~modephere.chembl.source` — read-only access to ``chembl_<N>.db``;
* :mod:`~modephere.chembl.extract` — the drug / clinical selection, raw
  activities and per (parent, target) bioactivity summaries as Parquet;
* :mod:`~modephere.chembl.lookup` — offline name / ChEMBL ID -> structure;
* :mod:`~modephere.chembl.labels` — bioactivity labels for ``LibraryDataset``.

ChEMBL is CC BY-SA 3.0: keep ChEMBL IDs and the release with derived data.
"""

from modephere.chembl.extract import (
    StagingManifest,
    aggregate_bioactivity,
    eligible_activities,
    extract,
    read_table,
)
from modephere.chembl.labels import bioactivity_labels
from modephere.chembl.lookup import ChemblHit, ChemblLookup
from modephere.chembl.source import LICENSE, ChemblRelease

__all__ = [
    "LICENSE",
    "ChemblHit",
    "ChemblLookup",
    "ChemblRelease",
    "StagingManifest",
    "aggregate_bioactivity",
    "bioactivity_labels",
    "eligible_activities",
    "extract",
    "read_table",
]
