"""Modephere: molecule and 2D/3D pharmacophore representations.

Start from SMILES, PubChem CIDs or drug names; get validated molecular graphs,
pharmacophore graphs, conformer ensembles, 3D feature points and
fingerprints, stored as a resumable library and ready for PyTorch Geometric.

* :mod:`modephere.types` — Pydantic schemas (the contracts between layers)
* :mod:`modephere.library` — the builder (``python -m modephere.library``)
* :mod:`modephere.featurize` — PyG ``Data`` / ``HeteroData`` (``ml`` extra)
"""

from __future__ import annotations


def main() -> int:
    """``modephere`` console script: the library CLI."""
    from modephere.library.__main__ import main as library_main

    return library_main()
