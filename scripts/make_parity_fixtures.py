"""Write dev_Pgen golden outputs for the Modephere parity tests (PLAN M3).

Run with dev_Pgen's interpreter, from anywhere::

    ~/Projects/dev_Pgen/.venv/bin/python scripts/make_parity_fixtures.py \
        --dev-pgen ~/Projects/dev_Pgen --out tests/fixtures/dev_pgen_parity.npz

For every SMILES in ``PANEL`` it stores dev_Pgen's drug graph and pharmacophore
graph (``strip_salts=True``) as numpy arrays keyed ``<i>/<attr>``. The RDKit
version used is stored too: parity only holds on the same RDKit release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PANEL: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CN(C)C(=N)N=C(N)N",  # metformin (guanidine)
    "CN=C(NC#N)NCCSCc1nc[nH]c1C",  # cimetidine (cyanoguanidine, imidazole)
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # warfarin (4-hydroxycoumarin)
    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",  # meloxicam (vinylogous acid)
    "CN1CCNCC1",  # N-methylpiperazine (protonates once)
    "c1c[nH]cn1",  # imidazole
    "CC(=O)[O-].[Na+]",  # sodium acetate (salt stripping)
    "FS(F)(F)(F)(F)F",  # SF6 (one-hot saturation)
    "O=[N+]([O-])c1ccccc1",  # nitrobenzene (zwitterionic notation)
    "[O-][n+]1ccccc1",  # pyridine N-oxide
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",  # morphine
    "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",  # losartan (tetrazole)
    "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",  # sulfamethoxazole
    "COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1",  # glibenclamide
    "OCC1OC(O)C(O)C(O)C1O",  # glucose
    "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",  # testosterone
    "O=C1Cc2ccccc2N1",  # oxindole (tautomer must stay)
    "NCCN",  # ethylenediamine
    "c1ccccc1",  # benzene
    "CC(C)(C)C1CCCCC1",  # lumped hydrophobes
    "N",  # ammonia: one feature
    "C",  # methane: no feature
)

_DRUG_ATTRS = ("x", "edge_index", "edge_attr", "global_feats")
_PHARM_ATTRS = (*_DRUG_ATTRS, "atom2feat")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-pgen", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(args.dev_pgen.resolve()))
    import numpy as np
    import rdkit
    from src.data.library.drugs import smiles_to_graph
    from src.data.library.pharmacophore import smiles_to_pharmacophore_graph

    arrays: dict[str, np.ndarray] = {}
    for i, smiles in enumerate(PANEL):
        drug = smiles_to_graph(smiles, strip_salts=True)
        pharm = smiles_to_pharmacophore_graph(smiles, strip_salts=True)
        if drug is None or pharm is None:
            msg = f"dev_Pgen could not parse panel SMILES {smiles!r}"
            raise SystemExit(msg)
        for attr in _DRUG_ATTRS:
            arrays[f"{i}/drug/{attr}"] = getattr(drug, attr).numpy()
        for attr in _PHARM_ATTRS:
            arrays[f"{i}/pharm/{attr}"] = getattr(pharm, attr).numpy()
        arrays[f"{i}/pharm/num_drug_atoms"] = np.array(pharm.num_drug_atoms)

    meta = {"rdkit": rdkit.__version__, "panel": list(PANEL)}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {len(PANEL)} panel entries to {args.out} (rdkit {rdkit.__version__})")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
