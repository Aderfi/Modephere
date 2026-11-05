● Modephere: Phase 1 Discovery and Architecture Proposal

  I haven't written any code or files.

  Before the plan, the main finding: ~/Projects/dev_Pgen/src/data/library already contains a mature
  version of half of this project. drugs.py turns SMILES into a 61/18-dim atom graph. pharmacophore.py
  perceives six pharmacophore families on the pH 7.4 form of the molecule and builds a topological
  feature graph from them. So Modephere should be a 3D-capable successor that reuses those designs,
  not a rewrite.

  A. Conventions to replicate

  ┌────────────┬────────────────────────────────────────────────────────────┬─────────────────────┐
  │    Area    │                    dev_Pgen convention                     │     Carry over?     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │ Runtime    │ Python >=3.14,<3.15, uv, ruff.toml (py314, line 88, rules  │ Yes: copy ruff.toml │
  │            │ PL,E,W,I,F,UP,FA,TC), mypy check_untyped_defs              │  as is              │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ from __future__ import annotations everywhere; type-only   │                     │
  │ Imports    │ imports go in if TYPE_CHECKING:; the package __init__      │ Yes                 │
  │            │ loads exports lazily via __getattr__                       │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ X | None, list[...], type aliases with Annotated[int,      │                     │
  │ Typing     │ Field(gt=0)] (PubChemCID), StrEnum for codes,              │ Yes                 │
  │            │ Literal["built","skipped","failed"] return values          │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ Pydantic v2 at every boundary; build config is a           │                     │
  │ Validation │ frozen=True model with a from_settings() classmethod;      │ Yes                 │
  │            │ plain dataclasses only for internal results (BuildSummary) │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ pydantic-settings loads TOML (config/data/*.toml) plus env │ Yes, with a         │
  │ Config     │  vars (PREFIX_A__B), cached by @lru_cache get_settings();  │ MODEPHERE_ prefix   │
  │            │ paths come from Settings.paths and are never hard-coded    │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ One root exception (PharmagenException(message, details))  │ Yes: a              │
  │ Errors     │ with subtypes DataError → GraphError/BioinformaticsError   │ ModephereException  │
  │            │                                                            │ tree                │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ logging.getLogger(__name__), %s formatting, no emoji, no   │                     │
  │ Logging    │ print. Each audit report gets its own non-propagating      │ Yes                 │
  │            │ FileHandler logger (_build_file_logger)                    │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ Each XGraphBuilder.build(path, manifest=) → (built,        │                     │
  │ Builders   │ skipped, failed) call is resumable (skips existing outputs │ Yes                 │
  │            │  unless force), counts failures by a FailureCategory       │                     │
  │            │ StrEnum, and logs a summary at the end                     │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │ Manifest   │ A Pydantic BuildManifest, written atomically (write a temp │ Yes                 │
  │            │  file, then rename) after each stage                       │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ RawCache keeps every upstream response verbatim in an      │                     │
  │ Network    │ append-only JSONL file; builds then run offline from it.   │ Yes, for PubChem    │
  │            │ Clients get an injectable session and sleep for tests      │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │ Schemas    │ Feature widths are exported constants (PHARM_NODE_DIM =    │ Yes                 │
  │            │ 11) and documented in the module docstring. Tests pin them │                     │
  ├────────────┼────────────────────────────────────────────────────────────┼─────────────────────┤
  │            │ tests/unit/<area>/test_*.py, named SMILES constants        │ Yes, plus a network │
  │ Tests      │ (ASPIRIN, CAFFEINE), tmp_path TSV fixtures,                │  marker             │
  │            │ --strict-markers, markers slow/integration                 │                     │
  └────────────┴────────────────────────────────────────────────────────────┴─────────────────────┘

  Things I'd deliberately not copy:
  1. Private cross-module imports. pharmacophore.py imports _largest_fragment, _one_hot and
     _build_file_logger from drugs.py. Those should live in a shared module.
  2. Plain dicts at the input boundary. DrugRow = dict[str, Any] breaks the project's own rule against
     dict[str, Any] at boundaries. Use a Pydantic record instead.
  3. Pickled outputs. Library outputs are saved with torch.save (pickle), so loading them on torch
     ≥2.6 needs weights_only=False. The project already prefers safetensors for checkpoints.
  4. Import root. dev_Pgen imports from src.. Modephere's scaffold is a proper uv_build src/modephere/
     package, so imports become modephere.library…. I'd keep that.

  B. Ecosystem research (checked against PyPI and live probes today)

  B1. Resolving names and CIDs, and standardizing structures

  ┌─────────────────────────────────────┬──────────────────────────────────────┬──────────────────┐
  │               Option                │                Status                │     Verdict      │
  ├─────────────────────────────────────┼──────────────────────────────────────┼──────────────────┤
  │ Own PUG-REST client on httpx 0.28.1 │ —                                    │ Recommended      │
  │  + the RawCache pattern             │                                      │                  │
  ├─────────────────────────────────────┼──────────────────────────────────────┼──────────────────┤
  │                                     │ Maintained again, but synchronous    │                  │
  │ pubchempy 1.0.5 (Sep 2025)          │ urllib, with no throttling, caching  │ Reference only   │
  │                                     │ or test hooks                        │                  │
  ├─────────────────────────────────────┼──────────────────────────────────────┼──────────────────┤
  │ chembl_webresource_client 0.10.9    │                                      │ Optional         │
  │ (Feb 2024)                          │ Stale                                │ fallback for     │
  │                                     │                                      │ names            │
  ├─────────────────────────────────────┼──────────────────────────────────────┼──────────────────┤
  │ chembl_structure_pipeline 1.2.4     │ Active; picks the parent molecule    │ Optional         │
  │ (Nov 2025)                          │ from a salt list, which beats        │ standardizer     │
  │                                     │ "largest fragment"                   │                  │
  ├─────────────────────────────────────┼──────────────────────────────────────┼──────────────────┤
  │ cirpy (2016), molvs (2018)          │ Dead; MolVS now lives inside RDKit   │ No               │
  │                                     │ as rdMolStandardize                  │                  │
  └─────────────────────────────────────┴──────────────────────────────────────┴──────────────────┘

  Live checks against PubChem today:
  - PubChem renamed its SMILES properties. Asking for IsomericSMILES/CanonicalSMILES now silently
    returns nothing. The current names are SMILES and ConnectivitySMILES.
  - Every response carries an X-Throttling-Control header (green/yellow/red). The client should slow
    down based on it, on top of a hard cap of about 5 requests/s. I couldn't confirm that cap on
    PubChem's own policy page, which was behind a CAPTCHA; it comes from secondary sources.
  - CID lookups can be batched (comma-separated CIDs in one request). Name lookups can't. A name can
    also return several CIDs, so the client needs a rule for ambiguous names.

  Why async isn't worth it: at about 5 requests/s, async httpx gives no extra throughput. A
  synchronous httpx.Client with an injectable transport (httpx.MockTransport in tests) matches the
  NCBIClient pattern and stays simple.

  Conflict with your brief: tautomer canonicalization. dev_Pgen's pharmacophore.py deliberately does
  not use RDKit's canonical tautomer. Its comment says the canonical tautomer "is a canonical
  representative, not the dominant tautomer"; it turns oxindoles into hydroxyindoles and makes up
  donors that aren't there. I propose two representations of each molecule:
  - A registration form: the neutral parent in its canonical tautomer. It's used only for the identity
    key and deduplication (InChIKey).
  - A working form: the neutral parent in its input tautomer, then protonated for pH 7.4 using the
    ported rule-based _microspecies function. It's used for the 3D structures and the pharmacophore
    features.

  dimorphite_dl 2.1.0 (Oct 2026) could be an optional alternative protonation backend.

  B2. Cheminformatics and pharmacophores

  - RDKit: the latest is 2026.9.1; dev_Pgen has 2025.9.6. The 2026.03 release changed SMILES
    canonicalization for stereo double bonds, and 2026.09 changed how ring sets are symmetrized. So
    canonical SMILES aren't stable across versions: cache keys should use the InChIKey, and the
    manifest should record the RDKit version. 2026.09 ships ETKDGv4 as a beta; it isn't the default,
    so we'd use ETKDGv3 and make the method configurable. Chem.Fingerprints is deprecated, so we use
    rdFingerprintGenerator.
  - Pharm2D/Pharm3D: no changes in recent releases. I confirmed the modern import paths
    (rdDistGeom.ETKDGv3, rdForceFieldHelpers, rdMolAlign, Pharm2D.Gobbi_Pharm2D, Pharm3D.EmbedLib,
    rdMolStandardize.*) all import on 3.14.
  - OpenPharmacophore: last push July 2023 and not on PyPI. Don't depend on it.
  - ODDT: 0.7, released 2019. Dead.
  - PharmacoNet: MIT licensed, v2.1.1 from Dec 2024. It models pharmacophores from a protein pocket,
    which only helps if you have receptor structures.
  - MDAnalysis: 2.11 (Oct 2026) is active, but it's for MD trajectories. Not needed.

  B3. Deep-learning framework (Python 3.14 is the deciding constraint)

  ┌──────────────────────────┬─────────────────────────────────────┬─────────────────────────────┐
  │        Framework         │           On Python 3.14?           │           Verdict           │
  ├──────────────────────────┼─────────────────────────────────────┼─────────────────────────────┤
  │ PyTorch Geometric 2.8.1  │ Yes. PyG 2.8 supports Python        │                             │
  │ (Oct 2026) + torch 2.12  │ 3.10–3.14 and torch 2.9–2.12; it    │ Recommended                 │
  │ + pyg-lib 0.7            │ already powers dev_Pgen's drug      │                             │
  │                          │ tower                               │                             │
  ├──────────────────────────┼─────────────────────────────────────┼─────────────────────────────┤
  │ TensorFlow 2.21 / tf-gnn │ No. No cp314 wheels; tf-gnn has had │ Ruled out                   │
  │  1.0.3 (May 2024)        │  no release since 2024              │                             │
  ├──────────────────────────┼─────────────────────────────────────┼─────────────────────────────┤
  │ DeepChem 2.8.0 (Apr      │ No. It requires Python <3.12        │ Ruled out                   │
  │ 2024)                    │                                     │                             │
  ├──────────────────────────┼─────────────────────────────────────┼─────────────────────────────┤
  │ DGL 2.2.1 (May 2024) /   │ No. No cp314 wheels; effectively    │                             │
  │ DGL-LifeSci 0.3.2 (Feb   │ unmaintained                        │ Ruled out                   │
  │ 2023)                    │                                     │                             │
  ├──────────────────────────┼─────────────────────────────────────┼─────────────────────────────┤
  │ torchmd-net 3.1 / e3nn   │                                     │ Optional later, for         │
  │ 0.6                      │ Pure Python                         │ TensorNet or                │
  │                          │                                     │ Equiformer-style models     │
  └──────────────────────────┴─────────────────────────────────────┴─────────────────────────────┘

  3D graphs need PyG 2.8. On dev_Pgen's PyG 2.7.0, radius_graph fails today with "requires
  'torch-cluster'", and torch-cluster has no wheel for this torch/Python combination. PyG 2.8 drops
  torch-cluster and moves that code to pyg-lib>=0.6. So Modephere needs torch-geometric>=2.8.1, and
  dev_Pgen would have to upgrade if it ever imports Modephere.

  PyG 2.8 ships SchNet, DimeNet++ and ViSNet. PaiNN and Equiformer aren't in it.

  Recommended split: the chemistry layers produce plain numpy arrays and dataclasses and never import
  torch, so they're fast to test. A single thin adapter layer turns them into PyG Data/HeteroData.
  This keeps the expensive RDKit work independent of the framework.

  C. Proposed architecture

  C1. Layout (mirrors src/data/library, plus the core/config/domain layers it relies on)

  Modephere/
  ├── pyproject.toml   ruff.toml   .python-version (3.14)
  ├── src/modephere/
  │   ├── __init__.py                 # lazy exports
  │   ├── core/{__init__,exceptions,log}.py           # ModephereException tree, setup_logging
  │   ├── config/{__init__,settings,paths}.py
  │   │   └── data/{paths,settings}.toml              # MODEPHERE_* env overrides
  │   ├── domain/
  │   │   ├── base.py  types.py                       # PubChemCID, InChIKey, IdentifierKind(StrEnum)
  │   │   └── schemas/{compound,conformer,pharmacophore}.py
  │   └── library/                                    # python -m modephere.library
  │       ├── __init__.py  __main__.py
  │       ├── config.py        # LibraryBuildConfig (frozen, from_settings)
  │       ├── builder.py       # LibraryBuilder.run() -> BuildSummary
  │       ├── manifest.py      # BuildManifest (atomic JSON; records rdkit/pyg versions)
  │       ├── reporting.py     # FailureCategory StrEnum, build_file_logger, FeatureSaturation
  │       ├── ingest/
  │       │   ├── models.py    # CompoundQuery, ResolvedCompound (provenance, ambiguity)
  │       │   ├── catalog.py   # tsv/csv/json/parquet query lists -> CompoundQuery
  │       │   ├── raw_cache.py # append-only JSONL (port of src/kg/raw_cache.py)
  │       │   ├── pubchem.py   # PUG-REST client (throttle, retry, batched CIDs)
  │       │   ├── chembl.py    # optional fallback
  │       │   └── resolve.py   # Resolver: SMILES | CID | name -> ResolvedCompound
  │       ├── standardize.py   # parent, uncharge, registration key, pH 7.4 microspecies
  │       ├── conformers.py    # ConformerGenerator -> ConformerEnsemble
  │       ├── molecules.py     # 2D atom/bond featurizer (port of drugs.py schema 61/18/1038)
  │       ├── descriptors.py   # physchem + rdFingerprintGenerator
  │       ├── pharmacophore/
  │       │   ├── features.py     # 6-family perception (ported normalisation rules)
  │       │   ├── graph2d.py      # topological graph (dev_Pgen schema 11/13/258)
  │       │   ├── fingerprints.py # Gobbi Pharm2D + 3D distance-binned triplets
  │       │   └── points3d.py     # per-conformer feature points, vectors, distances
  │       ├── storage.py       # Parquet registry + safetensors shards
  │       └── featurize/       # the only package importing torch / torch_geometric
  │           ├── homogeneous.py  # Data (2D graph; 3D with pos)
  │           └── hetero.py       # HeteroData: atom + pharm nodes
  └── tests/{unit/library/..., integration/, fixtures/}

  C2. Pipeline layers

  1. Ingest and resolve:
     - A CompoundQuery(kind, value) goes to the Resolver.
     - SMILES: parsed locally. PubChem is only asked for the CID if you want one.
     - CID: batched property requests for SMILES, ConnectivitySMILES, InChIKey, IUPACName and Title.
     - Name: goes to /compound/name/{n}/cids, then follows the CID path. Ambiguous names follow a set
       rule (see Q3).
     - Every HTTP response goes into the RawCache first, and an offline mode raises CacheMissError.
     - Output: a ResolvedCompound, with provenance, passed through standardize.py to produce the
       registration form (InChIKey key) and the working form.
  2. Conformers:
     - AddHs, then EmbedMultipleConfs with ETKDGv3 (fixed randomSeed, numThreads=0, pruneRmsThresh;
       retry with useRandomCoords on failure).
     - Optimize with MMFFOptimizeMoleculeConfs using MMFF94s, falling back to UFF if MMFF can't
       parametrize the molecule.
     - Keep conformers within an energy window, e.g. 10 kcal/mol of the minimum.
     - Remove near-duplicates after optimization using symmetry-aware heavy-atom RMSD (GetBestRMS) and
       Butina clustering.
     - The conformer count scales with rotatable bonds.
     - Unspecified stereocentres follow your policy (Q5).
  3. Pharmacophore:
     - Perception runs once, on the working form, and the same family/atom-ID features feed all four
       outputs below.
     - (a) The 2D topological graph, schema-compatible with dev_Pgen.
     - (b) Gobbi 2D fingerprints.
     - (c) 3D feature points per conformer: a centroid and family for each feature, plus direction
       vectors for aromatic ring normals and donor/acceptor directions.
     - (d) A 3D distance-binned triplet fingerprint combined across the conformer ensemble.
     - Excluded volumes need a receptor, so they're out of scope unless you have one (Q1).
  4. ML featurization:
     - Data: x, edge_index, edge_attr, global_feats, plus pos sampled from a stored conf_pos [C,N,3]
       at load time, which doubles as conformer augmentation.
     - HeteroData:
       - atom–bond–atom edges from the molecular graph;
       - atom–member_of–pharm edges, the existing atom2feat mapping;
       - pharm–dist–pharm edges, with radial-basis-expanded 3D distances.
     - It feeds SchNet/ViSNet directly, or dev_Pgen's GATv2/GINE towers.

  C3. Dependencies (pyproject.toml sketch)

  requires-python = ">=3.14,<3.15"
  dependencies = [
    "pydantic>=2.12.5", "pydantic-settings>=2.14", "polars>=1.39", "numpy>=2.4",
    "rdkit>=2025.9.6,<2027",          # exact version recorded in manifest
    "httpx>=0.28.1", "tqdm",
    "safetensors>=0.8",
  ]
  [project.optional-dependencies]
  ml  = ["torch==2.12.1", "torch-geometric>=2.8.1,<2.9", "pyg_lib==0.7.0"]
  std = ["chembl_structure_pipeline>=1.2.4", "dimorphite_dl>=2.1"]
  dev = ["pytest>=8", "pytest-cov", "ruff", "mypy"]
  # [tool.uv.index] pytorch-cu132 + pyg flat index, copied from dev_Pgen

  Notes on these choices:
  - No tenacity: retries are hand-rolled to match ncbi.py.
  - No pyarrow: polars reads and writes Parquet itself.
  - torch/PyG are an optional extra, so the chemistry layers install and test without CUDA.
  - All of this works with uv or pip on Debian/WSL.

  C4. Roadmap (each milestone ends green on ruff check, ruff format --check and pytest)

  ┌───────────┬──────────────────────────────────┬────────────────────────────────────────────────┐
  │ Milestone │              Scope               │                   Checkpoint                   │
  ├───────────┼──────────────────────────────────┼────────────────────────────────────────────────┤
  │ M0        │ Scaffold: core/config/domain,    │ Settings load; exception tests pass            │
  │           │ ruff/mypy/pytest config, CI      │                                                │
  ├───────────┼──────────────────────────────────┼────────────────────────────────────────────────┤
  │           │ Ingest: models, catalog, raw     │ MockTransport tests (throttle header, 503      │
  │ M1        │ cache, PubChem client, resolver  │ retry, ambiguous name, renamed properties);    │
  │           │                                  │ offline replay                                 │
  ├───────────┼──────────────────────────────────┼────────────────────────────────────────────────┤
  │ M2        │ Standardize                      │ Golden tests: salts, zwitterions, oxindole     │
  │           │                                  │ tautomer kept, InChIKey stable                 │
  ├───────────┼──────────────────────────────────┼────────────────────────────────────────────────┤
  │ M3        │ 2D graph + descriptors + 2D      │ Parity tests: byte-identical to dev_Pgen       │
  │           │ pharmacophore (ports)            │ outputs on its fixture set; dims pinned        │
  ├───────────┼──────────────────────────────────┼────────────────────────────────────────────────┤
  │ M4        │ Conformers                       │ Determinism under a fixed seed; RMSD pruning;  │
  │           │                                  │ MMFF→UFF fallback; benchmark vs. 8 cores       │
  ├───────────┼──────────────────────────────────┼────────────────────────────────────────────────┤
  │ M5        │ 3D pharmacophore + fingerprints  │ Translation/rotation invariance tests; same    │
  │           │                                  │ features across conformers                     │
  ├───────────┼──────────────────────────────────┼────────────────────────────────────────────────┤
  │ M6        │ Storage, manifest, builder, CLI  │ Interrupt-and-resume test; failure-log         │
  │           │                                  │ contents; e2e on a ~20-compound catalog        │
  ├───────────┼──────────────────────────────────┼────────────────────────────────────────────────┤
  │ M7        │ featurize/ (PyG)                 │ Batch collation, HeteroData validation,        │
  │           │                                  │ forward pass through SchNet/ViSNet smoke test  │
  └───────────┴──────────────────────────────────┴────────────────────────────────────────────────┘

  D. Questions before Phase 2

  Each question has my default in brackets, so you can answer "defaults" for any of them.

  Scope and integration
  1. Downstream task: QSAR/property prediction, ligand-based virtual screening, de novo generation,
     docking or binding affinity, or feeding dev_Pgen's drug tower? Docking, affinity and excluded
     volumes need receptor structures. [QSAR + dev_Pgen drug tower; ligand-only]
  2. Is Modephere standalone, or will dev_Pgen import it? If dev_Pgen imports it, dev_Pgen must move
     to PyG ≥2.8, and we must keep its 2D schemas (61/18/1038 and 11/13/258) identical. [Standalone,
     with schema-compatible 2D outputs]
  3. Ambiguous names (one name, several CIDs): take the first PubChem hit, fail, or prefer a parent
     compound that isn't a mixture? Should synonyms or ChEMBL be used as a fallback? [First hit +
     logged warning; ChEMBL fallback off]
  4. Is there an existing catalog to start from (e.g. dev_Pgen's data/drugs_cid.tsv)? How many
     compounds — hundreds or millions? This decides whether batching and storage sharding matter.

  Chemistry
  5. Unspecified stereocentres: enumerate stereoisomers (capped, e.g. ≤8) or embed with random stereo?
  [Enumerate, cap 8]
  6. Conformer sampling depth: one lowest-energy conformer, or an ensemble? If an ensemble, what cap
  and energy window? [Ensemble: 10–50 by rotatable bonds, 10 kcal/mol, RMSD 0.5 Å]
  7. Protonation: keep dev_Pgen's rule-based pH 7.4 microspecies, or offer dimorphite_dl too? A single
  state or several? [Rule-based port, single state]
  8. Salt stripping: "largest fragment" like dev_Pgen, or the ChEMBL salt-list parent? [Largest
  fragment, ChEMBL optional]
  9. Metals, inorganics and peptides larger than about 100 heavy atoms: reject with a failure
  category, or try anyway? [Reject; reported]

  ML
  10. First model target: 2D GNN only, 3D invariant (SchNet/ViSNet), or equivariant models? [2D +
  ViSNet; equivariant later]
  11. HeteroData (atom + pharmacophore nodes), separate graphs like dev_Pgen's two files, or both?
  [Both: shared perception, two adapters]
  12. ADMET-AI profile: port admet.py as well? [Not in v1]

  Infrastructure
  13. Hardware: I see an RTX 4070 Ti SUPER (16 GB), 8 vCPU and about 15 GB RAM under WSL2. Is that the
  target, or is there a cluster? Can embedding use all 8 cores? [This box; all cores]
  14. Storage format: Parquet registry + safetensors shards (pickle-free; my pick), per-compound .pt
  like dev_Pgen, LMDB (random access at scale), or SDF too (for exchange with other tools)? [Parquet +
  safetensors, optional SDF export]
  15. Network policy: can the build reach PubChem, or must resolution be a separate fetch step that
  runs before a fully offline build, like src.kg? [Separate fetch step, then offline]
  16. Python package and CLI name: modephere with python -m modephere.library (the scaffold currently
  points at a non-existent modephere:main)? Same license (GPLv3) and author as dev_Pgen?

  ─────────────────────────────────────────────────────────────────────────────────────────────────

  E. Decisions log

  2026-11-04 — types package (first implementation step)
  - Section D questions are unanswered; the bracketed defaults are in force until changed.
  - Tabular data is polars only, never pandas (user rule). Resolved to polars 2.1.0.
  - The C1 `domain/` layer is replaced by `src/modephere/types/` (named `types`, not `stubs`,
    which in Python means .pyi files). It is the leaf of the import graph and never imports RDKit.
      base.py          ModephereModel / FrozenModel / ArrayModel (element-wise eq for arrays)
      scalars.py       PubChemCID, InChIKey, ChemblID, SmilesStr, AtomIds, Angstrom, Vec3, UnitVec3…
      arrays.py        numpy field types (same_kind casts, range-checked ints, read-only copies)
      enums.py         IdentifierKind, PharmacophoreFamily (dev_Pgen order), ForceField,
                       FailureCategory, BuildStage, …
      identifiers.py   SmilesQuery | CidQuery | NameQuery (discriminated on `kind`), parse_query
      compound.py      ResolvedCompound (provenance), StandardizedCompound (registration vs working)
      conformer.py     ConformerParams (Q5/Q6 defaults), ConformerRecord, ConformerEnsemble [C,N,3]
      pharmacophore.py PharmacophoreFeature/Point, Pharmacophore2D/3D (canonical feature order)
      layouts.py       GraphLayout; MOLECULE_GRAPH_V1 (61/18/1038), PHARMACOPHORE_GRAPH_V1 (11/13/258)
      failures.py      FailureRecord
      frames.py        polars schemas (compounds / conformers / failures) + check_frame
      tensors.py       torch/PyG contracts (lazy, `ml` extra): Molecule/Pharmacophore/Hetero/
                       ConformerTensors, PharmacophoreData (batch-safe atom2feat)
  - PyG batching only offsets attributes named *index*: a plain `atom2feat` (dev_Pgen) is not
    offset when batched. PharmacophoreData.__inc__ fixes it here; HeteroData uses member_of edges.
  - Installed: pydantic 2.14.2, numpy 2.5.4, polars 2.1.0, torch 2.12.1+cu132, PyG 2.8.1,
    pyg_lib 0.7.0 (radius_graph works without torch-cluster).

  2026-11-04 — M0–M7 built (rest of the library)
  - Layout as C1, with these deviations: `core/` has exceptions + log (build_file_logger lives there,
    no separate reporting.py); `library/pipeline.py` (per-compound stages + build_compound) and
    `library/export.py` (SDF) added; no `chembl.py` (Q3 default: ChEMBL fallback off).
  - RDKit pinned ==2025.9.6 (same as dev_Pgen): 2D outputs are byte-identical to dev_Pgen on a
    25-molecule panel (tests/unit/library/test_parity.py; regenerate with
    scripts/make_parity_fixtures.py). Bump RDKit only together with dev_Pgen.
  - Settings priority is init > MODEPHERE_* env > TOML (dev_Pgen passes TOML as init args, so its
    env vars cannot override TOML keys).
  - PubChem client never raises inside a batch: lookups are ok / not_found / missing (offline,
    never fetched) / error (retries exhausted, not cached). CacheMissError was dropped.
  - Standardization: metals other than alkali/alkaline-earth counterions are rejected in any
    fragment (ferrocene/cisplatin written as ions would otherwise lose the metal silently);
    the working form drops isotopic H (InChIKey keeps them).
  - Stereoisomers are enumerated on the neutral form: a protonated amine (N+-H) is not a
    stereocentre to enumerate (it inverts through deprotonation).
  - RMSD pruning is greedy in energy order (equivalent role to Butina, deterministic).
  - ProtonationMethod.DIMORPHITE removed: dimorphite re-orders atoms and would break atom2feat.
  - 3D: geometric models read z + pos and build their own radius graphs, so no RBF distance edges
    are precomputed; atom pos and pharm pos always come from the same conformer.
  - Storage: one safetensors file per compound (+ Parquet registry/conformers/failures); reads
    re-validate CompoundArtifacts. Sharding for millions of compounds still open (Q4).
  - Bug found while porting, also present in dev_Pgen src/kg/raw_cache.py: after a crash leaves a
    truncated final cache line, the next put() appends after it, and the following load fails
    ("corrupt cache line"). Fixed here (the partial line is dropped before appending).

  2026-11-05 — ChEMBL layer + Neo4j knowledge graph (section F)
  - User decision: ChEMBL is the knowledge source; scope = drug / clinical set + aggregated
    activities; raw activities stay in Parquet only; aggregated edges in the graph.
  - Defaults taken (user may revise): code lives in Modephere (`modephere.chembl`,
    `modephere.kg`); Modephere's own Neo4j container (ports 7475/7688, docker-compose.kg.yml),
    with dev_Pgen's keys and provenance (Gene{symbol}, Dataset{id}, IngestRun{id}, sources,
    run_id) so the layers can share one database later. min_confidence = 7.
  - At this size (~46k nodes, ~150k relationships) the driver MERGE load takes ~25 s:
    neo4j-admin bulk import is not needed.
  - ChEMBL curates many annotations on salt records (21% of mechanisms, 12% of indications,
    32% of warnings): the graph attaches them to the parent and keeps the record as form/forms.
  - `with sqlite3.connect()` never closes the connection; ChemblRelease.connect() closes.
  - Found: Modephere's identity InChIKey (canonical tautomer, RDKit defaults) drops sp3 stereo
    at tautomer-affected centres (alpha-amino acids, alpha-carbonyl carbons): enantiomers can
    collide. 76% of approved ChEMBL parents match exactly; a stereo-preserving canonical
    tautomer (tautomerRemoveSp3Stereo/BondStereo = False) gives 88%.
  - 2026-11-05, user approved: the registration form keeps its stereo
    (standardize._canonical_tautomer). L-/D-alanine now get distinct keys (tests). No library
    had been built, so no stored keys changed. The remaining 12% differ in connectivity: the
    key is that of the canonical tautomer (keto/enol moves are outside standard InChI's
    mobile-H), plus salt-only "parents" (calcium carbonate). Joining labels on chembl_id
    (F5) removes the dependence on InChIKey agreement.
  2026-11-05 — ChEMBL resolver + graph links (F5)
  - Registry schema gained four chembl_* columns (and IdentifierKind "chembl_id"): registries
    written before this are rejected by check_frame. No library existed yet.
  - Drug links never guess: an ambiguous ChEMBL name (several parents) is left unlinked.
    dev_Pgen's drugs_cid.tsv: 4,280 names, 75% linked (2,747 by CID InChIKey, 472 by name);
    the rest are mostly non-drugs (pesticides, IUPAC fragments) or racemates ("dl-dopa")
    whose structure is not the approved drug's.
  - Live link (dev_Pgen graph, 1,161 Drug nodes): 870 linked (76%). dev_Pgen's name->CID map
    has wrong CIDs (formoterol -> salmeterol's, omalizumab -> formoterol's, apatinib ->
    lapatinib's) and gives drug classes / combinations one drug's CID. Rules: when the name
    resolves uniquely in ChEMBL to another parent than the CID, the curated name wins (15
    cases; the edge evidence records the overridden CID); ATC class names and "a / b"
    combinations are never linked to one compound. Left: CIDs of names ChEMBL does not
    know cannot be checked (daunorubicinol -> daunorubicin).
  - dev_Pgen's pharmagen-neo4j container exited right after every start (its own container
    state; the data volume was fine). Recreated from dev_Pgen's docker-compose.kg.yml, data
    volume kept. The system Neo4j service that held 7474/7687 was stopped by the user.

  2026-11-05 — sp2 selenium under UFF (ebselen)
  - Ebselen and selenophenes failed at the conformer stage ("no parameters in
    MMFF94s, UFF"). MMFF94 has no Se; UFF has one Se type (Se3+2), but RDKit's
    UFF typer derives the type from hybridization and asks for "Se2+2" for Se in
    a conjugated ring. Fix: UFF parametrizes / optimises a copy with Se and Te
    retyped sp3 (conformers._uff_view); coordinates are copied back, the stored
    molecule keeps its sp2 perception. Ebselen: C-Se-N 86 deg, planar ring; Se
    bonds ~0.07 A short (UFF aromatic bond order). ETKDG still logs
    "UFFTYPER: Unrecognized atom type: Se2+2" (its bounds use UFF); harmless.

  2026-11-05 — inspection scripts + IUPAC atom numbering
  - scripts/inspect_compound.py (ChEMBL / PubChem / PharmGKB view of one compound),
    inspect_graph.py (Neo4j schema, one node's neighbourhood), reconstruct_molecule.py
    (graph -> molecule round trip, pictures, 3D viewer). Not tests: run by hand.
  - Atom indices everywhere are positions in the working SMILES (dev_Pgen parity);
    they are not IUPAC locants. The scripts add IUPAC locants by name: PubChem's
    IUPAC name (by CID, else InChIKey, else name) -> OPSIN 2.9.0 (CML locant
    labels) -> skeleton match onto our atoms. Parts of the name are recovered by
    joining bonded atoms while locants stay unique. Needs Java + the OPSIN jar
    (project root, gitignored); no IUPAC numbering without a PubChem name.
  - PubChemClient.inchikey_to_cids (same cache file as names, key "inchikey:<key>").

  F. ChEMBL knowledge graph

  F1. Data flow
      chembl_<N>.db (read-only, SHA-256 in the manifest)
        -> python -m modephere.chembl extract   -> data/chembl/chembl_<N>/*.parquet
        -> python -m modephere.kg load-chembl   -> Neo4j (MERGE, idempotent, provenance)
      Labels for LibraryDataset come from bioactivity.parquet (modephere.chembl.labels); the
      offline name / ID lookup from compounds + synonyms (modephere.chembl.lookup).

  F2. Selection: molecules with a max_phase, plus any molecule named by a mechanism, indication,
      warning, ATC code or metabolism record; their parents; every form of those parents.
      ChEMBL 37: 22,631 compounds, 2,365,157 raw activities, 59,205 (parent, target) pairs.

  F3. Bioactivity summary per (parent, target): pChEMBL present, relation '=', target assigned,
      assay confidence >= min_confidence, not a potential duplicate, no data-validity issue
      ('Manually validated' allowed). n, median/mean/std/min/max, n_assays, n_documents,
      max_confidence, assay/standard types, first/last year. `aggregate` re-runs it at another
      threshold without reading SQLite.

  F4. Graph model: see src/modephere/kg/schema.py. Gene nodes for human proteins only (HGNC
      symbols, dev_Pgen's key). HAS_BIOACTIVITY is a measurement, not an effect.

  F5. Resolver and links (done 2026-11-05):
      - ChemblQuery (kind "chembl_id"); names: ChEMBL release first, PubChem for the rest;
        every resolution gets its ChEMBL parent by ID / name / PubChem InChIKey / structure
        InChIKey (ResolvedCompound.chembl_*; registry columns chembl_id, chembl_record_id,
        chembl_match, chembl_release). bioactivity_labels(library=...) joins on chembl_id.
      - (:LibraryCompound {inchikey})-[:REPRESENTS]->(:Compound): kg link-library.
      - (:Drug {name, cid})-[:SAME_AS {method, evidence}]->(:Compound): kg link-drugs, from
        dev_Pgen's graph (--drugs-uri), a file, or Drug nodes already in the graph.
      Open: the KG side of targets beyond the drug set.
