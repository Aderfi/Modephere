# Modephere

Molecule and 2D/3D pharmacophore representations from **SMILES, PubChem CIDs or
drug names**, stored as a resumable library and ready for PyTorch Geometric.

For every compound it builds:

| Output | Shape / layout |
|---|---|
| Molecular graph (working form) | `molecule/v1`: 61 node, 18 edge, 1038 global features (identical to dev_Pgen) |
| Pharmacophore graph (pH 7.4) | `pharmacophore/v1`: 11 node, 13 edge, 258 global features (identical to dev_Pgen) |
| Conformer ensemble | ETKDGv3 + MMFF94s (UFF fallback), stereoisomers enumerated, energy window, RMSD pruning |
| 3D pharmacophore per conformer | feature centroids + H-bond / ring-normal directions |
| Fingerprints | Gobbi 2D pharmacophore (39 972 bits), 3D feature triplets (4 096 bits) |

Design decisions and their reasons are in [`PLAN.md`](PLAN.md).

## Install

```bash
uv sync --extra dev            # chemistry + storage
uv sync --extra ml --extra dev # + torch / PyG featurization (CUDA 13.2 wheels)
uv sync --extra gpu --extra dev # + nvMolKit GPU conformers (else RDKit on CPU)
uv sync --all-extras           # everything, incl. ChEMBL salt-list standardizer
```

Extras: `ml` (torch + PyG), `gpu` (torch + nvMolKit), `kg` (neo4j), `pgx`
(neo4j + openpyxl), `std` (ChEMBL structure pipeline), `viz` (matplotlib),
`all` (every one of those), `dev` (tests, linters, graphify).

### As a wheel

Each GitHub release carries the wheel. From another uv project:

```bash
uv add "modephere[all] @ https://github.com/Aderfi/Modephere/releases/download/v0.1.1/modephere-0.1.1-py3-none-any.whl"
```

`[tool.uv.sources]` is not part of the wheel, and uv applies a project's
sources only to its direct dependencies. So the project that adds
`modephere[all]` (or `[ml]`) lists `torch` and `pyg_lib` itself and routes
them: `pyg_lib` is only on the PyG index, and torch should come from the
CUDA 13.2 index (from PyPI otherwise). In its `pyproject.toml`:

```toml
[project]
dependencies = [
    "torch>=2.12,<2.13",
    "pyg_lib",
    "modephere[all]",
]

[[tool.uv.index]]
name = "pytorch-cu132"
url = "https://download.pytorch.org/whl/cu132"
explicit = true

[[tool.uv.index]]
name = "pyg"
url = "https://data.pyg.org/whl/torch-2.12.0+cu132.html"
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu132" }
pyg_lib = { index = "pyg" }
modephere = { url = "https://github.com/Aderfi/Modephere/releases/download/v0.1.1/modephere-0.1.1-py3-none-any.whl" }
```

With pip: `--extra-index-url https://download.pytorch.org/whl/cu132
--find-links https://data.pyg.org/whl/torch-2.12.0+cu132.html`.
To build the wheel locally: `uv build --wheel` (→ `dist/`).

Installed, Modephere resolves `data/`, `logs/` and the other entries of
`paths.toml` against the working directory (override with
`MODEPHERE_PATHS__<NAME>`), and ingest runs record `code_version` as
`modephere==<version>` rather than a git commit.

## Command line

```bash
# 1. the only networked step: resolve names / CIDs on PubChem into the raw cache
modephere fetch --catalog drugs.tsv
# 2. build offline from the cache (resumable; --force rebuilds)
modephere build --catalog drugs.tsv
# without a catalog, straight from PubChem, 2D only
modephere build --name ibuprofen --cid 2244 --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --online --2d-only
# conformers of one compound as SDF
modephere export-sdf BSYNRYMUTXBXSQ-UHFFFAOYSA-N aspirin.sdf
```

Catalogs: `.tsv` / `.csv` / `.txt` / `.parquet` with `kind,value[,label]` columns,
or any of `smiles` / `cid` / `name` columns (dev_Pgen's `cid, smiles,
cmpd_name_cleaned` works as is); `.json` as a list of queries or `{cid: smiles}`.
Exit status is 2 when a compound failed; see `failures.parquet` and
`logs/library/build_failures_<date>.log`.

Paths and PubChem etiquette come from `src/modephere/config/data/*.toml`,
overridable with `MODEPHERE_*` variables (e.g. `MODEPHERE_PATHS__LIBRARY=/scratch/lib`).

## Python

```python
from modephere.library import build_compound, LibraryStore

art = build_compound("CC(=O)Oc1ccccc1C(=O)O")  # one compound, nothing on disk
art.molecule.x.shape  # (13, 61)
[f.family for f in art.pharmacophore_2d.features]
art.conformers.num_conformers, art.pharmacophores_3d[0].points[0].direction

from modephere.featurize import LibraryDataset  # needs the ml extra
from torch_geometric.loader import DataLoader

ds = LibraryDataset("data/library", kind="hetero", conformer="sample", targets=labels)
for batch in DataLoader(ds, batch_size=32, shuffle=True):
    ...  # batch["atom"].z / .pos for SchNet / ViSNet, batch["pharm"], member_of edges
```

`targets` is a polars frame with an `inchikey` column; the other columns become `y`.

## ChEMBL and the knowledge graph

ChEMBL (CC BY-SA 3.0) supplies drug annotations and bioactivity labels from a local
SQLite release; Neo4j holds them as a graph. Raw activities stay in Parquet.

```bash
uv sync --all-extras                                  # includes the neo4j driver (kg extra)
uv run python -m modephere.chembl extract             # chembl_<N>.db -> data/chembl/chembl_<N>/
docker compose -f docker-compose.kg.yml up -d         # Neo4j on bolt://localhost:7688
uv run python -m modephere.kg load-chembl             # idempotent MERGE, ~25 s for the drug set
uv run python -m modephere.kg audit                   # provenance check
uv run python -m modephere.kg search gleevec
uv run python -m modephere.kg link-library            # built compounds -[:REPRESENTS]-> ChEMBL
uv run python -m modephere.kg link-drugs --drugs-uri bolt://localhost:7687  # dev_Pgen drugs
```

With a staged release the library builder resolves names through ChEMBL first (offline;
a salt name maps to its parent; PubChem only for names ChEMBL lacks), accepts ChEMBL IDs
(`--chembl-id`, or a `chembl_id` catalog column) and records each compound's ChEMBL parent
in the registry (`chembl_id`, `chembl_release`). `--no-chembl` turns this off.

```python
from pathlib import Path

from modephere.chembl import ChemblLookup, bioactivity_labels

staging = Path("data/chembl/chembl_37")
ChemblLookup(staging).by_name("gleevec")[0].parent_chembl_id  # 'CHEMBL941' (imatinib)
# one pChEMBL column per target, keyed by the library's InChIKeys via chembl_id
labels = bioactivity_labels(staging, ["CHEMBL1862"], library=Path("data/library"))
```

The SQLite path is `[chembl] sqlite` in `settings.toml` (`MODEPHERE_CHEMBL__SQLITE`). Graph
model: `src/modephere/kg/schema.py`; selection and aggregation rules: `PLAN.md` section F.

## Pharmacogenomic layer

`modephere.pgx` (ported from dev_Pgen's `src.kg`) merges PharmGKB variant annotations,
their allele labels (rsIDs, star alleles) and the canonical GRCh38 variants behind them
(PharmVar / ClinPGx *cis* sets; every dbSNP alternative of an rsID) into the same Neo4j.
The layers meet at `Gene` (HGNC symbol) and `Drug` (`SAME_AS` a ChEMBL `Compound`);
model in `src/modephere/kg/schema.py`.

- A variant is its GRCh38 genomic HGVS, `Variant {hgvs_g38: "NC_000010.11:g.94781859G>A"}`
  (from the NCBI-canonical SPDI, kept as `spdi_g38`); rsIDs and star alleles are
  metadata on it (`rsids`, `star_alleles`). Assembly-specific properties carry the
  assembly suffix, so `hgvs_g37` / `hgvs_t2t` can be added beside them later.
- A star allele is a `Haplotype` with `HAS_VARIANT` to each mutation it groups and
  its cis set as `hgvs_g38` (`NC_000010.11:g.[94775367A>G;94781859G>A;94842866A>G]`).
- Outcomes that depend on two alleles keep them: `(:Annotation)-[:TESTED|COMPARED_TO]->
  (:Genotype {name: "CYP2C19*1/*2"})-[:HAS_COPY {copies}]->(:Haplotype)`, or
  `->(:Variant)` for a site genotype (`rs4244285:A/G`, `ref_copies: 1`). Genotype HGVS
  is diploid (`g.[…];[=]`); `copies` is exact or null (carrier-only report, `xN`),
  `min_copies` the lower bound, and `gene_copies` / `gene_copies_min` count gene copies.
- Structural alleles: whole-gene deletions (`CYP2D6*5`, `CYP2A6*4`, `GSTM1*0` behind
  `GSTM1 null`) are `Haplotype {structure: "gene_deletion"}` with HGVS over the NCBI Gene
  span (`g.(?_start)_(end_?)del`, plus `hgvs_t2t` where T2T-CHM13 places the gene, e.g.
  GSTT1, absent from GRCh38's primary assembly); `*2xN` counts copies of `*2`; HLA alleles
  are `Haplotype {structure: "hla_typed"}` named by IPD-IMGT/HLA (no HGVS). CYP2D6-2D7 /
  CYP2A6-2A7 hybrids stay unresolved. Links read on PharmGKB's forward-strand convention
  where the strand cannot be proven carry `strand_assumed: true`.

```bash
uv sync --extra pgx                              # neo4j + openpyxl
uv run python -m modephere.pgx fetch             # only networked step: dbSNP, NCBI SPDI, ClinPGx
uv run python -m modephere.pgx ingest            # offline build -> Neo4j, provenance audit
uv run python -m modephere.pgx ingest --replace  # rebuild the layer; ChEMBL data untouched
uv run python -m modephere.kg link-drugs         # Drug -[:SAME_AS]-> Compound
uv run python -m modephere.kg link-library       # LibraryCompound -[:REPRESENTS]-> Compound
uv run python -m modephere.pgx export            # -> data/pgx/export/*.parquet
```

Inputs live under `data/pgx/` (`pharmgkb/var_*_ann.tsv`, `pharmvar/<GENE>/`,
`drugs/`), raw responses under `data/cache/{ncbi,clinpgx}/` (same format as dev_Pgen's
`data/library/kg/raw/`); see `modephere.pgx.inputs`. Every pgx run is an
`:IngestRun {layer: "pgx"}`, so `--replace` withdraws only its datasets. Each ingest rewrites
`data/pgx/reports/unresolved.log`: one line per unresolved case
(`<category> | <reason> | <subject> | annotation=<id> | <raw notation>`), next to
`resolution_report.tsv` and `disagreements.tsv`. `export` writes
`annotation_rows.parquet` (each drug as `{name, cid, chembl_id, inchikey}`, the InChIKey
naming its `compounds/<InChIKey>.safetensors`), `allele_variants.parquet`,
`annotation_variants.parquet`, `annotation_genotypes.parquet` (each tested / compared
genotype with its per-variant dosage, star alleles expanded into their mutations) and
`variants.parquet` (each variant's genes, rsIDs, star alleles, dbSNP SO consequence terms
and its own protein change, `NP_…:p.`) and `haplotypes.parquet` (every PharmVar / ClinPGx
haplotype with its cis variants, HGVS, structure and core allele, named by an annotation or
not).

## Library layout

```
data/library/
├── compounds/<InChIKey>.safetensors   every array of one compound (no pickle)
├── compounds.parquet                  registry (query, source, CID, SMILES forms)
├── conformers.parquet                 energies, stereoisomer, force field
├── failures.parquet                   failures of the latest run
└── build_manifest.json                options + toolkit versions
data/cache/pubchem/*.jsonl             raw PubChem responses (offline replay)
data/chembl/chembl_<N>/*.parquet       ChEMBL staging: compounds, synonyms, mechanisms,
                                       indications, ..., activities (raw), bioactivity
```

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src tests
uv run pytest
```

`tests/unit/library/test_parity.py` checks byte-identical 2D graphs against
dev_Pgen golden data (`scripts/make_parity_fixtures.py` regenerates it with
dev_Pgen's interpreter). RDKit is pinned to dev_Pgen's version for that reason.
