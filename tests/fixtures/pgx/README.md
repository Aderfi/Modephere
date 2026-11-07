# KG fixtures

Real responses recorded on 2026-10-18 with `scripts/record_kg_fixtures.py`.

| Path | Source | License |
|---|---|---|
| `esummary_snp.json` | NCBI E-utilities ESummary, `db=snp`, ids `3892097,17879456,4244285,1065852,999999999999` (dbSNP build 157) | Public domain (NCBI) |
| `spdi_canonical/`, `spdi_hgvs/`, `hgvs_contextuals/` | NCBI Variation Services v0 | Public domain (NCBI) |
| `allele_tables/{TPMT,UGT1A1}.xlsx` | ClinPGx (PharmGKB) allele-definition tables, `https://api.clinpgx.org/v1/download/file/attachment/<GENE>_allele_definition_table.xlsx` | CC BY-SA 4.0 |

Notes:
- `rs17879456` is a merged ID: its record reports `snp_id` 4244285.
- `999999999999` does not exist: ESummary returns it in `uids` with
  `{"error": "cannot get document summary"}`, not by omission.
- `NC_000006.12:g.18149032_18149033insT` (ClinPGx TPMT) carries NCBI's
  warning that the HGVS is not at its most 3' position.
