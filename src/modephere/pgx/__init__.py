"""Pharmacogenomic layer of the knowledge graph (ported from dev_Pgen's ``src.kg``).

PharmGKB variant annotations, their allele labels (rsIDs, star alleles) and
the canonical GRCh38 variants behind them, merged into the same Neo4j as the
ChEMBL layer. The two meet at ``Gene`` (HGNC symbol) and ``Drug`` (``SAME_AS``
a ChEMBL ``Compound``). Model: :mod:`modephere.kg.schema`.

Workflow::

    python -m modephere.pgx fetch            # the only networked step -> raw cache
    python -m modephere.pgx ingest --replace # offline build -> Neo4j, then audit
    python -m modephere.kg link-drugs        # Drug -[:SAME_AS]-> Compound
    python -m modephere.pgx export           # -> data/pgx/export/*.parquet

Variants are identified by their GRCh38 genomic HGVS (``hgvs_g38``, from the
NCBI-canonical SPDI); rsIDs and star alleles are metadata on them. Star alleles
are their *cis* variant sets (PharmVar, ClinPGx allele tables); rsIDs are sites
with all their dbSNP alleles; the genotypes an annotation compares (``*1/*22``,
``CT``) are ``Genotype`` nodes with the copies of each allele
(:mod:`modephere.pgx.genotype`). Every node and relationship carries
``sources`` and ``run_id``.
Needs the ``pgx`` extra (``uv sync --extra pgx``).
"""
