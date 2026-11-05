"""CLI for the library builder.

Usage::

    # 1. the only networked step: resolve names / CIDs into the raw cache
    python -m modephere.library fetch --catalog data/drugs.tsv
    # 2. build offline from the cache (resumable; --force rebuilds)
    python -m modephere.library build --catalog data/drugs.tsv
    # one-off structures, no catalog, 2D only
    python -m modephere.library build --smiles "CC(=O)Oc1ccccc1C(=O)O" --name ibuprofen --online --2d-only
    # conformers of one compound as SDF
    python -m modephere.library export-sdf BSYNRYMUTXBXSQ-UHFFFAOYSA-N aspirin.sdf

Defaults come from the project settings (``MODEPHERE_*`` env vars, TOML), so
most invocations only name their inputs. Exit status: 0 on success, 2 when
any compound failed (details in ``failures.parquet`` and the failure log).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from modephere.config import get_settings
from modephere.core import ModephereException, setup_logging
from modephere.library.builder import LibraryBuilder
from modephere.library.config import LibraryBuildConfig
from modephere.library.pipeline import PipelineOptions
from modephere.library.standardize import StandardizeOptions
from modephere.library.storage import LibraryStore
from modephere.types import AmbiguityPolicy, ConformerParams, FragmentPolicy

logger = logging.getLogger(__name__)


def _add_inputs(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("inputs (any combination)")
    g.add_argument("--catalog", type=Path, help="tsv/csv/txt/parquet/json query list.")
    g.add_argument("--smiles", action="append", default=[], metavar="SMILES")
    g.add_argument("--cid", action="append", default=[], metavar="CID")
    g.add_argument("--name", action="append", default=[], metavar="NAME")
    g.add_argument("--chembl-id", action="append", default=[], metavar="CHEMBL_ID")
    p.add_argument(
        "--no-chembl",
        action="store_true",
        help="Ignore the staged ChEMBL release (names via PubChem only).",
    )
    p.add_argument(
        "--ambiguity",
        choices=[a.value for a in AmbiguityPolicy],
        default=AmbiguityPolicy.FIRST_HIT.value,
        help="A name matching several compounds: take the best hit, or fail.",
    )
    p.add_argument("--library", type=Path, help="Library root (default: settings).")
    p.add_argument("--cache", type=Path, help="Raw PubChem cache directory.")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logging.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modephere.library",
        description="Build molecule + 2D/3D pharmacophore libraries.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Resolve queries on PubChem into the cache.")
    _add_inputs(fetch)

    build = sub.add_parser("build", help="Build (offline by default) and store.")
    _add_inputs(build)
    build.add_argument("--online", action="store_true", help="Query PubChem directly.")
    build.add_argument("--force", action="store_true", help="Rebuild stored compounds.")
    build.add_argument("--2d-only", dest="two_d_only", action="store_true")
    build.add_argument(
        "--num-conformers", type=int, help="Fixed embeddings per isomer."
    )
    build.add_argument("--max-conformers", type=int, default=50)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--threads", type=int, default=0, help="0 = all cores.")
    build.add_argument(
        "--fragment-policy",
        choices=[f.value for f in FragmentPolicy],
        default=FragmentPolicy.LARGEST_FRAGMENT.value,
    )
    build.add_argument("--max-heavy-atoms", type=int, default=100)

    export = sub.add_parser("export-sdf", help="Write a compound's conformers to SDF.")
    export.add_argument("inchikey")
    export.add_argument("out", type=Path)
    export.add_argument(
        "--library", type=Path, help="Library root (default: settings)."
    )
    export.add_argument("--verbose", "-v", action="store_true")
    return parser


def _config(args: argparse.Namespace, *, build: bool) -> LibraryBuildConfig:
    overrides: dict[str, object] = {
        "catalog": args.catalog,
        "smiles": tuple(args.smiles),
        "cids": tuple(args.cid),
        "names": tuple(args.name),
        "chembl_ids": tuple(args.chembl_id),
        "ambiguity": args.ambiguity,
        "library_root": args.library,
        "cache_dir": args.cache,
    }
    if build:
        conformers = ConformerParams(
            num_conformers=args.num_conformers,
            max_conformers=args.max_conformers,
            min_conformers=min(10, args.max_conformers),
            random_seed=args.seed,
            num_threads=args.threads,
        )
        overrides |= {
            "offline": not args.online,
            "force": args.force,
            "pipeline": PipelineOptions(
                standardize=StandardizeOptions(
                    fragment_policy=args.fragment_policy,
                    max_heavy_atoms=args.max_heavy_atoms,
                ),
                conformers=conformers,
                build_3d=not args.two_d_only,
            ),
        }
    return LibraryBuildConfig.from_settings(use_chembl=not args.no_chembl, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(name="library", level=logging.DEBUG if args.verbose else logging.INFO)
    try:
        if args.command == "export-sdf":
            root = args.library or get_settings().paths.library
            n = LibraryStore(root).export_sdf(args.inchikey, args.out)
            logger.info("Wrote %d conformers of %s to %s", n, args.inchikey, args.out)
            return 0
        if args.command == "fetch":
            fetched = LibraryBuilder(_config(args, build=False)).fetch()
            return 0 if fetched.failed == 0 else 2
        summary = LibraryBuilder(_config(args, build=True)).run()
        return 0 if summary.ok else 2
    except (ModephereException, ValueError) as e:
        logger.error("%s", e)  # noqa: TRY400 -- user-facing message, no traceback
        return 1


if __name__ == "__main__":
    sys.exit(main())
