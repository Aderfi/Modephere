"""Provenance primitives (W3C PROV-O), shared with dev_Pgen's ``src.kg``.

* :class:`Dataset` — a ``prov:Entity``: one pinned input (here a ChEMBL
  release), with checksum and licence;
* :class:`IngestRun` — a ``prov:Activity``: one load into the graph;
* :class:`Stamp` — what every node and relationship carries: the datasets it
  derives from and the run that wrote it.

Same property names as dev_Pgen (``sources``, ``run_id``, ``:Dataset {id}``,
``:IngestRun {id}``), so both projects can write to one graph and its audit
query holds for either.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class Dataset(BaseModel):
    """A pinned input to the knowledge graph (``prov:Entity``)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1)
    name: str
    version: str
    uri: str
    retrieved_at: datetime
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    license: str | None = None

    def props(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "uri": self.uri,
            "retrieved_at": self.retrieved_at.isoformat(),
            "sha256": self.sha256,
            "license": self.license,
        }


class IngestRun(BaseModel):
    """One load into the graph (``prov:Activity``)."""

    id: str
    started_at: datetime
    finished_at: datetime | None = None
    code_version: str
    dirty: bool
    tool: str = Field(default_factory=lambda: f"modephere {version('modephere')}")
    params: dict[str, str] = Field(default_factory=dict)
    layer: str | None = Field(
        default=None,
        description="Graph layer the run writes (e.g. 'pgx'), so a layer can be "
        "rebuilt without touching the others.",
    )

    @classmethod
    def start(
        cls,
        params: dict[str, str],
        repo: Path | None = None,
        *,
        layer: str | None = None,
    ) -> IngestRun:
        sha, dirty = code_version(repo) if repo is not None else ("unknown", True)
        return cls(
            id=uuid.uuid4().hex,
            started_at=datetime.now(UTC),
            code_version=sha,
            dirty=dirty,
            params=dict(params),
            layer=layer,
        )

    def props(self) -> dict[str, object]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": None
            if self.finished_at is None
            else self.finished_at.isoformat(),
            "code_version": self.code_version,
            "dirty": self.dirty,
            "tool": self.tool,
            "layer": self.layer,
            # Neo4j properties cannot be maps: the parameters travel as JSON.
            "params": json.dumps(self.params, sort_keys=True),
        }


class Stamp(BaseModel):
    """Provenance carried by every node and relationship."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[str, ...] = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    method: str | None = None

    def props(self) -> dict[str, object]:
        """Run id (and method) as a property map; ``sources`` is merged apart."""
        out: dict[str, object] = {"run_id": self.run_id}
        if self.method is not None:
            out["method"] = self.method
        return out


def _release() -> tuple[str, bool]:
    """``("modephere==X.Y.Z", False)``: an installed build is a fixed version."""
    try:
        return f"modephere=={version('modephere')}", False
    except PackageNotFoundError:
        return "unknown", True


def _is_checkout(repo: Path) -> bool:
    """The running modephere code is ``repo``'s own source tree."""
    import modephere  # noqa: PLC0415 -- the package is fully loaded by now

    package = Path(modephere.__file__).resolve().parent
    return package.is_relative_to(repo.resolve())


def code_version(repo: Path) -> tuple[str, bool]:
    """``(commit sha, dirty)`` of ``repo`` when modephere runs from that checkout.

    An installed wheel is identified by its release (``modephere==X.Y.Z``,
    clean) instead: the working directory, or any git repository around it,
    says nothing about the code that ran. ``("unknown", True)`` when git fails
    on a checkout or no version is known.
    """
    if not _is_checkout(repo):
        return _release()
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Cannot read git version of %s: %s", repo, e)
        return "unknown", True
    return sha, bool(status.strip())


__all__ = ["Dataset", "IngestRun", "Stamp", "code_version"]
