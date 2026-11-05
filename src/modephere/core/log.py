"""Logging configuration.

Call :func:`setup_logging` once at application start-up (the CLI does).
Library modules only do ``logging.getLogger(__name__)`` and inherit it.
Dedicated per-report audit files use :func:`build_file_logger`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import TYPE_CHECKING

from modephere.config import get_settings

if TYPE_CHECKING:
    from pathlib import Path

_NOISY = ("httpx", "httpcore")


def setup_logging(
    name: str = "modephere",
    level: int | None = None,
    console_level: int | None = None,
) -> logging.Logger:
    """Configure the root logger: a daily-rotating file plus the console.

    Idempotent: a second call returns the already-configured root logger.
    """
    root = logging.getLogger()
    if root.handlers:
        return root

    logs = get_settings().paths.logs
    logs.mkdir(parents=True, exist_ok=True)
    log_file = logs / f"{name}_{datetime.now():%Y-%m-%d}.log"

    root.setLevel(logging.DEBUG)
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=7, encoding="utf-8"
    )
    file_handler.setLevel(level or logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level or level or logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    for noisy in _NOISY:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # RDKit's own parse / sanitization messages go to stderr by default and
    # duplicate the build's failure records: route them through logging, kept
    # only at DEBUG verbosity.
    from rdkit import rdBase

    rdBase.LogToPythonLogger()
    logging.getLogger("rdkit").setLevel(
        logging.DEBUG
        if (level or logging.WARNING) <= logging.DEBUG
        else logging.CRITICAL
    )
    return root


def build_file_logger(logger_name: str, log_path: Path | None) -> logging.Logger:
    """Return a dedicated, non-propagating logger with a fresh FileHandler.

    Idempotent per logger: existing handlers are closed first so repeated
    builds never duplicate lines. With ``log_path=None`` the logger has no
    handler and, since it never propagates, drops its records: the in-memory
    counters and main-log summaries still report them.
    """
    file_logger = logging.getLogger(logger_name)
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False
    for handler in list(file_logger.handlers):
        file_logger.removeHandler(handler)
        handler.close()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        file_logger.addHandler(handler)
    return file_logger


__all__ = ["build_file_logger", "setup_logging"]
