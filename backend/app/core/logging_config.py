"""
Application-wide logging configuration.

Every module obtains its logger via `get_logger(__name__)` so log
records carry the originating module path, making the ML pipeline and
API layers independently traceable in the console/log file.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "noiseguard.log", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers, force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
