"""Loguru setup: colored console sink, rotating app log, separate audit log.

Call :func:`setup_logging` once, as early as possible, before any other
module logs anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from ultron.config import LoggingConfig

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"


def _is_audit_record(record: dict) -> bool:
    return bool(record["extra"].get("audit", False))


def setup_logging(cfg: LoggingConfig, data_dir: Path) -> None:
    """Configure loguru sinks: console, app log file, and tools audit log."""
    logs_dir = Path(data_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    has_console = sys.stdout is not None
    if has_console:
        logger.add(
            sys.stdout,
            level=cfg.level,
            format=_CONSOLE_FORMAT,
            colorize=True,
            filter=lambda record: not _is_audit_record(record),
        )

    logger.add(
        logs_dir / "ultron.log",
        level=cfg.level,
        format=_FILE_FORMAT,
        rotation=cfg.rotation,
        retention=cfg.retention,
        encoding="utf-8",
        filter=lambda record: not _is_audit_record(record),
    )

    logger.add(
        logs_dir / "tools_audit.log",
        level="INFO",
        format=_FILE_FORMAT,
        rotation=cfg.rotation,
        retention=cfg.retention,
        encoding="utf-8",
        filter=_is_audit_record,
    )
