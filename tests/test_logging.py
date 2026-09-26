"""Tests for ultron.logging_setup (docs/tz/01_muhit_sozlash.md §4.9)."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ultron.config import LoggingConfig
from ultron.logging_setup import setup_logging


def test_audit_sink_only_receives_audit_records(tmp_path: Path) -> None:
    setup_logging(LoggingConfig(), tmp_path)

    logger.info("normal event")
    logger.bind(audit=True).info("dangerous tool called")

    logger.remove()  # flush and close file handles before reading them back

    app_log = (tmp_path / "logs" / "ultron.log").read_text(encoding="utf-8")
    audit_log = (tmp_path / "logs" / "tools_audit.log").read_text(encoding="utf-8")

    assert "normal event" in app_log
    assert "dangerous tool called" not in app_log

    assert "dangerous tool called" in audit_log
    assert "normal event" not in audit_log


def test_creates_logs_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "data"

    setup_logging(LoggingConfig(), data_dir)
    logger.remove()

    assert (data_dir / "logs").is_dir()
