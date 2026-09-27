"""Tests for ultron.memory.db (docs/tz/07_tools.md §7D.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.memory import db as memory_db
from ultron.memory.db import Database


@pytest.fixture(autouse=True)
async def _reset_default() -> None:
    yield
    await memory_db.reset_for_tests()


async def test_connect_creates_db_file_and_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "sub" / "ultron.sqlite"
    db = Database(db_path)

    await db.connect()

    assert db_path.is_file()
    tables = await (
        await db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ).fetchall()
    names = {row[0] for row in tables}
    assert {"facts", "reminders", "conversations", "file_index"} <= names

    await db.close()


async def test_conn_raises_before_connect(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite")

    with pytest.raises(RuntimeError):
        _ = db.conn


async def test_connect_is_idempotent(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite")

    await db.connect()
    first_conn = db.conn
    await db.connect()

    assert db.conn is first_conn
    await db.close()


async def test_get_db_uses_configured_path(tmp_path: Path) -> None:
    target = tmp_path / "configured.sqlite"
    memory_db.configure(target)

    db = await memory_db.get_db()

    assert db.db_path == target
    assert target.is_file()


async def test_get_db_returns_same_instance_across_calls(tmp_path: Path) -> None:
    memory_db.configure(tmp_path / "shared.sqlite")

    first = await memory_db.get_db()
    second = await memory_db.get_db()

    assert first is second
