"""Shared SQLite persistence: facts, reminders, conversation log, and the
file search index (docs/tz/07_tools.md §7B.2, §7D.1).

One :class:`Database` wraps a single ``aiosqlite`` connection (WAL mode) and
applies the full schema on first ``connect()``. ``tools/file_tools.py``,
``tools/memory_tools.py``, and ``tools/reminder_tools.py`` all share one
instance via :func:`get_db`, configured once at startup with
:func:`configure`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY,
  key TEXT UNIQUE,
  value TEXT NOT NULL,
  category TEXT DEFAULT 'general',
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS reminders (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  due_at TEXT NOT NULL,
  repeat TEXT DEFAULT NULL,
  kind TEXT DEFAULT 'reminder',
  status TEXT DEFAULT 'pending',
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY,
  ts TEXT,
  role TEXT,
  text TEXT,
  tool_name TEXT,
  latency_ms INTEGER
);

CREATE TABLE IF NOT EXISTS file_index (
  path TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  ext TEXT,
  size INTEGER,
  mtime REAL
);
CREATE INDEX IF NOT EXISTS idx_file_index_name ON file_index(name);
"""

DEFAULT_DB_PATH = Path("data/db/ultron.sqlite")


class Database:
    """Thin ``aiosqlite`` wrapper around one connection to ``db_path``."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._conn is not None:
            return
        async with self._connect_lock:
            if self._conn is not None:  # re-check inside the lock
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(self.db_path))
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.executescript(_SCHEMA_SQL)
            await conn.commit()
            self._conn = conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database ulanmagan -- avval connect() chaqirilishi kerak")
        return self._conn

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()


_db: Database | None = None


def configure(db_path: Path) -> None:
    """Point future :func:`get_db` calls at ``db_path`` (docs/tz/07_tools.md
    §7D.1's ``memory.db_path``). Replaces any previously configured/opened
    database -- callers holding an older instance should stop using it."""
    global _db
    _db = Database(Path(db_path))


async def get_db() -> Database:
    """Return the shared, connected :class:`Database`, creating it with the
    default path if :func:`configure` was never called (e.g. a script run
    without going through ``ultron.tools.configure_all``)."""
    global _db
    if _db is None:
        _db = Database(DEFAULT_DB_PATH)
    await _db.connect()
    return _db


async def close_db() -> None:
    """Close the shared connection, if one was ever opened, and drop it so
    the next :func:`get_db` call reconnects fresh. ``aiosqlite`` runs each
    connection's blocking I/O on a dedicated background thread that only
    stops once ``close()`` is called -- skipping this leaves that thread (and
    the process) alive after ``__main__.py``'s main loop returns. Call once
    at shutdown (``ultron.tools.async_shutdown_all``)."""
    global _db
    if _db is not None:
        await _db.close()
    _db = None


async def reset_for_tests() -> None:
    """Test-only alias for :func:`close_db` (kept for readability at call sites)."""
    await close_db()
