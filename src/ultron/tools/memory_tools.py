"""Long-term memory tools: remember/recall/forget facts about the user,
backed by the shared SQLite ``facts`` table (docs/tz/07_tools.md §7D.2).

``get_facts_for_prompt`` feeds ``llm/prompts.py``'s ``{facts}`` slot. It must
be synchronous (``llm/agent.py`` calls its ``facts_provider`` without
``await``), so an in-memory cache mirrors the table; :func:`load_cache` fills
it once at startup and every mutating tool below keeps it in sync.
"""

from __future__ import annotations

import datetime
from typing import Annotated

from loguru import logger

from ultron.config import RootConfig
from ultron.memory import db as memory_db
from ultron.tools.base import ToolResult, is_mock_mode, tool

_config: RootConfig | None = None
_facts_cache: dict[str, str] = {}  # key -> value, insertion order == recency


def configure(config: RootConfig) -> None:
    global _config
    _config = config


async def load_cache() -> None:
    """Populate the in-memory cache from SQLite. Call once at startup."""
    db = await memory_db.get_db()
    cursor = await db.conn.execute("SELECT key, value FROM facts ORDER BY updated_at ASC")
    rows = await cursor.fetchall()
    _facts_cache.clear()
    for key, value in rows:
        _facts_cache[key] = value


def get_facts_for_prompt() -> list[str]:
    """Sync accessor for ``llm/prompts.build_system_prompt``'s ``facts``
    argument: the most recently updated ``max_facts_in_prompt`` facts, as
    ``"key: value"`` strings (docs/tz/07_tools.md §7D.2)."""
    max_n = _config.memory.max_facts_in_prompt if _config else 40
    if max_n <= 0:
        return []
    items = list(_facts_cache.items())[-max_n:]
    return [f"{key}: {value}" for key, value in items]


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


@tool(
    name="remember_fact",
    description="Remember a fact about the user for future conversations,"
    " keyed by a short identifier.",
)
async def remember_fact(
    key: Annotated[str, "Short identifier for this fact, e.g. 'birthday'"],
    value: Annotated[str, "The fact's value, e.g. '15-mart'"],
    category: Annotated[str, "Grouping category, e.g. 'personal'"] = "general",
) -> ToolResult:
    key_norm = key.strip().lower()
    value_norm = value.strip()
    if not key_norm or not value_norm:
        return ToolResult(ok=False, error="key va value bo'sh bo'lmasligi kerak")

    if is_mock_mode():
        logger.bind(audit=True).info(
            "[MOCK] remember_fact {}={} ({})", key_norm, value_norm, category
        )
        return ToolResult(
            ok=True, data={"key": key_norm, "value": value_norm}, speak="Yodda saqladim."
        )

    db = await memory_db.get_db()
    now = _now_iso()
    await db.conn.execute(
        """
        INSERT INTO facts (key, value, category, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value, category = excluded.category, updated_at = excluded.updated_at
        """,
        (key_norm, value_norm, category, now, now),
    )
    await db.conn.commit()

    _facts_cache.pop(key_norm, None)  # re-insert at the end -> most recent
    _facts_cache[key_norm] = value_norm

    return ToolResult(ok=True, data={"key": key_norm, "value": value_norm}, speak="Yodda saqladim.")


@tool(
    name="recall_facts",
    description="Recall previously remembered facts, optionally filtered by a search text.",
)
async def recall_facts(
    query: Annotated[str, "Optional search text to filter by key or value"] | None = None,
) -> ToolResult:
    db = await memory_db.get_db()
    if query and query.strip():
        like = f"%{query.strip().lower()}%"
        cursor = await db.conn.execute(
            "SELECT key, value, category FROM facts"
            " WHERE lower(key) LIKE ? OR lower(value) LIKE ? ORDER BY updated_at DESC",
            (like, like),
        )
    else:
        cursor = await db.conn.execute(
            "SELECT key, value, category FROM facts ORDER BY updated_at DESC"
        )
    rows = await cursor.fetchall()
    if not rows:
        return ToolResult(ok=True, data={"facts": []}, speak="Hech narsa topilmadi.")

    facts = [{"key": r[0], "value": r[1], "category": r[2]} for r in rows]
    speak = "; ".join(f"{f['key']}: {f['value']}" for f in facts[:5])
    return ToolResult(ok=True, data={"facts": facts}, speak=speak)


@tool(
    name="forget_fact",
    description="Forget a previously remembered fact by its key.",
    dangerous=True,
    confirm_question_uz="Bu faktni unutishni tasdiqlaysizmi?",
)
async def forget_fact(key: Annotated[str, "The fact's key to forget"]) -> ToolResult:
    key_norm = key.strip().lower()

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] forget_fact {}", key_norm)
        return ToolResult(ok=True, data={"key": key_norm}, speak="Unutdim.")

    db = await memory_db.get_db()
    cursor = await db.conn.execute("DELETE FROM facts WHERE key = ?", (key_norm,))
    await db.conn.commit()
    if cursor.rowcount == 0:
        return ToolResult(ok=False, error=f"'{key}' nomli fakt topilmadi")

    _facts_cache.pop(key_norm, None)
    return ToolResult(ok=True, data={"key": key_norm}, speak="Unutdim.")
