"""Reminders and timers: set/list/cancel, backed by SQLite + APScheduler
(docs/tz/07_tools.md §7D.3).
"""

from __future__ import annotations

import asyncio
import datetime
from collections.abc import Awaitable, Callable
from typing import Annotated

from loguru import logger
from rapidfuzz import fuzz, process

from ultron.config import RootConfig
from ultron.events import ReminderDue
from ultron.i18n.uz import REMINDER_OVERDUE_PREFIX, REMINDER_PREFIX
from ultron.memory import db as memory_db
from ultron.memory.scheduler import ReminderScheduler
from ultron.tools.base import ToolResult, is_mock_mode, tool

_VALID_REPEATS: frozenset[str | None] = frozenset({None, "daily", "weekdays", "weekly"})
_FUZZY_SCORE_CUTOFF = 60
_UPCOMING_LIMIT = 10

_config: RootConfig | None = None
_scheduler: ReminderScheduler | None = None
_due_listeners: list[Callable[[ReminderDue], Awaitable[None]]] = []


def configure(config: RootConfig) -> None:
    global _config
    _config = config


def add_due_listener(callback: Callable[[ReminderDue], Awaitable[None]]) -> None:
    """Registration hook for docs/tz/08_orchestrator.md: the orchestrator
    subscribes here to play the confirm earcon, speak ``event.text`` via
    TTS, and queue it if Ultron is already talking. Multiple listeners may
    be registered."""
    _due_listeners.append(callback)


async def _show_toast(event: ReminderDue) -> None:
    try:
        import win11toast

        prefix = REMINDER_OVERDUE_PREFIX if event.overdue else REMINDER_PREFIX
        await asyncio.to_thread(win11toast.toast, "Ultron", prefix + event.text)
    except Exception as exc:  # noqa: BLE001 -- a toast failure must not break the scheduler
        logger.warning("Toast ko'rsatib bo'lmadi: {}", exc)


async def _handle_due(event: ReminderDue) -> None:
    logger.bind(audit=True).info(
        "reminder_due id={} text={} kind={} overdue={}",
        event.id,
        event.text,
        event.kind,
        event.overdue,
    )
    if not is_mock_mode():
        await _show_toast(event)
    for listener in list(_due_listeners):
        await listener(event)


async def _get_scheduler() -> ReminderScheduler:
    global _scheduler
    if _scheduler is None:
        db = await memory_db.get_db()
        _scheduler = ReminderScheduler(db, _handle_due)
        _scheduler.start()
    return _scheduler


def shutdown() -> None:
    """Stop the live scheduler, if one was ever started. Call once at
    shutdown (``ultron.tools.async_shutdown_all``)."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown()
        _scheduler = None


async def reschedule_pending() -> tuple[int, int]:
    """Reload every pending reminder/timer from SQLite into the live
    scheduler. Call once at startup (docs/tz/07_tools.md §7D.3's "ishga
    tushishda... qayta rejalashtiriladi"). Returns ``(rescheduled, missed)``."""
    scheduler = await _get_scheduler()
    return await scheduler.load_pending()


async def _insert_reminder(text: str, due: datetime.datetime, repeat: str | None, kind: str) -> int:
    db = await memory_db.get_db()
    cursor = await db.conn.execute(
        "INSERT INTO reminders (text, due_at, repeat, kind, status, created_at)"
        " VALUES (?, ?, ?, ?, 'pending', ?)",
        (
            text,
            due.isoformat(),
            repeat,
            kind,
            datetime.datetime.now().isoformat(timespec="seconds"),
        ),
    )
    await db.conn.commit()
    return cursor.lastrowid


@tool(
    name="set_reminder",
    description=(
        "Schedule a one-time or repeating reminder for a specific local datetime"
        " (ISO 8601, already resolved from the user's relative time by the caller)."
    ),
)
async def set_reminder(
    text: Annotated[str, "What to remind the user about"],
    due_at_iso: Annotated[str, "ISO 8601 local datetime, e.g. '2026-09-28T09:00:00'"],
    repeat: Annotated[str, "One of: daily, weekdays, weekly"] | None = None,
) -> ToolResult:
    try:
        due = datetime.datetime.fromisoformat(due_at_iso)
    except ValueError:
        return ToolResult(ok=False, error="due_at_iso ISO 8601 formatida bo'lishi kerak")
    if repeat not in _VALID_REPEATS:
        return ToolResult(ok=False, error=f"Noto'g'ri repeat qiymati: {repeat}")
    if repeat is None and due <= datetime.datetime.now():
        return ToolResult(ok=False, error="Bu vaqt allaqachon o'tib ketgan")

    if is_mock_mode():
        logger.bind(audit=True).info(
            "[MOCK] set_reminder {} @ {} repeat={}", text, due_at_iso, repeat
        )
        return ToolResult(
            ok=True, data={"text": text, "due_at": due.isoformat()}, speak="Eslatma qo'yildi."
        )

    reminder_id = await _insert_reminder(text, due, repeat, "reminder")
    scheduler = await _get_scheduler()
    scheduler.schedule_new(reminder_id, text, due, repeat, "reminder")

    return ToolResult(
        ok=True,
        data={"id": reminder_id, "text": text, "due_at": due.isoformat()},
        speak="Eslatma qo'yildi.",
    )


@tool(name="set_timer", description="Set a countdown timer for N seconds from now.")
async def set_timer(
    seconds: Annotated[int, "Seconds from now"],
    label: Annotated[str, "Optional label for the timer"] | None = None,
) -> ToolResult:
    if seconds <= 0:
        return ToolResult(ok=False, error="seconds musbat son bo'lishi kerak")

    due = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
    text = label or "Taymer"

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] set_timer {}s label={}", seconds, label)
        return ToolResult(
            ok=True, data={"text": text, "due_at": due.isoformat()}, speak="Taymer qo'yildi."
        )

    reminder_id = await _insert_reminder(text, due, None, "timer")
    scheduler = await _get_scheduler()
    scheduler.schedule_new(reminder_id, text, due, None, "timer")

    return ToolResult(
        ok=True,
        data={"id": reminder_id, "text": text, "due_at": due.isoformat()},
        speak="Taymer qo'yildi.",
    )


@tool(name="list_reminders", description="List the next upcoming pending reminders/timers.")
async def list_reminders() -> ToolResult:
    db = await memory_db.get_db()
    cursor = await db.conn.execute(
        "SELECT id, text, due_at, repeat, kind FROM reminders"
        " WHERE status = 'pending' ORDER BY due_at ASC LIMIT ?",
        (_UPCOMING_LIMIT,),
    )
    rows = await cursor.fetchall()
    if not rows:
        return ToolResult(ok=True, data={"reminders": []}, speak="Hech qanday eslatma yo'q.")

    reminders = [
        {"id": r[0], "text": r[1], "due_at": r[2], "repeat": r[3], "kind": r[4]} for r in rows
    ]
    speak = "; ".join(f"{r['text']} — {r['due_at']}" for r in reminders[:3])
    return ToolResult(ok=True, data={"reminders": reminders}, speak=speak)


@tool(
    name="cancel_reminder",
    description="Cancel a pending reminder/timer by its id or a fuzzy match on its text.",
)
async def cancel_reminder(
    id_or_text: Annotated[str, "Reminder id, or text to fuzzy-match against pending reminders"],
) -> ToolResult:
    db = await memory_db.get_db()
    cursor = await db.conn.execute("SELECT id, text FROM reminders WHERE status = 'pending'")
    rows = await cursor.fetchall()
    if not rows:
        return ToolResult(ok=False, error="Bekor qilinadigan eslatma yo'q")

    target_id: int | None = None
    stripped = id_or_text.strip()
    if stripped.isdigit():
        candidate_id = int(stripped)
        if any(row_id == candidate_id for row_id, _text in rows):
            target_id = candidate_id
    if target_id is None:
        choices = {row_id: text for row_id, text in rows}
        best = process.extractOne(
            stripped, choices, scorer=fuzz.WRatio, score_cutoff=_FUZZY_SCORE_CUTOFF
        )
        if best:
            target_id = best[2]
    if target_id is None:
        return ToolResult(ok=False, error=f"'{id_or_text}' bilan mos eslatma topilmadi")

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] cancel_reminder {}", target_id)
        return ToolResult(ok=True, data={"id": target_id}, speak="Eslatma bekor qilindi.")

    await db.conn.execute("UPDATE reminders SET status = 'cancelled' WHERE id = ?", (target_id,))
    await db.conn.commit()
    scheduler = await _get_scheduler()
    scheduler.cancel(target_id)

    return ToolResult(ok=True, data={"id": target_id}, speak="Eslatma bekor qilindi.")
