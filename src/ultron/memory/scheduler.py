"""Reminder/timer scheduling: APScheduler fires jobs in-process; the SQLite
``reminders`` table is the durable source of truth, reloaded at startup
(docs/tz/07_tools.md §7D.3).
"""

from __future__ import annotations

import datetime
from collections.abc import Awaitable, Callable
from typing import Any

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from ultron.events import ReminderDue
from ultron.memory.db import Database

MISSED_CUTOFF_HOURS = 24


def _build_trigger(due: datetime.datetime, repeat: str | None) -> Any:
    if repeat is None:
        return DateTrigger(run_date=due)
    if repeat == "daily":
        return CronTrigger(hour=due.hour, minute=due.minute, second=due.second)
    if repeat == "weekdays":
        return CronTrigger(
            day_of_week="mon-fri", hour=due.hour, minute=due.minute, second=due.second
        )
    if repeat == "weekly":
        return CronTrigger(
            day_of_week=due.weekday(), hour=due.hour, minute=due.minute, second=due.second
        )
    return CronTrigger.from_crontab(repeat)  # arbitrary cron expression, escape hatch


class ReminderScheduler:
    """Wraps one ``AsyncIOScheduler``. ``on_due`` is awaited from inside the
    running event loop every time a reminder/timer fires."""

    def __init__(self, db: Database, on_due: Callable[[ReminderDue], Awaitable[None]]) -> None:
        self._db = db
        self._on_due = on_due
        self.scheduler = AsyncIOScheduler()

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    async def load_pending(self) -> tuple[int, int]:
        """Reschedule every ``pending`` reminder from SQLite (called once at
        startup). Returns ``(rescheduled, missed)``; anything more than
        :data:`MISSED_CUTOFF_HOURS` overdue is marked ``missed`` instead of
        firing, per §7D.3."""
        cursor = await self._db.conn.execute(
            "SELECT id, text, due_at, repeat, kind FROM reminders WHERE status = 'pending'"
        )
        rows = await cursor.fetchall()
        now = datetime.datetime.now()
        rescheduled = 0
        missed = 0
        for row_id, text, due_at, repeat, kind in rows:
            due = datetime.datetime.fromisoformat(due_at)
            if repeat is None and due < now - datetime.timedelta(hours=MISSED_CUTOFF_HOURS):
                await self._mark_status(row_id, "missed")
                missed += 1
                continue
            overdue = repeat is None and due < now
            self.schedule_new(row_id, text, due, repeat, kind, overdue=overdue)
            rescheduled += 1
        return rescheduled, missed

    def schedule_new(
        self,
        row_id: int,
        text: str,
        due: datetime.datetime,
        repeat: str | None,
        kind: str,
        *,
        overdue: bool = False,
    ) -> None:
        trigger = _build_trigger(due, repeat)
        self.scheduler.add_job(
            self._fire,
            trigger=trigger,
            args=[row_id, text, kind, repeat, overdue],
            id=f"reminder-{row_id}",
            replace_existing=True,
            misfire_grace_time=None,
        )

    def cancel(self, row_id: int) -> None:
        try:
            self.scheduler.remove_job(f"reminder-{row_id}")
        except JobLookupError:
            pass

    async def _fire(
        self, row_id: int, text: str, kind: str, repeat: str | None, overdue: bool
    ) -> None:
        if repeat is None:
            await self._mark_status(row_id, "done")
        await self._on_due(
            ReminderDue(id=row_id, text=text, kind=kind, repeat=repeat, overdue=overdue)
        )

    async def _mark_status(self, row_id: int, status: str) -> None:
        await self._db.conn.execute(
            "UPDATE reminders SET status = ? WHERE id = ?", (status, row_id)
        )
        await self._db.conn.commit()
