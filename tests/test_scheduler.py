"""Tests for ultron.memory.scheduler.ReminderScheduler (docs/tz/07_tools.md §7D.3).

``load_pending``'s missed-vs-rescheduled classification is pure datetime
comparison, tested with ``freezegun`` per the TZ's own note ("vaqtni
freezegun... bilan"). Actual firing is tested with a real (tiny-delay)
APScheduler run rather than freezegun, since mixing frozen time with the
scheduler's own asyncio-clock-driven waiting would be fragile.
"""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path

import pytest
from freezegun import freeze_time

from ultron.events import ReminderDue
from ultron.memory.db import Database
from ultron.memory.scheduler import ReminderScheduler, _build_trigger


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "sched_test.sqlite")
    await database.connect()
    yield database
    await database.close()


async def _insert(db: Database, text: str, due_at: str, repeat: str | None = None) -> int:
    cursor = await db.conn.execute(
        "INSERT INTO reminders (text, due_at, repeat, kind, status, created_at)"
        " VALUES (?, ?, ?, 'reminder', 'pending', ?)",
        (text, due_at, repeat, "2026-01-01T00:00:00"),
    )
    await db.conn.commit()
    return cursor.lastrowid


async def _status(db: Database, row_id: int) -> str:
    row = await (
        await db.conn.execute("SELECT status FROM reminders WHERE id = ?", (row_id,))
    ).fetchone()
    return row[0]


def test_build_trigger_maps_repeat_kinds() -> None:
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger

    due = datetime.datetime(2026, 1, 1, 9, 30, 0)

    assert isinstance(_build_trigger(due, None), DateTrigger)
    assert isinstance(_build_trigger(due, "daily"), CronTrigger)
    assert isinstance(_build_trigger(due, "weekdays"), CronTrigger)
    assert isinstance(_build_trigger(due, "weekly"), CronTrigger)


async def test_load_pending_reschedules_future_reminder(db: Database) -> None:
    with freeze_time("2026-03-10T08:00:00"):
        await _insert(db, "Kelajakdagi eslatma", "2026-03-10T09:00:00")

        scheduler = ReminderScheduler(db, on_due=lambda e: asyncio.sleep(0))
        rescheduled, missed = await scheduler.load_pending()

        assert rescheduled == 1
        assert missed == 0
        assert scheduler.scheduler.get_job("reminder-1") is not None


async def test_load_pending_marks_old_overdue_reminder_as_missed(db: Database) -> None:
    with freeze_time("2026-03-10T08:00:00"):
        row_id = await _insert(db, "Juda eski eslatma", "2026-03-08T08:00:00")  # 48h old

        scheduler = ReminderScheduler(db, on_due=lambda e: asyncio.sleep(0))
        rescheduled, missed = await scheduler.load_pending()

        assert rescheduled == 0
        assert missed == 1
        assert await _status(db, row_id) == "missed"


async def test_load_pending_reschedules_recently_overdue_reminder_as_overdue(db: Database) -> None:
    with freeze_time("2026-03-10T08:00:00"):
        await _insert(db, "Yaqinda o'tgan eslatma", "2026-03-10T07:00:00")  # 1h old

        fired: list[ReminderDue] = []

        async def on_due(event: ReminderDue) -> None:
            fired.append(event)

        scheduler = ReminderScheduler(db, on_due=on_due)
        rescheduled, missed = await scheduler.load_pending()

        assert rescheduled == 1
        assert missed == 0

    # The job's run_date is already in the past, so starting the scheduler
    # fires it almost immediately (misfire_grace_time=None never discards it).
    scheduler.start()
    try:
        await asyncio.sleep(0.3)
    finally:
        scheduler.shutdown()

    assert len(fired) == 1
    assert fired[0].overdue is True


async def test_recurring_reminder_is_never_classified_as_missed(db: Database) -> None:
    with freeze_time("2026-03-10T08:00:00"):
        await _insert(db, "Har kuni sport", "2026-01-01T07:00:00", repeat="daily")

        scheduler = ReminderScheduler(db, on_due=lambda e: asyncio.sleep(0))
        rescheduled, missed = await scheduler.load_pending()

        assert rescheduled == 1
        assert missed == 0


async def test_schedule_new_fires_on_due_and_marks_done(db: Database) -> None:
    row_id = await _insert(
        db, "Choy ichish", (datetime.datetime.now() + datetime.timedelta(seconds=60)).isoformat()
    )
    fired: list[ReminderDue] = []

    async def on_due(event: ReminderDue) -> None:
        fired.append(event)

    scheduler = ReminderScheduler(db, on_due=on_due)
    scheduler.start()
    try:
        due = datetime.datetime.now() + datetime.timedelta(milliseconds=50)
        scheduler.schedule_new(row_id, "Choy ichish", due, None, "reminder")
        await asyncio.sleep(0.3)
    finally:
        scheduler.shutdown()

    assert len(fired) == 1
    assert fired[0].id == row_id
    assert fired[0].overdue is False
    assert await _status(db, row_id) == "done"


async def test_recurring_job_does_not_mark_done_on_fire(db: Database) -> None:
    row_id = await _insert(db, "Takroriy", datetime.datetime.now().isoformat(), repeat="daily")
    fired: list[ReminderDue] = []

    async def on_due(event: ReminderDue) -> None:
        fired.append(event)

    scheduler = ReminderScheduler(db, on_due=on_due)
    # Fire directly, bypassing the cron trigger's real daily wait.
    await scheduler._fire(row_id, "Takroriy", "reminder", "daily", False)

    assert len(fired) == 1
    assert await _status(db, row_id) == "pending"


async def test_cancel_prevents_pending_job_from_firing(db: Database) -> None:
    row_id = await _insert(
        db,
        "Bekor qilinadigan",
        (datetime.datetime.now() + datetime.timedelta(seconds=60)).isoformat(),
    )
    fired: list[ReminderDue] = []

    async def on_due(event: ReminderDue) -> None:
        fired.append(event)

    scheduler = ReminderScheduler(db, on_due=on_due)
    scheduler.start()
    try:
        due = datetime.datetime.now() + datetime.timedelta(milliseconds=150)
        scheduler.schedule_new(row_id, "Bekor qilinadigan", due, None, "reminder")
        scheduler.cancel(row_id)
        await asyncio.sleep(0.3)
    finally:
        scheduler.shutdown()

    assert fired == []


def test_cancel_unknown_job_does_not_raise(db: Database) -> None:
    scheduler = ReminderScheduler(db, on_due=lambda e: asyncio.sleep(0))

    scheduler.cancel(9999)  # must not raise JobLookupError
