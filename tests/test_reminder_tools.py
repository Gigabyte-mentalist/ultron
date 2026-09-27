"""Tests for ultron.tools.reminder_tools (docs/tz/07_tools.md §7D.3).

The real ``ReminderScheduler``/APScheduler firing behavior is covered by
test_scheduler.py; here ``_get_scheduler`` is replaced with a small
recording fake so these tests focus on validation and SQLite state.
"""

from __future__ import annotations

import datetime

import pytest

from ultron.events import ReminderDue
from ultron.memory import db as memory_db
from ultron.tools import reminder_tools


class _FakeScheduler:
    def __init__(self) -> None:
        self.scheduled: list[tuple] = []
        self.cancelled: list[int] = []

    def schedule_new(self, row_id, text, due, repeat, kind, *, overdue: bool = False) -> None:
        self.scheduled.append((row_id, text, due, repeat, kind))

    def cancel(self, row_id: int) -> None:
        self.cancelled.append(row_id)


@pytest.fixture
def fake_scheduler(monkeypatch: pytest.MonkeyPatch) -> _FakeScheduler:
    fake = _FakeScheduler()

    async def _fake_get_scheduler() -> _FakeScheduler:
        return fake

    monkeypatch.setattr(reminder_tools, "_get_scheduler", _fake_get_scheduler)
    return fake


@pytest.fixture(autouse=True)
async def _reset_state(tmp_path_factory: pytest.TempPathFactory) -> None:
    db_path = tmp_path_factory.mktemp("remtools_db") / "test.sqlite"
    memory_db.configure(db_path)
    reminder_tools._scheduler = None
    reminder_tools._due_listeners.clear()
    yield
    await memory_db.reset_for_tests()
    reminder_tools._config = None
    reminder_tools._scheduler = None
    reminder_tools._due_listeners.clear()


def _future_iso(seconds: int = 3600) -> str:
    return (datetime.datetime.now() + datetime.timedelta(seconds=seconds)).isoformat()


# -- set_reminder -------------------------------------------------------------


async def test_set_reminder_inserts_row_and_schedules(
    fake_scheduler: _FakeScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)

    result = await reminder_tools.set_reminder(text="Choy ich", due_at_iso=_future_iso())

    assert result.ok is True
    assert len(fake_scheduler.scheduled) == 1
    listed = await reminder_tools.list_reminders()
    assert listed.data["reminders"][0]["text"] == "Choy ich"


async def test_set_reminder_rejects_past_datetime(fake_scheduler: _FakeScheduler) -> None:
    past = (datetime.datetime.now() - datetime.timedelta(seconds=60)).isoformat()

    result = await reminder_tools.set_reminder(text="Kech", due_at_iso=past)

    assert result.ok is False
    assert fake_scheduler.scheduled == []


async def test_set_reminder_rejects_invalid_iso(fake_scheduler: _FakeScheduler) -> None:
    result = await reminder_tools.set_reminder(text="x", due_at_iso="not-a-date")

    assert result.ok is False


async def test_set_reminder_rejects_invalid_repeat(fake_scheduler: _FakeScheduler) -> None:
    result = await reminder_tools.set_reminder(text="x", due_at_iso=_future_iso(), repeat="hourly")

    assert result.ok is False


async def test_set_reminder_allows_past_time_when_repeating(
    fake_scheduler: _FakeScheduler,
) -> None:
    past = (datetime.datetime.now() - datetime.timedelta(seconds=60)).isoformat()

    result = await reminder_tools.set_reminder(text="Har kuni", due_at_iso=past, repeat="daily")

    assert result.ok is True


async def test_set_reminder_mock_mode_does_not_touch_db_or_scheduler(
    fake_scheduler: _FakeScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await reminder_tools.set_reminder(text="Ghost", due_at_iso=_future_iso())

    assert result.ok is True
    assert fake_scheduler.scheduled == []
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    listed = await reminder_tools.list_reminders()
    assert listed.data["reminders"] == []


# -- set_timer -----------------------------------------------------------------


async def test_set_timer_inserts_and_schedules(
    fake_scheduler: _FakeScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)

    result = await reminder_tools.set_timer(seconds=300, label="Choy")

    assert result.ok is True
    assert len(fake_scheduler.scheduled) == 1
    assert fake_scheduler.scheduled[0][4] == "timer"


async def test_set_timer_rejects_non_positive_seconds(fake_scheduler: _FakeScheduler) -> None:
    result = await reminder_tools.set_timer(seconds=0)

    assert result.ok is False


async def test_set_timer_default_label(
    fake_scheduler: _FakeScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)

    result = await reminder_tools.set_timer(seconds=10)

    assert result.data["text"] == "Taymer"


async def test_set_timer_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await reminder_tools.set_timer(seconds=10)

    assert result.ok is True


# -- list_reminders -------------------------------------------------------------


async def test_list_reminders_empty() -> None:
    result = await reminder_tools.list_reminders()

    assert result.ok is True
    assert result.data["reminders"] == []


async def test_list_reminders_orders_by_due_at(
    fake_scheduler: _FakeScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    await reminder_tools.set_reminder(text="Ikkinchi", due_at_iso=_future_iso(7200))
    await reminder_tools.set_reminder(text="Birinchi", due_at_iso=_future_iso(60))

    result = await reminder_tools.list_reminders()

    assert [r["text"] for r in result.data["reminders"]] == ["Birinchi", "Ikkinchi"]


# -- cancel_reminder ------------------------------------------------------------


async def test_cancel_reminder_by_id(
    fake_scheduler: _FakeScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    created = await reminder_tools.set_reminder(text="Bekor qilinsin", due_at_iso=_future_iso())
    reminder_id = created.data["id"]

    result = await reminder_tools.cancel_reminder(id_or_text=str(reminder_id))

    assert result.ok is True
    assert fake_scheduler.cancelled == [reminder_id]
    listed = await reminder_tools.list_reminders()
    assert listed.data["reminders"] == []


async def test_cancel_reminder_by_fuzzy_text(
    fake_scheduler: _FakeScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    await reminder_tools.set_reminder(text="Doktorga borish", due_at_iso=_future_iso())

    result = await reminder_tools.cancel_reminder(id_or_text="doktor")

    assert result.ok is True


async def test_cancel_reminder_no_match() -> None:
    result = await reminder_tools.cancel_reminder(id_or_text="mavjud emas narsa")

    assert result.ok is False


async def test_cancel_reminder_mock_mode_does_not_touch_scheduler(
    fake_scheduler: _FakeScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    created = await reminder_tools.set_reminder(text="Test", due_at_iso=_future_iso())
    reminder_id = created.data["id"]
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await reminder_tools.cancel_reminder(id_or_text=str(reminder_id))

    assert result.ok is True
    assert fake_scheduler.cancelled == []


# -- due handling / toast -------------------------------------------------------


async def test_add_due_listener_is_invoked_on_handle_due(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_toast(event: ReminderDue) -> None:
        return None

    monkeypatch.setattr(reminder_tools, "_show_toast", _noop_toast)
    received: list[ReminderDue] = []

    async def listener(event: ReminderDue) -> None:
        received.append(event)

    reminder_tools.add_due_listener(listener)

    await reminder_tools._handle_due(ReminderDue(id=1, text="x", kind="reminder", repeat=None))

    assert len(received) == 1


async def test_handle_due_skips_toast_in_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    calls = []
    monkeypatch.setattr(reminder_tools, "_show_toast", lambda e: calls.append(e))

    await reminder_tools._handle_due(ReminderDue(id=1, text="x", kind="reminder", repeat=None))

    assert calls == []


async def test_show_toast_calls_win11toast(monkeypatch: pytest.MonkeyPatch) -> None:
    import win11toast

    calls = []
    monkeypatch.setattr(win11toast, "toast", lambda title, msg: calls.append((title, msg)))

    await reminder_tools._show_toast(
        ReminderDue(id=1, text="Choy", kind="reminder", repeat=None, overdue=False)
    )

    assert calls == [("Ultron", "Eslatma: Choy")]


async def test_show_toast_prefixes_overdue(monkeypatch: pytest.MonkeyPatch) -> None:
    import win11toast

    calls = []
    monkeypatch.setattr(win11toast, "toast", lambda title, msg: calls.append((title, msg)))

    await reminder_tools._show_toast(
        ReminderDue(id=1, text="Choy", kind="reminder", repeat=None, overdue=True)
    )

    assert calls == [("Ultron", "O'tkazib yuborilgan eslatma: Choy")]


async def test_show_toast_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import win11toast

    def _boom(title: str, msg: str) -> None:
        raise RuntimeError("no toast backend")

    monkeypatch.setattr(win11toast, "toast", _boom)

    await reminder_tools._show_toast(ReminderDue(id=1, text="Choy", kind="reminder", repeat=None))
    # must not raise
