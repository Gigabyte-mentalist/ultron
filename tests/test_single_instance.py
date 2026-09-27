"""Tests for ultron.core.single_instance.SingleInstanceLock
(docs/tz/09_autostart.md §3.2.1).

Runs the real Windows mutex API (this project is Windows-only, see
CLAUDE.md) rather than faking it -- the whole point of this class is a
handful of OS calls, so a fake would just test the fake.
"""

from __future__ import annotations

import uuid

from ultron.core.single_instance import SingleInstanceLock


def _unique_name() -> str:
    return f"UltronTest{uuid.uuid4().hex}"


def test_second_instance_fails_to_acquire_and_exits_immediately() -> None:
    name = _unique_name()
    first = SingleInstanceLock(name)
    second = SingleInstanceLock(name)

    assert first.acquire() is True
    assert second.acquire() is False

    first.release()
    second.release()


def test_lock_is_reacquirable_after_release() -> None:
    name = _unique_name()
    first = SingleInstanceLock(name)
    assert first.acquire() is True
    first.release()

    second = SingleInstanceLock(name)
    assert second.acquire() is True
    second.release()


def test_different_names_do_not_conflict() -> None:
    a = SingleInstanceLock(_unique_name())
    b = SingleInstanceLock(_unique_name())

    assert a.acquire() is True
    assert b.acquire() is True

    a.release()
    b.release()


def test_context_manager_releases_on_exit() -> None:
    name = _unique_name()
    with SingleInstanceLock(name) as lock:
        assert lock is not None
        second = SingleInstanceLock(name)
        assert second.acquire() is False

    third = SingleInstanceLock(name)
    assert third.acquire() is True
    third.release()
