"""Tests for ultron.supervisor (docs/tz/09_autostart.md §3.2/§4).

``RestartPolicy`` is pure (no subprocess/mutex/file IO) so it's driven
directly. ``Supervisor``'s child-lifecycle methods are exercised against a
small in-process fake standing in for ``asyncio.subprocess.Process`` --
spawning a real ``pythonw.exe -m ultron`` would need a full model/audio
stack and is out of scope for a fast unit test (covered instead by
``docs/manual_tests_09.md``'s real-hardware checklist).
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from ultron.config import RootConfig
from ultron.supervisor import EXIT_CLEAN_QUIT, EXIT_IMMEDIATE_RESTART, RestartPolicy, Supervisor

# -- RestartPolicy ------------------------------------------------------


def test_exit_zero_stops() -> None:
    policy = RestartPolicy()
    decision = policy.decide(EXIT_CLEAN_QUIT)
    assert decision.action == "stop"


def test_exit_ten_restarts_immediately_and_resets_backoff() -> None:
    policy = RestartPolicy()
    # Rack up some backoff first.
    policy.decide(1)
    policy.decide(1)

    decision = policy.decide(EXIT_IMMEDIATE_RESTART)
    assert decision.action == "restart_now"

    # Backoff should be back at the start (2 s), not continuing to escalate.
    next_decision = policy.decide(1)
    assert next_decision.action == "restart_after_delay"
    assert next_decision.delay_s == RestartPolicy.BACKOFF_S[0]


def test_crash_backoff_escalates_then_cooldown_kicks_in() -> None:
    """With the tuned defaults, ``len(BACKOFF_S) == CRASH_THRESHOLD == 5``: the
    backoff ladder is walked exactly once before the 6th crash in the window
    hits cooldown instead of a 6th (nonexistent) backoff step -- and the
    crash right after cooldown starts the ladder over from the bottom."""
    policy = RestartPolicy()
    actions_and_delays = [(d.action, d.delay_s) for d in (policy.decide(1) for _ in range(7))]

    assert actions_and_delays == [
        ("restart_after_delay", RestartPolicy.BACKOFF_S[0]),
        ("restart_after_delay", RestartPolicy.BACKOFF_S[1]),
        ("restart_after_delay", RestartPolicy.BACKOFF_S[2]),
        ("restart_after_delay", RestartPolicy.BACKOFF_S[3]),
        ("restart_after_delay", RestartPolicy.BACKOFF_S[4]),
        ("cooldown", RestartPolicy.COOLDOWN_S),
        ("restart_after_delay", RestartPolicy.BACKOFF_S[0]),
    ]


def test_backoff_pins_at_longest_step_once_ladder_is_exhausted() -> None:
    """Isolates the backoff ladder's own clamping behavior from the
    crash-count cooldown by using a threshold well above the ladder's
    length -- with the tuned defaults the two happen to line up exactly
    (see the test above), so this only matters if either constant changes."""

    class _HighThresholdPolicy(RestartPolicy):
        CRASH_THRESHOLD = 100

    policy = _HighThresholdPolicy()
    delays = [policy.decide(1).delay_s for _ in range(len(RestartPolicy.BACKOFF_S) + 2)]

    assert delays[: len(RestartPolicy.BACKOFF_S)] == list(RestartPolicy.BACKOFF_S)
    assert delays[-1] == RestartPolicy.BACKOFF_S[-1]
    assert delays[-2] == RestartPolicy.BACKOFF_S[-1]


def test_more_than_five_crashes_in_window_triggers_cooldown() -> None:
    now = [0.0]
    policy = RestartPolicy(clock=lambda: now[0])

    decisions = []
    for _ in range(6):
        decisions.append(policy.decide(1))
        now[0] += 1.0  # all well within the 10-minute window

    assert [d.action for d in decisions[:5]] == ["restart_after_delay"] * 5
    assert decisions[5].action == "cooldown"
    assert decisions[5].delay_s == RestartPolicy.COOLDOWN_S


def test_crashes_outside_window_do_not_count_toward_threshold() -> None:
    now = [0.0]
    policy = RestartPolicy(clock=lambda: now[0])

    for _ in range(5):
        policy.decide(1)
        now[0] += 1.0

    # Far outside the 10-minute crash window -- the counter should reset.
    now[0] += RestartPolicy.CRASH_WINDOW_S + 1.0
    decision = policy.decide(1)

    assert decision.action == "restart_after_delay"
    assert decision.delay_s == RestartPolicy.BACKOFF_S[0]


def test_cooldown_resets_backoff_and_crash_history() -> None:
    now = [0.0]
    policy = RestartPolicy(clock=lambda: now[0])
    for _ in range(6):
        policy.decide(1)
        now[0] += 1.0

    # After the cooldown, the very next crash should look like a fresh start.
    decision = policy.decide(1)
    assert decision.action == "restart_after_delay"
    assert decision.delay_s == RestartPolicy.BACKOFF_S[0]


# -- Supervisor: child lifecycle ------------------------------------------


class _FakeProcess:
    """Resolves ``wait()`` immediately with a fixed exit code."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False
        self.pid = 1234

    async def wait(self) -> int:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class _HangingProcess:
    """Never exits on its own -- ``wait()`` only resolves once ``terminate()``
    (or ``kill()``) is called, simulating a wedged main process."""

    def __init__(self, exit_code_on_kill: int = 1) -> None:
        self._future: asyncio.Future[int] = asyncio.get_event_loop().create_future()
        self.exit_code_on_kill = exit_code_on_kill
        self.terminated = False
        self.killed = False
        self.pid = 5678

    async def wait(self) -> int:
        return await self._future

    def terminate(self) -> None:
        self.terminated = True
        if not self._future.done():
            self._future.set_result(self.exit_code_on_kill)

    def kill(self) -> None:
        self.killed = True
        if not self._future.done():
            self._future.set_result(self.exit_code_on_kill)


def _make_supervisor(tmp_path: Path) -> Supervisor:
    config = RootConfig()
    config.app.data_dir = tmp_path
    return Supervisor(config, project_root=tmp_path)


async def test_supervise_child_returns_exit_code_when_it_exits_on_its_own(
    tmp_path: Path,
) -> None:
    supervisor = _make_supervisor(tmp_path)
    proc = _FakeProcess(0)

    exit_code = await supervisor._supervise_child(proc)

    assert exit_code == 0
    assert proc.terminated is False


async def test_supervise_child_kills_process_when_heartbeat_goes_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ultron.supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "_HEARTBEAT_POLL_S", 0.05)
    monkeypatch.setattr(supervisor_module, "_HEARTBEAT_STALE_S", 0.1)

    supervisor = _make_supervisor(tmp_path)
    heartbeat = supervisor._heartbeat_path
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.write_text(str(time.time()), encoding="utf-8")
    # Backdate it well past the (test-shortened) staleness threshold.
    stale_time = time.time() - 10
    os.utime(heartbeat, (stale_time, stale_time))

    proc = _HangingProcess(exit_code_on_kill=1)

    exit_code = await asyncio.wait_for(supervisor._supervise_child(proc), timeout=5.0)

    assert exit_code == 1
    assert proc.terminated is True


async def test_supervise_child_does_not_kill_when_heartbeat_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No heartbeat file yet just means "still starting up" -- must not be
    mistaken for a hang (docs/tz/09_autostart.md §3.2 step 5)."""
    import ultron.supervisor as supervisor_module

    monkeypatch.setattr(supervisor_module, "_HEARTBEAT_POLL_S", 0.05)
    monkeypatch.setattr(supervisor_module, "_HEARTBEAT_STALE_S", 0.1)

    supervisor = _make_supervisor(tmp_path)
    proc = _HangingProcess(exit_code_on_kill=1)

    task = asyncio.ensure_future(supervisor._supervise_child(proc))
    await asyncio.sleep(0.3)  # several poll cycles' worth
    assert proc.terminated is False

    proc.terminate()  # let the test end cleanly
    await task


# -- Supervisor: full restart-loop routing (spawn/ollama faked out) --------


class _Done(Exception):
    """Raised by the fake spawner once the scripted exit-code sequence is
    exhausted, so ``Supervisor.run()``'s ``while True`` ends deterministically
    instead of needing a real child process to eventually quit."""


async def _noop_async() -> None:
    return None


async def _noop_async_arg(_text: str) -> None:
    return None


def _drive_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_codes: list[int]
) -> tuple[Supervisor, list[float]]:
    """Wires a ``Supervisor`` whose ``_spawn_child`` hands back
    ``_FakeProcess(code)`` for each ``code`` in ``exit_codes`` in turn (and
    whose ``asyncio.sleep`` calls -- both the startup delay and every
    backoff/cooldown wait -- are recorded instead of actually waited out),
    ready for the caller to ``await supervisor.run()``."""
    import ultron.supervisor as supervisor_module

    supervisor = _make_supervisor(tmp_path)
    monkeypatch.setattr(supervisor, "_ensure_ollama", _noop_async)
    monkeypatch.setattr(supervisor, "_toast", _noop_async_arg)

    codes = iter(exit_codes)
    sleeps: list[float] = []

    async def _fake_spawn() -> _FakeProcess:
        try:
            code = next(codes)
        except StopIteration as exc:
            raise _Done from exc
        return _FakeProcess(code)

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(supervisor, "_spawn_child", _fake_spawn)
    monkeypatch.setattr(supervisor_module.asyncio, "sleep", _fake_sleep)
    return supervisor, sleeps


async def test_run_stops_on_exit_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor, sleeps = _drive_run(tmp_path, monkeypatch, [EXIT_CLEAN_QUIT])

    await supervisor.run()  # returns normally (no _Done raised) -> "stop"

    assert sleeps == [supervisor._config.autostart.startup_delay_s]


async def test_run_restarts_immediately_on_exit_ten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor, sleeps = _drive_run(
        tmp_path, monkeypatch, [EXIT_IMMEDIATE_RESTART, EXIT_CLEAN_QUIT]
    )

    await supervisor.run()

    # Only the one-time startup delay was slept through -- no backoff delay
    # was inserted between the exit-10 restart and the next spawn.
    assert sleeps == [supervisor._config.autostart.startup_delay_s]


async def test_run_backs_off_after_a_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    supervisor, sleeps = _drive_run(tmp_path, monkeypatch, [1, EXIT_CLEAN_QUIT])

    await supervisor.run()

    assert sleeps == [supervisor._config.autostart.startup_delay_s, RestartPolicy.BACKOFF_S[0]]
