"""Watchdog process that keeps the main Ultron process running (docs/tz/09_autostart.md §3.2).

Launched by Task Scheduler as ``pythonw.exe -m ultron.supervisor`` (no
console window -- see ``scripts/install_autostart.ps1``). This process never
touches the GPU, audio devices, or Ollama for real work -- it only
starts/monitors/restarts the child ``pythonw.exe -m ultron`` process, which
is where all of that lives. Kept deliberately separate from ``core.orchestrator``
so a wedged or crashing main process can always be recovered from outside.

:class:`RestartPolicy` is pure decision logic (no subprocess/mutex/file IO),
so ``tests/test_supervisor.py`` can drive it directly and deterministically
without spawning any real process.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ultron.config import ConfigError, RootConfig, load_config
from ultron.core.ollama_launcher import ensure_ollama_running
from ultron.core.single_instance import SingleInstanceLock
from ultron.i18n.uz import REPEATED_CRASH_TOAST

# subprocess.CREATE_NO_WINDOW -- named locally so this module stays importable
# (for RestartPolicy's tests) on non-Windows CI.
_CREATE_NO_WINDOW = 0x08000000

_HEARTBEAT_STALE_S = 30.0
_HEARTBEAT_POLL_S = 5.0
_KILL_WAIT_S = 5.0
_OLLAMA_WAIT_S = 60.0

# docs/tz/09_autostart.md §3.2 steps 7-8: the main process's two meaningful
# exit codes; everything else is "it crashed."
EXIT_CLEAN_QUIT = 0
EXIT_IMMEDIATE_RESTART = 10

# Files a heartbeat-window disk cleanup (core.orchestrator's health monitor)
# must never delete out from under an actively-running process.
ACTIVE_LOG_NAMES = frozenset(
    {"ultron.log", "tools_audit.log", "supervisor.log", "metrics.jsonl", "ultron_stdout.log"}
)


@dataclass
class RestartDecision:
    action: str  # "stop" | "restart_now" | "restart_after_delay" | "cooldown"
    delay_s: float = 0.0


class RestartPolicy:
    """docs/tz/09_autostart.md §3.2 steps 6-8.

    - exit 0 -> ``"stop"`` (the user quit via tray; supervisor exits too).
    - exit 10 -> ``"restart_now"``, backoff reset (the user hit tray
      "Qayta ishga tushirish" -- a full, immediate process relaunch).
    - anything else -> a crash: 2/5/10/30/60 s backoff, escalating with each
      consecutive crash. More than 5 crashes within a rolling 10-minute
      window -> ``"cooldown"`` for 5 minutes (and the caller shows a toast).
    """

    BACKOFF_S: tuple[float, ...] = (2.0, 5.0, 10.0, 30.0, 60.0)
    CRASH_WINDOW_S = 600.0
    CRASH_THRESHOLD = 5
    COOLDOWN_S = 300.0

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._crash_times: list[float] = []

    def decide(self, exit_code: int) -> RestartDecision:
        if exit_code == EXIT_CLEAN_QUIT:
            return RestartDecision("stop")
        if exit_code == EXIT_IMMEDIATE_RESTART:
            self._crash_times.clear()
            return RestartDecision("restart_now")

        now = self._clock()
        self._crash_times = [t for t in self._crash_times if now - t < self.CRASH_WINDOW_S]
        self._crash_times.append(now)
        count = len(self._crash_times)

        if count > self.CRASH_THRESHOLD:
            self._crash_times.clear()
            return RestartDecision("cooldown", delay_s=self.COOLDOWN_S)

        # The backoff step is derived from how many crashes fall inside the
        # rolling window (not a separately-incrementing counter), so it
        # naturally eases back down once old crashes age out of the window
        # instead of only ever resetting via "restart_now"/cooldown.
        delay = self.BACKOFF_S[min(count - 1, len(self.BACKOFF_S) - 1)]
        return RestartDecision("restart_after_delay", delay_s=delay)


class Supervisor:
    def __init__(self, config: RootConfig, *, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._policy = RestartPolicy()
        self._data_dir = Path(config.app.data_dir)
        self._heartbeat_path = self._data_dir / "tmp" / "heartbeat"
        self._stdout_log_path = self._data_dir / "logs" / "ultron_stdout.log"

    async def run(self) -> None:
        await asyncio.sleep(self._config.autostart.startup_delay_s)
        while True:
            await self._ensure_ollama()
            proc = await self._spawn_child()
            exit_code = await self._supervise_child(proc)
            logger.info("Asosiy jarayon tugadi (exit_code={})", exit_code)

            decision = self._policy.decide(exit_code)
            if decision.action == "stop":
                logger.info("Toza chiqish (exit_code=0), supervisor ham to'xtaydi")
                return
            if decision.action == "restart_now":
                logger.info("Darhol qayta ishga tushirilmoqda (exit_code=10)")
                continue
            if decision.action == "cooldown":
                logger.warning(
                    "{} daqiqada {} dan ko'p marta qulashdi, {:.0f} daqiqa kutilmoqda",
                    self._policy.CRASH_WINDOW_S / 60,
                    self._policy.CRASH_THRESHOLD,
                    decision.delay_s / 60,
                )
                await self._toast(REPEATED_CRASH_TOAST)
                await asyncio.sleep(decision.delay_s)
                continue
            logger.warning("{:.0f} soniyadan keyin qayta ishga tushiriladi", decision.delay_s)
            await asyncio.sleep(decision.delay_s)

    # -- child process lifecycle -----------------------------------------

    async def _spawn_child(self) -> asyncio.subprocess.Process:
        pythonw = sys.executable
        self._stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        # A stale heartbeat left over from a prior run must not read as
        # "fresh" before the new child has written its own first one.
        with contextlib.suppress(OSError):
            self._heartbeat_path.unlink()

        log_fh = open(self._stdout_log_path, "ab")  # noqa: SIM115 -- lives across the child's run
        try:
            creationflags = _CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = await asyncio.create_subprocess_exec(
                pythonw,
                "-m",
                "ultron",
                cwd=str(self._project_root),
                stdout=log_fh,
                stderr=log_fh,
                creationflags=creationflags,
            )
        finally:
            log_fh.close()  # the child has its own handle; ours isn't needed anymore
        logger.info("Asosiy jarayon ishga tushirildi (pid={})", proc.pid)
        return proc

    async def _supervise_child(self, proc: asyncio.subprocess.Process) -> int:
        wait_task = asyncio.ensure_future(proc.wait())
        try:
            while True:
                done, _pending = await asyncio.wait({wait_task}, timeout=_HEARTBEAT_POLL_S)
                if wait_task in done:
                    return wait_task.result()
                if self._heartbeat_stale():
                    logger.warning(
                        "Heartbeat {:.0f} s dan beri yangilanmadi, jarayon osilib qolgan "
                        "hisoblanadi, o'chirilmoqda",
                        _HEARTBEAT_STALE_S,
                    )
                    await self._kill(proc)
                    return await wait_task
        finally:
            if not wait_task.done():
                wait_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await wait_task

    def _heartbeat_stale(self) -> bool:
        try:
            mtime = self._heartbeat_path.stat().st_mtime
        except OSError:
            return False  # not written yet -- still starting up, not a hang
        return (time.time() - mtime) > _HEARTBEAT_STALE_S

    async def _kill(self, proc: asyncio.subprocess.Process) -> None:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_KILL_WAIT_S)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    # -- ollama / toast ----------------------------------------------------

    async def _ensure_ollama(self) -> None:
        ready = await ensure_ollama_running(
            self._config.llm.host,
            start_if_missing=self._config.autostart.ollama_start_if_missing,
            wait_s=_OLLAMA_WAIT_S,
        )
        if not ready:
            logger.warning(
                "Ollama {:.0f} soniyada tayyor bo'lmadi, baribir davom etamiz", _OLLAMA_WAIT_S
            )

    async def _toast(self, text: str) -> None:
        try:
            import win11toast

            await asyncio.to_thread(win11toast.toast, "Ultron", text)
        except Exception as exc:  # noqa: BLE001 -- a toast failure must not break anything
            logger.warning("Toast ko'rsatib bo'lmadi: {}", exc)


# -- process entry point --------------------------------------------------


def _setup_logging(data_dir: Path) -> None:
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        logs_dir / "supervisor.log",
        level="INFO",
        rotation="10 MB",
        retention="14 days",
        encoding="utf-8",
    )
    if sys.stdout is not None:
        with contextlib.suppress(Exception):
            logger.add(sys.stdout, level="INFO")


def _emergency_log(message: str) -> None:
    """Config didn't even load, so ``app.data_dir`` is unknown -- fall back
    to the documented default project path (docs/tz/00_umumiy_arxitektura.md
    §4) rather than losing the error entirely."""
    try:
        fallback = Path("C:/ultron/data/logs/supervisor_bootstrap_error.log")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with fallback.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}\n")
    except OSError:
        pass


async def _amain() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        _emergency_log(f"Config yuklashda xato: {exc}")
        return 1

    _setup_logging(Path(config.app.data_dir))

    lock = SingleInstanceLock("UltronSupervisor")
    if not lock.acquire():
        logger.info("Supervisor allaqachon ishlamoqda, chiqilmoqda")
        return 0

    # src/ultron/supervisor.py -> ultron -> src -> project root.
    project_root = Path(__file__).resolve().parents[2]
    try:
        await Supervisor(config, project_root=project_root).run()
    finally:
        lock.release()
    return 0


def main() -> None:
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
