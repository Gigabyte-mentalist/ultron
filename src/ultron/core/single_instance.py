"""Windows named-mutex single-instance guard (docs/tz/09_autostart.md §3.2.1).

Two independent locks are used: ``UltronSupervisor`` (held by
``ultron.supervisor``) and ``UltronMain`` (held by the full voice pipeline,
``ultron.__main__._run_voice_mode``) -- a second supervisor launch (e.g. two
Task Scheduler triggers firing close together) and a second main-process
launch (e.g. someone running ``python -m ultron`` by hand while the
supervisor-managed instance is already up) each need to notice the other and
exit immediately rather than fight over the microphone/tray icon.

``Local\\`` (not ``Global\\``) is used because the mutex only needs to be
unique within this user's logon session -- ``Global\\`` requires elevated
privileges the Task Scheduler "Limited" run level intentionally avoids (§3.3).
"""

from __future__ import annotations

import sys
from types import TracebackType


class SingleInstanceLock:
    """``acquire()`` returns ``True`` iff this process now holds the lock.

    The underlying OS mutex handle is released automatically when the
    process exits (or ``release()``/``__exit__`` is called), which is what
    lets a crashed instance's lock be reacquired by the next launch without
    any manual cleanup.
    """

    def __init__(self, name: str) -> None:
        self._name = f"Local\\{name}"
        self._handle: int | None = None

    def acquire(self) -> bool:
        if sys.platform != "win32":  # pragma: no cover -- this project is Windows-only
            return True

        import win32api
        import win32event
        import winerror

        handle = win32event.CreateMutex(None, False, self._name)
        already_running = win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS
        if already_running:
            win32api.CloseHandle(handle)
            return False

        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is not None:
            import win32api

            win32api.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
