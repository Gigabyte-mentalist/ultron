"""Suspend/resume detection via a hidden window (docs/tz/09_autostart.md §3.5).

Windows only delivers ``WM_POWERBROADCAST`` to windows that pump messages,
so this creates an invisible, message-only-style top-level window on its own
dedicated thread purely to receive ``PBT_APMSUSPEND``/
``PBT_APMRESUMEAUTOMATIC`` and forward them out via plain callbacks. The
callbacks run on *this* thread, not the caller's asyncio loop -- callers
that need to reach asyncio from them use ``events.EventBus.publish``, which
is thread-safe by design (see ``events.py``'s module docstring).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable

from loguru import logger

PBT_APMSUSPEND = 0x0004
PBT_APMRESUMEAUTOMATIC = 0x0012
WM_POWERBROADCAST = 0x0218


class PowerWatcher:
    def __init__(self, *, on_suspend: Callable[[], None], on_resume: Callable[[], None]) -> None:
        self._on_suspend = on_suspend
        self._on_resume = on_resume
        self._thread: threading.Thread | None = None
        self._hwnd: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="PowerWatcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._hwnd is not None:
            import win32con
            import win32gui

            with contextlib.suppress(Exception):
                win32gui.PostMessage(self._hwnd, win32con.WM_CLOSE, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            import win32api
            import win32con
            import win32gui
        except ImportError:
            logger.warning("pywin32 mavjud emas, uyqu/uyg'onish kuzatuvi o'chirildi")
            return

        def _wnd_proc(hwnd: int, msg: int, wparam: int, lparam: int) -> int:
            if msg == WM_POWERBROADCAST:
                if wparam == PBT_APMSUSPEND:
                    logger.debug("WM_POWERBROADCAST: PBT_APMSUSPEND")
                    self._on_suspend()
                elif wparam == PBT_APMRESUMEAUTOMATIC:
                    logger.debug("WM_POWERBROADCAST: PBT_APMRESUMEAUTOMATIC")
                    self._on_resume()
                return True
            if msg == win32con.WM_CLOSE:
                win32gui.DestroyWindow(hwnd)
                return 0
            if msg == win32con.WM_DESTROY:
                win32gui.PostQuitMessage(0)
                return 0
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        try:
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = _wnd_proc
            wc.lpszClassName = "UltronPowerWatcherWindow"
            wc.hInstance = win32api.GetModuleHandle(None)
            try:
                class_atom = win32gui.RegisterClass(wc)
            except Exception:  # noqa: BLE001 -- already registered by a prior run in-process
                class_atom = wc.lpszClassName

            self._hwnd = win32gui.CreateWindow(
                class_atom,
                "UltronPowerWatcher",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                wc.hInstance,
                None,
            )
            win32gui.PumpMessages()
        except Exception:  # noqa: BLE001 -- must never take the whole process down
            logger.exception("Uyqu/uyg'onish kuzatuvchisida xato")
        finally:
            self._hwnd = None
