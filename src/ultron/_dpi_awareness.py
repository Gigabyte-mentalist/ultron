"""Marks this process per-monitor DPI aware (Windows only) -- must run before
any window is created, exactly like ``_cuda_dlls.py`` (imported right after
it, for the same "first thing, no exceptions" reason): Windows only allows a
process to set its own DPI awareness once, before its first window/device
context exists.

Without this, Windows silently scales a non-DPI-aware process's windows to
compensate for display scaling. That breaks
``ultron.companion.widget.CompanionWidget``'s screen-corner placement in a
very specific way: ``tkinter.Tk.winfo_screenwidth()`` reports a *virtualized*
logical size (e.g. 1536x864 on a 1920x1080 panel at 125% scaling) instead of
the real pixel count, so math done against it lands the window somewhere
other than the actual corner once Windows stretches the result back up to
physical pixels -- confirmed by screenshotting the widget before this fix
existed. It also leaves every window (tray icon, the "Matnli buyruq" dialog,
the companion overlay) blurrier than necessary.
"""

from __future__ import annotations

import os

from loguru import logger


def _set_dpi_aware() -> None:
    if os.name != "nt":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return
    except Exception:  # noqa: BLE001 -- shcore is Windows 8.1+; fall back below
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # Vista+ fallback, system-DPI-aware only
    except Exception:  # noqa: BLE001 -- cosmetic only, must never block startup
        logger.debug("DPI awareness sozlanmadi (muhim emas)")


_set_dpi_aware()
