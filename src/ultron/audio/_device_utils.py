"""Shared sounddevice device-selection helpers for capture.py and player.py."""

from __future__ import annotations

import contextlib
import sys
import threading
import time

import sounddevice as sd

# ``AudioCapture`` and ``AudioPlayer`` each run their own supervisor thread
# and independently call ``refresh_device_list()`` at the top of every
# (re)connect attempt -- including one triggered *by* the other's refresh.
# ``sd._terminate()``/``sd._initialize()`` tear down and rebuild PortAudio's
# entire process-wide session, which invalidates *any* other stream that
# happens to be open at that moment (raising "Invalid stream pointer" on the
# other side). Without a rate limit, that becomes a self-sustaining cascade:
# capture's refresh kills player's stream -> player errors and reconnects,
# refreshing in turn -> kills capture's (just-reopened) stream -> forever.
# Confirmed by reproducing it with capture+player run together in isolation,
# with neither Nahimic nor any other software involved. Capping actual
# refreshes to once per interval breaks the cascade -- after the first
# mutual collision, later reconnects skip the destructive step and the
# already-current device list (a device that's still valid a few seconds
# later needs no reinit) lets both streams settle. Hot-plug detection still
# works, just debounced rather than instant.
_refresh_lock = threading.Lock()
_last_refresh_at = 0.0
_MIN_REFRESH_INTERVAL_S = 5.0


def refresh_device_list() -> None:
    """Force PortAudio to re-enumerate devices so hot-plug changes are seen --
    at most once per ``_MIN_REFRESH_INTERVAL_S``, see the module docstring above."""
    global _last_refresh_at
    with _refresh_lock:
        now = time.monotonic()
        if now - _last_refresh_at < _MIN_REFRESH_INTERVAL_S:
            return
        _last_refresh_at = now
    with contextlib.suppress(Exception):
        sd._terminate()
        sd._initialize()


def resolve_device(name: str | None, *, kind: str) -> int | None:
    """Resolve a device index for ``kind`` ("input" or "output").

    ``name=None`` picks the system default device, preferring the WASAPI
    host API on Windows when the default device is also reachable through
    it. A non-null ``name`` is matched case-insensitively as a substring.
    """
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    wasapi_idx = next((i for i, h in enumerate(hostapis) if "wasapi" in h["name"].lower()), None)

    candidates = [(i, d) for i, d in enumerate(devices) if d[channels_key] > 0]
    if not candidates:
        raise RuntimeError(f"No {kind} audio devices found")

    if name:
        needle = name.lower()
        matched = [(i, d) for i, d in candidates if needle in d["name"].lower()]
        if not matched:
            raise RuntimeError(f"No {kind} device matching {name!r} found")
        candidates = matched

    if sys.platform == "win32" and wasapi_idx is not None:
        wasapi_candidates = [(i, d) for i, d in candidates if d["hostapi"] == wasapi_idx]
        if wasapi_candidates:
            candidates = wasapi_candidates

    if name is None:
        default_idx = sd.default.device[0 if kind == "input" else 1]
        for i, _ in candidates:
            if i == default_idx:
                return i

    return candidates[0][0]
