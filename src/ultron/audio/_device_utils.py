"""Shared sounddevice device-selection helpers for capture.py and player.py."""

from __future__ import annotations

import contextlib
import sys

import sounddevice as sd


def refresh_device_list() -> None:
    """Force PortAudio to re-enumerate devices so hot-plug changes are seen."""
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
