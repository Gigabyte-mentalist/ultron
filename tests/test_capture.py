"""Tests for ultron.audio.capture.AudioCapture (docs/tz/02_audio.md §4).

@hardware: needs a real microphone. Run manually:
    uv run pytest tests/test_capture.py -m hardware
"""

from __future__ import annotations

import asyncio
import time

import pytest

from ultron.audio.capture import AudioCapture

pytestmark = pytest.mark.hardware


async def test_receives_at_least_90_frames_in_3_seconds_of_steady_capture() -> None:
    """90 frames * 32 ms = 2.88 s of steady-state capture.

    Measured from the first frame, not from object construction: opening a
    real WASAPI stream has its own (~0.3-0.5 s) startup latency that isn't
    part of what this criterion is checking (sustained ~31 fps delivery).
    An 8 s wall-clock ceiling still catches a genuine hang.
    """
    capture = AudioCapture(sample_rate=16000, channels=1, block_size=512, input_device=None)
    timestamps: list[float] = []

    async def _collect() -> None:
        async for _frame in capture.frames():
            timestamps.append(time.monotonic())
            if len(timestamps) >= 90:
                break

    try:
        await asyncio.wait_for(_collect(), timeout=8.0)
    finally:
        capture.stop()

    assert len(timestamps) >= 90
    assert timestamps[-1] - timestamps[0] <= 3.0
