"""Tests for ultron.audio.player.AudioPlayer (docs/tz/02_audio.md §4).

@hardware: needs a real output device. Run manually:
    uv run pytest tests/test_player.py -m hardware
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ultron.audio.player import AudioPlayer

pytestmark = pytest.mark.hardware


def test_stop_silences_within_50ms() -> None:
    player = AudioPlayer(sample_rate=16000, output_device=None, output_volume=0.3)
    player.start()
    try:
        tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000 * 3) / 16000)).astype(np.float32)
        player.enqueue(tone, 16000)
        time.sleep(0.3)
        assert player.is_playing

        t0 = time.monotonic()
        player.stop()
        elapsed = time.monotonic() - t0

        assert elapsed <= 0.05
        assert not player.is_playing
    finally:
        player.stop_stream()


def test_enqueue_plays_clips_back_to_back() -> None:
    started = []
    ended = []
    player = AudioPlayer(
        sample_rate=16000,
        output_device=None,
        output_volume=0.3,
        on_playback_start=lambda: started.append(time.monotonic()),
        on_playback_end=lambda: ended.append(time.monotonic()),
    )
    player.start()
    try:
        tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)).astype(np.float32)
        player.enqueue(tone, 16000)
        player.enqueue(tone, 16000)
        time.sleep(1.2)

        assert len(started) == 1  # back-to-back clips don't re-fire on_playback_start
        assert len(ended) == 1
        assert not player.is_playing
    finally:
        player.stop_stream()
