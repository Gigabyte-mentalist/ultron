"""Procedurally generated UI sound cues (docs/tz/02_audio.md §3.5).

No audio files ship with the project; every earcon is a short numpy tone
rendered on demand, so the audio layer stays self-contained. Every tone gets
a Hann-shaped fade in/out so it never clicks.
"""

from __future__ import annotations

import numpy as np

_FADE_MS = 8.0


def _envelope(n: int, sample_rate: int, fade_ms: float = _FADE_MS) -> np.ndarray:
    fade_n = max(1, min(n // 2, int(sample_rate * fade_ms / 1000)))
    env = np.ones(n, dtype=np.float32)
    ramp = (0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_n, dtype=np.float32)))).astype(np.float32)
    env[:fade_n] *= ramp
    env[-fade_n:] *= ramp[::-1]
    return env


def _tone(freq: float, duration_s: float, sample_rate: int, amplitude: float = 0.35) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    sig = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return (amplitude * sig * _envelope(n, sample_rate)).astype(np.float32)


def _sweep(
    f_start: float, f_end: float, duration_s: float, sample_rate: int, amplitude: float = 0.35
) -> np.ndarray:
    n = int(duration_s * sample_rate)
    freq = np.linspace(f_start, f_end, n, dtype=np.float32)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    sig = np.sin(phase).astype(np.float32)
    return (amplitude * sig * _envelope(n, sample_rate)).astype(np.float32)


def wake(sample_rate: int = 16000) -> np.ndarray:
    """~120 ms, rising two-note cue: wake word recognized."""
    a = _tone(587.33, 0.055, sample_rate)  # D5
    b = _tone(880.00, 0.055, sample_rate)  # A5
    gap = np.zeros(int(0.01 * sample_rate), dtype=np.float32)
    return np.concatenate([a, gap, b])


def ack(sample_rate: int = 16000) -> np.ndarray:
    """~80 ms, soft single blip: command received, thinking."""
    return _tone(700.0, 0.08, sample_rate, amplitude=0.22)


def done(sample_rate: int = 16000) -> np.ndarray:
    """~120 ms, descending tone: conversation finished."""
    return _sweep(880.0, 587.33, 0.12, sample_rate, amplitude=0.3)


def error(sample_rate: int = 16000) -> np.ndarray:
    """~200 ms, low tone: error."""
    return _tone(220.0, 0.2, sample_rate, amplitude=0.35)


def confirm(sample_rate: int = 16000) -> np.ndarray:
    """~150 ms, two short beeps: waiting for yes/no confirmation."""
    beep = _tone(660.0, 0.06, sample_rate, amplitude=0.28)
    gap = np.zeros(int(0.03 * sample_rate), dtype=np.float32)
    return np.concatenate([beep, gap, beep])


_EARCONS = {
    "wake": wake,
    "ack": ack,
    "done": done,
    "error": error,
    "confirm": confirm,
}


def generate(name: str, sample_rate: int = 16000) -> np.ndarray:
    try:
        fn = _EARCONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown earcon: {name!r} (available: {sorted(_EARCONS)})") from exc
    return fn(sample_rate)
