"""Fixed-capacity circular audio buffer (docs/tz/02_audio.md §3.2).

Keeps the last ``capacity_seconds`` of int16 mono PCM so a speech segment's
pre-roll can be recovered after the endpointer has already confirmed speech
started -- without this, the first syllable ("Al" of "Altron") would be lost
to the detector's own reaction time.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity_seconds: float, sample_rate: int) -> None:
        self._capacity = max(1, int(capacity_seconds * sample_rate))
        self._sample_rate = sample_rate
        self._buf = np.zeros(self._capacity, dtype=np.int16)
        self._write_pos = 0
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, pcm: np.ndarray) -> None:
        n = len(pcm)
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity:
                self._buf[:] = pcm[-self._capacity :]
                self._write_pos = 0
                self._filled = self._capacity
                return
            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos : end] = pcm
            else:
                first = self._capacity - self._write_pos
                self._buf[self._write_pos :] = pcm[:first]
                self._buf[: end - self._capacity] = pcm[first:]
            self._write_pos = end % self._capacity
            self._filled = min(self._capacity, self._filled + n)

    def snapshot(self, seconds: float) -> np.ndarray:
        """Return the last ``seconds`` of audio, oldest sample first."""
        with self._lock:
            n = min(int(seconds * self._sample_rate), self._filled)
            if n <= 0:
                return np.zeros(0, dtype=np.int16)
            start = (self._write_pos - n) % self._capacity
            if start + n <= self._capacity:
                return self._buf[start : start + n].copy()
            first = self._capacity - start
            return np.concatenate([self._buf[start:], self._buf[: n - first]])

    def clear(self) -> None:
        with self._lock:
            self._write_pos = 0
            self._filled = 0
