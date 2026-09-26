"""Always-on output stream with fast, click-free interruption (docs/tz/02_audio.md §3.4).

The output stream is opened once and kept running; TTS sentences are queued
and played back-to-back with :meth:`enqueue`. :meth:`stop` is the barge-in
primitive: it must silence playback within ~50 ms without a click, which is
why the fade-out happens inline in the realtime callback rather than being
scheduled from Python.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from math import gcd

import numpy as np
import sounddevice as sd
from loguru import logger
from scipy.signal import resample_poly

from ultron.audio._device_utils import refresh_device_list, resolve_device

_BACKOFF_SCHEDULE_S = (1.0, 2.0, 5.0, 10.0)
_PLAYBACK_BLOCK = 256
_FADE_OUT_MS = 10.0


class AudioPlayer:
    def __init__(
        self,
        *,
        sample_rate: int,
        output_device: str | None,
        output_volume: float,
        on_playback_start: Callable[[], None] | None = None,
        on_playback_end: Callable[[], None] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._device_name = output_device
        self.output_volume = output_volume
        self._on_playback_start = on_playback_start
        self._on_playback_end = on_playback_end

        self._queue: deque[np.ndarray] = deque()
        self._queue_lock = threading.Lock()
        self._current: np.ndarray | None = None
        self._pos = 0
        self._is_playing = False
        self._stop_requested = False
        self._output_sr = sample_rate
        self._fade_samples = max(1, int(sample_rate * _FADE_OUT_MS / 1000))

        self._stream: sd.OutputStream | None = None
        self._thread: threading.Thread | None = None
        self._closing = True

    # -- public API ---------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    def start(self) -> None:
        if self._thread is not None:
            return
        self._closing = False
        self._thread = threading.Thread(
            target=self._supervisor_loop, name="AudioPlayer", daemon=True
        )
        self._thread.start()

    def stop_stream(self) -> None:
        """Shut the output stream down entirely (distinct from :meth:`stop`).

        Only the supervisor thread ever opens/closes the stream (inside its
        own ``with sd.OutputStream(...)`` block) -- closing it here too raced
        with that and corrupted PortAudio's internal state.
        """
        self._closing = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    async def play(self, pcm: np.ndarray, sr: int) -> None:
        self.enqueue(pcm, sr)

    def enqueue(self, pcm: np.ndarray, sr: int) -> None:
        clip = self._prepare_clip(pcm, sr)
        with self._queue_lock:
            self._queue.append(clip)

    def stop(self) -> None:
        """Silence playback within ~50 ms and clear the pending queue (barge-in)."""
        self._stop_requested = True
        deadline = time.monotonic() + 0.05
        while self._is_playing and time.monotonic() < deadline:
            time.sleep(0.002)

    # -- clip preparation -----------------------------------------------

    def _prepare_clip(self, pcm: np.ndarray, sr: int) -> np.ndarray:
        """Resample straight to the stream's *actual* rate (off the realtime thread).

        Resampling inside the audio callback would risk a real glitch on
        every clip's first block, since a multi-second clip can take longer
        to resample than one callback's ~5 ms real-time budget.
        """
        target_sr = self._output_sr
        clip = np.asarray(pcm, dtype=np.float32)
        if sr != target_sr:
            g = gcd(sr, target_sr)
            clip = resample_poly(clip, target_sr // g, sr // g).astype(np.float32)
        return np.clip(clip * self.output_volume, -1.0, 1.0).astype(np.float32)

    # -- supervisor / reconnect logic ----------------------------------

    def _supervisor_loop(self) -> None:
        backoff_idx = 0
        while not self._closing:
            try:
                self._open_and_run_stream()
                backoff_idx = 0
            except Exception as exc:  # noqa: BLE001 - must never kill the process
                logger.warning("Audio player stream error: {}", exc)
                delay = _BACKOFF_SCHEDULE_S[min(backoff_idx, len(_BACKOFF_SCHEDULE_S) - 1)]
                backoff_idx += 1
                self._sleep_unless_closing(delay)

    def _sleep_unless_closing(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._closing and time.monotonic() < deadline:
            time.sleep(0.1)

    def _open_and_run_stream(self) -> None:
        refresh_device_list()
        device_index = resolve_device(self._device_name, kind="output")
        self._output_sr = self._probe_output_sample_rate(device_index)
        self._fade_samples = max(1, int(self._output_sr * _FADE_OUT_MS / 1000))

        extra_settings = None
        if sys.platform == "win32":
            with contextlib.suppress(Exception):
                extra_settings = sd.WasapiSettings(exclusive=False)

        with sd.OutputStream(
            samplerate=self._output_sr,
            channels=1,
            dtype="float32",
            blocksize=_PLAYBACK_BLOCK,
            latency="low",
            device=device_index,
            callback=self._stream_callback,
            extra_settings=extra_settings,
        ) as stream:
            self._stream = stream
            while not self._closing:
                time.sleep(0.5)
                if not stream.active:
                    raise RuntimeError("output stream became inactive")
        self._stream = None

    def _probe_output_sample_rate(self, device_index: int | None) -> int:
        try:
            sd.check_output_settings(
                device=device_index, samplerate=self._sample_rate, channels=1, dtype="float32"
            )
            return self._sample_rate
        except Exception:
            fallback = int(sd.query_devices(device_index)["default_samplerate"])
            logger.info(
                "Output device does not support {} Hz directly, playing at {} Hz and resampling",
                self._sample_rate,
                fallback,
            )
            return fallback

    # -- audio callback (runs on the PortAudio thread) -------------------

    def _stream_callback(self, outdata: np.ndarray, frame_count: int, time_info, status) -> None:
        if status:
            logger.debug("Audio player status flags: {}", status)
        out = outdata[:, 0]

        if self._stop_requested:
            self._stop_requested = False
            out[:] = 0.0
            if self._current is not None and self._pos < len(self._current):
                n = min(self._fade_samples, len(self._current) - self._pos, frame_count)
                ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
                out[:n] = self._current[self._pos : self._pos + n] * ramp
            self._current = None
            self._pos = 0
            with self._queue_lock:
                self._queue.clear()
            self._set_playing(False)
            return

        filled = 0
        while filled < frame_count:
            if self._current is None or self._pos >= len(self._current):
                if not self._advance_clip():
                    out[filled:] = 0.0
                    return
            n = min(frame_count - filled, len(self._current) - self._pos)
            out[filled : filled + n] = self._current[self._pos : self._pos + n]
            self._pos += n
            filled += n

    def _advance_clip(self) -> bool:
        with self._queue_lock:
            if not self._queue:
                self._current = None
                self._pos = 0
                self._set_playing(False)
                return False
            self._current = self._queue.popleft()
        self._pos = 0
        self._set_playing(True)
        return True

    def _set_playing(self, playing: bool) -> None:
        if playing == self._is_playing:
            return
        self._is_playing = playing
        callback = self._on_playback_start if playing else self._on_playback_end
        if callback:
            callback()
