"""Continuous, resilient microphone capture (docs/tz/02_audio.md §3.1).

Runs 24/7. The callback given to ``sounddevice`` does nothing but push a
frame onto a thread-safe queue; everything else (device resolution,
resampling, reconnect backoff, watchdog) happens on a supervisor thread so
the audio callback thread never blocks.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import queue
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from loguru import logger
from scipy.signal import resample_poly

from ultron.audio._device_utils import refresh_device_list, resolve_device
from ultron.events import AudioFrame

_BACKOFF_SCHEDULE_S = (1.0, 2.0, 5.0, 10.0)
_DEVICE_LOST_AFTER_S = 60.0
_WATCHDOG_SILENCE_S = 2.0
_WATCHDOG_POLL_S = 0.5
_DROP_LOG_INTERVAL_S = 1.0
_FALLBACK_CAPTURE_SR = 48000


@dataclass
class AudioDeviceLost:
    ts: float
    detail: str


class AudioCapture:
    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        block_size: int,
        input_device: str | None,
        on_device_lost: Callable[[AudioDeviceLost], None] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size
        self._device_name = input_device
        self._on_device_lost = on_device_lost

        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._capture_sr = sample_rate
        self._needs_resample = False

        self._muted = False
        self._closing = True

        self._dropped_frames = 0
        self._last_drop_log = 0.0
        self._rms_dbfs = float("-inf")

        self._thread: threading.Thread | None = None
        self._last_frame_ts = time.monotonic()

    # -- public API ---------------------------------------------------

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def rms_dbfs(self) -> float:
        return self._rms_dbfs

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._closing = False
        self._thread = threading.Thread(
            target=self._supervisor_loop, name="AudioCapture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the supervisor thread to stop and wait for it to close the stream.

        The stream is only ever opened/closed on the supervisor thread itself
        (inside its own ``with sd.InputStream(...)`` block) -- closing it here
        from the caller's thread as well raced with that and corrupted
        PortAudio's internal state.
        """
        self._closing = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def clear_buffered_frames(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    async def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield frames as they arrive until :meth:`stop` is called."""
        self.start()
        loop = asyncio.get_running_loop()
        while not self._closing:
            try:
                frame = await loop.run_in_executor(None, self._queue.get, True, 0.5)
            except queue.Empty:
                continue
            yield frame

    # -- supervisor / reconnect logic ----------------------------------

    def _supervisor_loop(self) -> None:
        backoff_idx = 0
        disconnected_since: float | None = None
        device_lost_fired = False
        while not self._closing:
            try:
                self._open_and_run_stream()
                backoff_idx = 0
                disconnected_since = None
                device_lost_fired = False
            except Exception as exc:  # noqa: BLE001 - must never kill the process
                logger.warning("Audio capture stream error: {}", exc)
                if disconnected_since is None:
                    disconnected_since = time.monotonic()
                elapsed = time.monotonic() - disconnected_since
                if elapsed >= _DEVICE_LOST_AFTER_S and not device_lost_fired:
                    device_lost_fired = True
                    if self._on_device_lost:
                        self._on_device_lost(AudioDeviceLost(ts=time.time(), detail=str(exc)))
                delay = _BACKOFF_SCHEDULE_S[min(backoff_idx, len(_BACKOFF_SCHEDULE_S) - 1)]
                backoff_idx += 1
                self._sleep_unless_closing(delay)

    def _sleep_unless_closing(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._closing and time.monotonic() < deadline:
            time.sleep(0.1)

    def _open_and_run_stream(self) -> None:
        refresh_device_list()
        device_index = resolve_device(self._device_name, kind="input")
        self._capture_sr, self._needs_resample = self._probe_sample_rate(device_index)
        factor = self._capture_sr // self._sample_rate if self._needs_resample else 1
        capture_block = self._block_size * factor

        extra_settings = None
        if sys.platform == "win32":
            with contextlib.suppress(Exception):
                extra_settings = sd.WasapiSettings(exclusive=False)

        with sd.InputStream(
            samplerate=self._capture_sr,
            channels=self._channels,
            dtype="int16",
            blocksize=capture_block,
            device=device_index,
            callback=self._stream_callback,
            extra_settings=extra_settings,
        ) as stream:
            self._stream = stream
            self._last_frame_ts = time.monotonic()
            while not self._closing:
                time.sleep(_WATCHDOG_POLL_S)
                if not stream.active:
                    raise RuntimeError("input stream became inactive")
                if time.monotonic() - self._last_frame_ts > _WATCHDOG_SILENCE_S:
                    raise RuntimeError("no frames received from input stream (watchdog)")
        self._stream = None

    def _probe_sample_rate(self, device_index: int | None) -> tuple[int, bool]:
        try:
            sd.check_input_settings(
                device=device_index,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
            )
            return self._sample_rate, False
        except Exception:
            logger.info(
                "Input device does not support {} Hz directly, capturing at {} Hz and resampling",
                self._sample_rate,
                _FALLBACK_CAPTURE_SR,
            )
            return _FALLBACK_CAPTURE_SR, True

    # -- audio callback (runs on the PortAudio thread) -------------------

    def _stream_callback(self, indata: np.ndarray, frame_count: int, time_info, status) -> None:
        self._last_frame_ts = time.monotonic()
        if status:
            logger.debug("Audio capture status flags: {}", status)
        if self._muted:
            return

        pcm = indata[:, 0].copy()
        if self._needs_resample:
            factor = self._capture_sr // self._sample_rate
            pcm = np.clip(resample_poly(pcm, 1, factor), -32768, 32767).astype(np.int16)

        self._update_stats(pcm)
        frame = AudioFrame(pcm=pcm)
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
                self._dropped_frames += 1
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(frame)
            now = time.monotonic()
            if now - self._last_drop_log >= _DROP_LOG_INTERVAL_S:
                logger.warning(
                    "Audio capture queue full, dropping frames (total dropped={})",
                    self._dropped_frames,
                )
                self._last_drop_log = now

    def _update_stats(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        self._rms_dbfs = -math.inf if rms <= 0 else 20 * math.log10(rms / 32768.0)
