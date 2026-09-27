"""STT-based keyword-spotting wake detector (docs/tz/04_wake_word.md §3.2).

Runs on every VAD-endpointed ``Utterance`` while the orchestrator is idle: a
quick pass transcribes only the first ~1.5 s with a light KWS model to check
for the wake word, and -- only when the segment is long enough to plausibly
contain a command too -- a second pass re-transcribes the *whole* segment
with the main (more accurate) STT model to recover ``trailing_text``.

Transcription is a blocking GPU call (~150-250 ms). ``process_utterance`` is
``async`` and backed by a small bounded queue (docs/tz/04_wake_word.md §3.2:
"maksimum 3 ta, eskilari tashlanadi") so that utterances arriving faster than
the KWS model can keep up don't pile up unboundedly -- the oldest pending one
is dropped (resolved to ``None``) to make room for the newest.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque

import numpy as np
from loguru import logger

from ultron.events import AudioFrame, Utterance, WakeEvent
from ultron.stt.base import STTEngine
from ultron.wake.matcher import WakeMatcher

_SAMPLE_RATE = 16000
_MIN_SEGMENT_S = 0.3
_MAX_SEGMENT_S = 15.0
_KWS_HEAD_SECONDS = 1.5
_LONG_SEGMENT_S = 1.8
_KWS_INITIAL_PROMPT = "Altron."
_MAX_QUEUE = 3

# RMS gates below which a segment is skipped without spending a GPU call on
# it (docs/tz/04_wake_word.md §3.2/§3.5). Fixed implementation constants
# rather than config keys, same rationale as audio/vad.py's pre-roll/margin:
# the TZ states them as fixed values, not schema entries.
_RMS_GATE_DBFS = -50.0
_BARGE_IN_RMS_GATE_DBFS = -35.0


def _rms_dbfs(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    return -math.inf if rms <= 0 else 20 * math.log10(rms / 32768.0)


class STTKeywordSpotter:
    """Implements ``WakeDetector`` using two STT engines (see module docstring)."""

    def __init__(
        self,
        *,
        kws_engine: STTEngine,
        main_engine: STTEngine,
        matcher: WakeMatcher,
        cooldown_ms: int,
        barge_in_enabled: bool = True,
        max_queue: int = _MAX_QUEUE,
    ) -> None:
        self._kws_engine = kws_engine
        self._main_engine = main_engine
        self._matcher = matcher
        self._cooldown_s = cooldown_ms / 1000.0
        self._barge_in_enabled = barge_in_enabled
        self._max_queue = max_queue

        self._pending: deque[tuple[Utterance, asyncio.Future]] = deque()
        self._worker: asyncio.Task | None = None
        self._cooldown_until: float | None = None
        self._paused = False
        self._speaking = False
        self._loaded = False

    @property
    def kws_engine(self) -> STTEngine:
        return self._kws_engine

    @property
    def main_engine(self) -> STTEngine:
        return self._main_engine

    def load(self) -> None:
        if self._loaded:
            return
        self._kws_engine.load()
        self._main_engine.load()
        self._loaded = True

    def reset(self) -> None:
        for _, future in self._pending:
            if not future.done():
                future.set_result(None)
        self._pending.clear()
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        self._cooldown_until = None

    def set_speaking(self, speaking: bool) -> None:
        if not self._barge_in_enabled:
            self._paused = speaking
            return
        self._speaking = speaking

    def process_frame(self, frame: AudioFrame) -> WakeEvent | None:  # stt_kws is utterance-only
        return None

    async def process_utterance(self, utt: Utterance) -> WakeEvent | None:
        if self._paused:
            return None

        duration_s = len(utt.pcm) / _SAMPLE_RATE
        if not (_MIN_SEGMENT_S <= duration_s <= _MAX_SEGMENT_S):
            return None

        gate = _BARGE_IN_RMS_GATE_DBFS if self._speaking else _RMS_GATE_DBFS
        if _rms_dbfs(utt.pcm) < gate:
            return None

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending.append((utt, future))
        while len(self._pending) > self._max_queue:
            _, dropped = self._pending.popleft()
            if not dropped.done():
                dropped.set_result(None)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.ensure_future(self._drain())
        return await future

    async def _drain(self) -> None:
        while self._pending:
            utt, future = self._pending.popleft()
            if future.cancelled():
                continue
            try:
                result = await self._process_one(utt)
            except Exception as exc:  # noqa: BLE001 - one bad segment must not kill the worker
                logger.warning("Wake KWS ishlov berishda xato: {}", exc)
                result = None
            if not future.done():
                future.set_result(result)

    async def _process_one(self, utt: Utterance) -> WakeEvent | None:
        now = time.monotonic()
        if self._cooldown_until is not None and now < self._cooldown_until:
            return None

        head_samples = int(_KWS_HEAD_SECONDS * _SAMPLE_RATE)
        head_pcm = utt.pcm[:head_samples]
        transcript = await asyncio.to_thread(
            self._kws_engine.transcribe, head_pcm, initial_prompt=_KWS_INITIAL_PROMPT
        )
        match = self._matcher.match(transcript.text.lower())
        if not match.matched:
            return None

        duration_s = len(utt.pcm) / _SAMPLE_RATE
        trailing_text = None
        if duration_s > _LONG_SEGMENT_S:
            full_transcript = await asyncio.to_thread(self._main_engine.transcribe, utt.pcm)
            full_match = self._matcher.match(full_transcript.text.lower())
            if full_match.matched:
                trailing_text = full_match.trailing_text

        self._cooldown_until = time.monotonic() + self._cooldown_s
        return WakeEvent(
            ts=time.time(), score=match.score / 100.0, source="stt_kws", trailing_text=trailing_text
        )
