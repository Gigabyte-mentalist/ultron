"""Wake-word detector factory (docs/tz/04_wake_word.md §2, §3.4).

``create_wake_detector`` builds a ``WakeDetector`` from config alone, so it
is usable standalone by ``scripts/wake_live.py``/``scripts/benchmark_wake.py``
without an orchestrator. It never calls ``.load()`` on the detector it
returns for the plain ``stt_kws`` path -- the caller loads it when ready --
except that it *must* probe ``oww``'s ``.load()`` itself to know whether to
fall back, per docs/tz/04_wake_word.md §3.3/§5 ("oww modeli yo'q bo'lganda
tizim yiqilmaydi, stt_kws'ga o'tadi").
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from ultron.audio.ring_buffer import RingBuffer
from ultron.config import RootConfig
from ultron.events import AudioFrame, Utterance, WakeEvent
from ultron.stt.whisper_engine import WhisperEngine
from ultron.wake.base import WakeDetector
from ultron.wake.matcher import WakeMatcher
from ultron.wake.oww_detector import OWWDetector, WakeModelMissing
from ultron.wake.stt_kws import STTKeywordSpotter

__all__ = [
    "WakeDetector",
    "WakeMatcher",
    "STTKeywordSpotter",
    "OWWDetector",
    "WakeModelMissing",
    "HybridDetector",
    "create_wake_detector",
]

_HYBRID_DEDUPE_WINDOW_S = 2.0
_KWS_BEAM_SIZE = 1
_KWS_INITIAL_PROMPT = "Altron."


class HybridDetector:
    """``oww`` (frame-level) + ``stt_kws`` (utterance-level) together (§3.4).

    Whichever confirms first wins; the other is suppressed for the same
    utterance within ``dedupe_window_s`` so a single "Altron" never produces
    two ``WakeEvent``s.
    """

    def __init__(
        self, *, oww: OWWDetector, stt_kws: STTKeywordSpotter, dedupe_window_s: float
    ) -> None:
        self._oww = oww
        self._stt_kws = stt_kws
        self._dedupe_window_s = dedupe_window_s
        self._last_event_ts: float | None = None

    def load(self) -> None:
        self._oww.load()
        self._stt_kws.load()

    def reset(self) -> None:
        self._oww.reset()
        self._stt_kws.reset()
        self._last_event_ts = None

    def set_speaking(self, speaking: bool) -> None:
        self._oww.set_speaking(speaking)
        self._stt_kws.set_speaking(speaking)

    def process_frame(self, frame: AudioFrame) -> WakeEvent | None:
        event = self._oww.process_frame(frame)
        return self._dedupe(event)

    async def process_utterance(self, utt: Utterance) -> WakeEvent | None:
        event = await self._stt_kws.process_utterance(utt)
        return self._dedupe(event)

    def _dedupe(self, event: WakeEvent | None) -> WakeEvent | None:
        if event is None:
            return None
        now = time.monotonic()
        if self._last_event_ts is not None and (now - self._last_event_ts) < self._dedupe_window_s:
            return None
        self._last_event_ts = now
        return event


def _build_matcher(config: RootConfig) -> WakeMatcher:
    return WakeMatcher(
        phrases=config.wake.phrases,
        fuzzy_threshold=config.wake.fuzzy_threshold,
        max_prefix_words=config.wake.max_prefix_words,
    )


def _build_kws_engine(config: RootConfig) -> WhisperEngine:
    download_root = Path(config.app.data_dir) / "models" / "stt"
    return WhisperEngine(
        model=config.stt.kws_model,
        device=config.stt.kws_device,
        compute_type=config.stt.compute_type,
        language=config.stt.language,
        beam_size=_KWS_BEAM_SIZE,
        no_speech_threshold=config.stt.no_speech_threshold,
        log_prob_threshold=config.stt.log_prob_threshold,
        initial_prompt=_KWS_INITIAL_PROMPT,
        download_root=download_root,
    )


def _build_main_engine(config: RootConfig) -> WhisperEngine:
    download_root = Path(config.app.data_dir) / "models" / "stt"
    return WhisperEngine(
        model=config.stt.model,
        device=config.stt.device,
        compute_type=config.stt.compute_type,
        language=config.stt.language,
        beam_size=config.stt.beam_size,
        no_speech_threshold=config.stt.no_speech_threshold,
        log_prob_threshold=config.stt.log_prob_threshold,
        initial_prompt=config.stt.initial_prompt,
        download_root=download_root,
    )


def _build_stt_kws(config: RootConfig, matcher: WakeMatcher) -> STTKeywordSpotter:
    main_engine = _build_main_engine(config)
    same_model_and_device = (
        config.stt.kws_model == config.stt.model
        and config.stt.kws_device == config.stt.device
        and config.stt.beam_size == _KWS_BEAM_SIZE
    )
    kws_engine = main_engine if same_model_and_device else _build_kws_engine(config)
    return STTKeywordSpotter(
        kws_engine=kws_engine,
        main_engine=main_engine,
        matcher=matcher,
        cooldown_ms=config.wake.cooldown_ms,
        barge_in_enabled=config.wake.barge_in_enabled,
    )


def _build_oww(
    config: RootConfig, matcher: WakeMatcher, *, kws_engine: WhisperEngine, ring_buffer: RingBuffer
) -> OWWDetector:
    return OWWDetector(
        model_path=Path(config.wake.oww_model_path),
        threshold=config.wake.oww_threshold,
        cooldown_ms=config.wake.cooldown_ms,
        verify_with_stt=config.wake.oww_verify_with_stt,
        barge_in_enabled=config.wake.barge_in_enabled,
        barge_in_threshold_boost=config.wake.barge_in_threshold_boost,
        kws_engine=kws_engine,
        matcher=matcher,
        ring_buffer=ring_buffer,
        verifier_path=config.wake.oww_verifier_path,
        verifier_threshold=config.wake.oww_verifier_threshold,
    )


def create_wake_detector(
    config: RootConfig, *, ring_buffer: RingBuffer | None = None
) -> WakeDetector:
    """Build the configured ``WakeDetector`` (``wake.mode``), falling back to
    ``stt_kws`` if ``oww``/``hybrid`` was requested but the model isn't ready.

    ``ring_buffer`` is only used by ``oww``/``hybrid`` (to pull the last 1.5 s
    for STT verification of a borderline trigger); pass the same
    ``RingBuffer`` the caller feeds from ``AudioCapture``.
    """
    matcher = _build_matcher(config)
    mode = config.wake.mode

    if mode == "stt_kws":
        return _build_stt_kws(config, matcher)

    if mode not in ("oww", "hybrid"):
        raise ValueError(f"Noma'lum wake.mode: {mode!r}")

    stt_kws = _build_stt_kws(config, matcher)
    try:
        oww = _build_oww(config, matcher, kws_engine=stt_kws.kws_engine, ring_buffer=ring_buffer)
        oww.load()
    except WakeModelMissing as exc:
        logger.warning(
            "openWakeWord ishga tushmadi, wake.mode='stt_kws'ga qaytildi. Sabab: {}", exc
        )
        return stt_kws

    if mode == "oww":
        return oww
    return HybridDetector(oww=oww, stt_kws=stt_kws, dedupe_window_s=_HYBRID_DEDUPE_WINDOW_S)
