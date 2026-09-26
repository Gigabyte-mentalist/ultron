"""faster-whisper wrapper (docs/tz/03_stt.md §3.1).

Thread-safety: CTranslate2 models are not safe for concurrent calls from
multiple threads. A single ``threading.Lock`` serializes every
:meth:`WhisperEngine.transcribe` call, including the warmup call made from
:meth:`load`. Callers run this from asyncio via ``asyncio.to_thread``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from ultron.events import Transcript
from ultron.stt.filters import clean_transcript
from ultron.text.normalize import normalize_transcript

_SAMPLE_RATE = 16000
_WARMUP_SECONDS = 1.0


class WhisperEngine:
    """Wraps ``faster_whisper.WhisperModel`` behind the ``STTEngine`` protocol."""

    def __init__(
        self,
        *,
        model: str,
        device: str,
        compute_type: str,
        language: str,
        beam_size: int,
        no_speech_threshold: float,
        log_prob_threshold: float,
        initial_prompt: str | None,
        download_root: Path | str,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._no_speech_threshold = no_speech_threshold
        self._log_prob_threshold = log_prob_threshold
        self._default_initial_prompt = initial_prompt
        self._download_root = str(download_root)

        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model and run a 1 s silence warmup so the first real call is fast."""
        if self._model is not None:
            return
        logger.info(
            "STT modeli yuklanmoqda: {} (device={}, compute_type={})",
            self._model_name,
            self._device,
            self._compute_type,
        )
        Path(self._download_root).mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
            download_root=self._download_root,
        )
        warmup_pcm = np.zeros(int(_WARMUP_SECONDS * _SAMPLE_RATE), dtype=np.int16)
        self.transcribe(warmup_pcm)
        logger.info("STT modeli tayyor: {}", self._model_name)

    def transcribe(self, pcm: np.ndarray, *, initial_prompt: str | None = None) -> Transcript:
        if self._model is None:
            raise RuntimeError("WhisperEngine.load() birinchi chaqirilishi kerak")

        prompt = initial_prompt if initial_prompt is not None else self._default_initial_prompt
        duration_s = len(pcm) / _SAMPLE_RATE
        audio = pcm.astype(np.float32) / 32768.0

        start = time.monotonic()
        with self._lock:
            segments_iter, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=self._beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=prompt,
                without_timestamps=True,
                temperature=0.0,
                no_speech_threshold=self._no_speech_threshold,
                log_prob_threshold=self._log_prob_threshold,
            )
            segments = list(segments_iter)
        latency_ms = int((time.monotonic() - start) * 1000)

        raw_text = "".join(seg.text for seg in segments).strip()
        if segments:
            avg_logprob = sum(seg.avg_logprob for seg in segments) / len(segments)
            no_speech_prob = sum(seg.no_speech_prob for seg in segments) / len(segments)
            confidence = sum(np.exp(seg.avg_logprob) for seg in segments) / len(segments)
        else:
            avg_logprob = self._log_prob_threshold - 1.0
            no_speech_prob = 1.0
            confidence = 0.0

        cleaned = clean_transcript(
            raw_text,
            duration_s=duration_s,
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
            no_speech_threshold=self._no_speech_threshold,
            log_prob_threshold=self._log_prob_threshold,
        )
        text = normalize_transcript(cleaned)

        return Transcript(
            text=text,
            raw_text=raw_text,
            language=info.language,
            confidence=float(confidence),
            duration_s=duration_s,
            latency_ms=latency_ms,
        )
