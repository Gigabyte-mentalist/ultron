"""MMS-TTS (VITS) engine, CPU (docs/tz/05_tts.md §3.2).

Thread-safety mirrors ``stt/whisper_engine.py``: a single ``threading.Lock``
serializes every model forward pass (including the warmup and cache-priming
calls made from :meth:`load`), since only one synthesis is meant to run at a
time (docs/tz/05_tts.md §3.4).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, VitsModel

from ultron.audio.player import AudioPlayer
from ultron.i18n.uz import CACHED_PHRASES
from ultron.tts.preprocess import clean_text

_WARMUP_TEXT = "Salom."
_SEED = 42


def _has_speakable_content(text: str) -> bool:
    """False for "" and for punctuation-only leftovers (e.g. a message that
    was *only* a wake-word mention reduces to "." once ``clean_text`` strips
    it) -- feeding the VITS model a token sequence with no real phoneme
    content crashes deep inside its relative-position attention padding
    (``RuntimeError: narrow(): length must be non-negative``), found via
    docs/tz/08_orchestrator.md's own pipeline test."""
    return any(ch.isalnum() for ch in text)


# docs/tz/05_tts.md §3.2 suggests 4 (leaving the rest of the cores free for
# audio/VAD/STT). Measured on the target hardware (Ryzen 7 7435HS, 16
# logical cores) that undershoots the §5 acceptance budget (RTF <= 0.3):
# 4 threads averaged RTF=0.311 over the 15-sentence docs/tz/05_tts.md §3.1
# comparison set (data/reports/tts_compare/), 6 threads averaged RTF=0.257
# with the same margin of headroom the TZ intends (10 of 16 cores still
# free), so 6 is used instead -- see docs/PROGRESS.md for the measurements.
_TORCH_THREADS = 6


class MMSEngine:
    """Wraps ``transformers.VitsModel`` behind the ``TTSEngine`` protocol."""

    def __init__(
        self,
        *,
        model_name: str,
        model_script: str,
        cache_dir: Path | str,
        speed: float = 1.0,
        noise_scale: float | None = None,
        noise_scale_duration: float | None = None,
        sentence_pause_ms: int = 150,
    ) -> None:
        self._model_name = model_name
        self._model_script = model_script
        self._cache_dir = str(cache_dir)
        self._speed = speed
        self._noise_scale = noise_scale
        self._noise_scale_duration = noise_scale_duration
        self.sentence_pause_ms = sentence_pause_ms

        self.sample_rate = 16000
        self._model: VitsModel | None = None
        self._tokenizer = None
        self._phrase_cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._cancel_flag = threading.Event()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model, warm it up, and pre-synthesize every cached phrase."""
        if self._model is not None:
            return

        torch.set_num_threads(_TORCH_THREADS)
        Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info("TTS modeli yuklanmoqda: {}", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=self._cache_dir)
        self._model = VitsModel.from_pretrained(self._model_name, cache_dir=self._cache_dir)
        self._model.eval()
        self.sample_rate = int(self._model.config.sampling_rate)

        if self._noise_scale is not None:
            self._model.noise_scale = self._noise_scale
        if self._noise_scale_duration is not None:
            self._model.noise_scale_duration = self._noise_scale_duration

        if getattr(self._tokenizer, "is_uroman", False):
            logger.warning(
                "Model tokenizer'i is_uroman=True talab qiladi, lekin uroman "
                "romanizatsiyasi ulanmagan: {}",
                self._model_name,
            )

        self._synthesize_raw(clean_text(_WARMUP_TEXT, model_script=self._model_script))
        self._prime_phrase_cache()
        logger.info(
            "TTS tayyor: {} (sample_rate={}, {} ibora keshlandi)",
            self._model_name,
            self.sample_rate,
            len(self._phrase_cache),
        )

    def _prime_phrase_cache(self) -> None:
        for phrase in CACHED_PHRASES:
            cleaned = clean_text(phrase, model_script=self._model_script)
            if _has_speakable_content(cleaned) and cleaned not in self._phrase_cache:
                self._phrase_cache[cleaned] = self._synthesize_raw(cleaned)

    def synthesize(self, text: str) -> np.ndarray:
        cleaned = clean_text(text, model_script=self._model_script)
        cached = self._phrase_cache.get(cleaned)
        if cached is not None:
            return cached
        if not _has_speakable_content(cleaned):
            return np.zeros(0, dtype=np.float32)
        return self._synthesize_raw(cleaned)

    def _synthesize_raw(self, cleaned_text: str) -> np.ndarray:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MMSEngine.load() birinchi chaqirilishi kerak")

        with self._lock:
            inputs = self._tokenizer(cleaned_text, return_tensors="pt")
            torch.manual_seed(_SEED)
            with torch.inference_mode():
                output = self._model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    speaking_rate=self._speed,
                ).waveform
        return output.squeeze(0).to(torch.float32).numpy()

    def cancel(self) -> None:
        """Stop :meth:`speak_stream` from synthesizing any further queued sentences."""
        self._cancel_flag.set()

    def _append_pause(self, pcm: np.ndarray) -> np.ndarray:
        if self.sentence_pause_ms <= 0 or pcm.size == 0:
            return pcm
        pause_samples = int(self.sample_rate * self.sentence_pause_ms / 1000)
        return np.concatenate([pcm, np.zeros(pause_samples, dtype=np.float32)])

    async def speak_stream(self, sentences: AsyncIterator[str], player: AudioPlayer) -> None:
        """Synthesize each sentence and queue it on ``player`` as soon as it's ready.

        Synthesis of the next sentence begins right after the previous one
        is handed to ``player.enqueue`` (which returns immediately -- actual
        playback happens on a separate thread), so synthesis of sentence
        N+1 overlaps with playback of sentence N (docs/tz/05_tts.md §3.4).
        """
        self._cancel_flag.clear()
        async for sentence in sentences:
            if self._cancel_flag.is_set():
                break
            pcm = await asyncio.to_thread(self.synthesize, sentence)
            if self._cancel_flag.is_set() or pcm.size == 0:
                continue
            await player.play(self._append_pause(pcm), self.sample_rate)
