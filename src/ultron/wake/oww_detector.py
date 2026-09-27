"""openWakeWord frame-level wake detector (docs/tz/04_wake_word.md §3.3).

Interface-level for this stage: works once a trained model file exists at
``wake.oww_model_path``, but no such model is trained yet (that is
docs/tz/10_wake_word_training.md). Until then -- and if the shared
melspectrogram/embedding feature models or the ``openwakeword`` package
itself are missing -- :meth:`load` raises :class:`WakeModelMissing` with a
message naming the exact fix, and ``wake/__init__.py``'s factory catches it
and falls back to ``stt_kws``.
"""

from __future__ import annotations

import time
from pathlib import Path

from ultron.audio.ring_buffer import RingBuffer
from ultron.events import AudioFrame, Utterance, WakeEvent
from ultron.stt.base import STTEngine
from ultron.wake.matcher import WakeMatcher

_KWS_INITIAL_PROMPT = "Altron."
_VERIFY_SECONDS = 1.5
_HIGH_CONFIDENCE_SCORE = 0.85


class WakeModelMissing(RuntimeError):
    """Raised by :meth:`OWWDetector.load` when it cannot become usable.

    Callers (``wake/__init__.py``'s ``create_wake_detector``) catch this and
    fall back to ``stt_kws`` rather than letting the process crash.
    """


class OWWDetector:
    """Implements ``WakeDetector`` on top of ``openwakeword.model.Model`` (CPU, ONNX)."""

    def __init__(
        self,
        *,
        model_path: Path,
        threshold: float,
        cooldown_ms: int,
        verify_with_stt: bool,
        barge_in_enabled: bool = True,
        barge_in_threshold_boost: float = 0.0,
        kws_engine: STTEngine | None = None,
        matcher: WakeMatcher | None = None,
        ring_buffer: RingBuffer | None = None,
        verifier_path: Path | None = None,
        verifier_threshold: float = 0.3,
    ) -> None:
        self._model_path = Path(model_path)
        self._threshold = threshold
        self._cooldown_s = cooldown_ms / 1000.0
        self._verify_with_stt = verify_with_stt
        self._barge_in_enabled = barge_in_enabled
        self._barge_in_boost = barge_in_threshold_boost
        self._kws_engine = kws_engine
        self._matcher = matcher
        self._ring_buffer = ring_buffer
        self._verifier_path = Path(verifier_path) if verifier_path is not None else None
        self._verifier_threshold = verifier_threshold

        self._model = None
        self._model_key = self._model_path.stem
        self._cooldown_until: float | None = None
        self._paused = False
        self._speaking = False

    def load(self) -> None:
        if self._model is not None:
            return

        if not self._model_path.is_file():
            raise WakeModelMissing(
                f"openWakeWord modeli topilmadi: {self._model_path}\n"
                "Bu model docs/tz/10_wake_word_training.md bosqichida o'qitiladi. "
                "Hozircha wake.mode='stt_kws' ishlatiladi."
            )

        feature_dir = self._model_path.parent
        melspec_path = feature_dir / "melspectrogram.onnx"
        embedding_path = feature_dir / "embedding_model.onnx"
        if not melspec_path.is_file() or not embedding_path.is_file():
            raise WakeModelMissing(
                f"openWakeWord umumiy feature modellari topilmadi ({feature_dir}). "
                "Ishga tushirish: uv run python scripts/download_models.py --oww"
            )

        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise WakeModelMissing(
                "openwakeword kutubxonasi o'rnatilmagan (`uv add openwakeword`)"
            ) from exc

        # docs/tz/10_wake_word_training.md §3.4: shaxsiy verifier ixtiyoriy --
        # fayl mavjud bo'lmasa yoki config'da o'chirilgan bo'lsa, oddiy oww bilan davom etiladi.
        verifier_kwargs: dict[str, object] = {}
        if self._verifier_path is not None and self._verifier_path.is_file():
            verifier_kwargs["custom_verifier_models"] = {self._model_key: str(self._verifier_path)}
            verifier_kwargs["custom_verifier_threshold"] = self._verifier_threshold

        try:
            self._model = Model(
                wakeword_models=[str(self._model_path)],
                inference_framework="onnx",
                melspec_model_path=str(melspec_path),
                embedding_model_path=str(embedding_path),
                vad_threshold=0,
                **verifier_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - any load failure must fall back, not crash
            raise WakeModelMissing(f"openWakeWord modelini yuklab bo'lmadi: {exc}") from exc

    def reset(self) -> None:
        self._cooldown_until = None
        if self._model is not None:
            for buf in self._model.prediction_buffer.values():
                buf.clear()

    def set_speaking(self, speaking: bool) -> None:
        if not self._barge_in_enabled:
            self._paused = speaking
            return
        self._speaking = speaking

    def process_frame(self, frame: AudioFrame) -> WakeEvent | None:
        if self._model is None or self._paused:
            return None

        now = time.monotonic()
        if self._cooldown_until is not None and now < self._cooldown_until:
            return None

        predictions = self._model.predict(frame.pcm)
        score = float(predictions.get(self._model_key, 0.0))
        threshold = self._threshold + (self._barge_in_boost if self._speaking else 0.0)
        if score < threshold:
            return None
        return self._handle_trigger(score)

    async def process_utterance(self, utt: Utterance) -> WakeEvent | None:  # oww is frame-only
        return None

    def _handle_trigger(self, score: float) -> WakeEvent | None:
        self._cooldown_until = time.monotonic() + self._cooldown_s

        if score >= _HIGH_CONFIDENCE_SCORE or not self._verify_with_stt:
            return WakeEvent(ts=time.time(), score=score, source="oww", trailing_text=None)

        if self._ring_buffer is None or self._kws_engine is None:
            # No verifier wired up: accept rather than silently drop a real trigger.
            return WakeEvent(ts=time.time(), score=score, source="oww", trailing_text=None)

        snapshot = self._ring_buffer.snapshot(_VERIFY_SECONDS)
        if snapshot.size == 0:
            return None
        transcript = self._kws_engine.transcribe(snapshot, initial_prompt=_KWS_INITIAL_PROMPT)
        match = self._matcher.match(transcript.text.lower()) if self._matcher else None
        if match is None or not match.matched:
            return None
        return WakeEvent(
            ts=time.time(), score=score, source="oww", trailing_text=match.trailing_text
        )
