"""Silero VAD (ONNX, CPU) and endpointing state machine (docs/tz/02_audio.md §3.3).

``SileroVAD`` wraps the bundled Silero ONNX graph directly through
onnxruntime (rather than the ``silero_vad`` pip package's own wrapper,
which unconditionally imports torch) so the always-on audio layer stays
light. ``Endpointer`` turns the resulting per-frame speech probabilities
into ``Utterance`` boundaries.

Two numbers below are fixed implementation constants rather than config
keys, because the TZ states them as fixed values rather than listing them
under ``vad:`` in the config schema: the 300 ms pre-roll and the 0.15
hysteresis margin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ultron.audio.ring_buffer import RingBuffer
from ultron.events import Utterance

_MODEL_SAMPLE_RATE = 16000
_MODEL_CHUNK_SAMPLES = 512
_MODEL_CONTEXT_SAMPLES = 64
_STATE_SHAPE = (2, 1, 128)

_DEFAULT_PREROLL_MS = 300.0
_DEFAULT_HYSTERESIS_MARGIN = 0.15

_MODEL_RELATIVE_PATH = Path("models") / "vad" / "silero_vad.onnx"


def default_model_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / _MODEL_RELATIVE_PATH


class SileroVAD:
    """Call :meth:`process` once per 512-sample (32 ms @ 16 kHz) int16 frame."""

    def __init__(self, model_path: Path) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}; run: "
                "uv run python scripts/download_models.py --vad"
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _MODEL_CONTEXT_SAMPLES), dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _MODEL_CONTEXT_SAMPLES), dtype=np.float32)

    def process(self, pcm: np.ndarray) -> float:
        """Return the speech probability (0..1) for one 512-sample int16 frame."""
        if pcm.shape[-1] != _MODEL_CHUNK_SAMPLES:
            raise ValueError(f"Expected {_MODEL_CHUNK_SAMPLES} samples, got {pcm.shape[-1]}")
        chunk = (pcm.astype(np.float32) / 32768.0).reshape(1, _MODEL_CHUNK_SAMPLES)
        x = np.concatenate([self._context, chunk], axis=1)
        out, state = self._session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": np.array(_MODEL_SAMPLE_RATE, dtype=np.int64),
            },
        )
        self._state = state
        self._context = x[:, -_MODEL_CONTEXT_SAMPLES:]
        return float(out[0, 0])


class _State(Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    TRAILING_SILENCE = "trailing_silence"


@dataclass
class ListenTimeout:
    ts: float


class Endpointer:
    """Turns a stream of (pcm frame, VAD probability) pairs into Utterance boundaries.

    The caller must write each frame into ``ring_buffer`` before calling
    :meth:`push` with that same frame, so the pre-roll snapshot taken at
    speech onset ends exactly at the confirming frame.
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        *,
        sample_rate: int,
        threshold: float,
        min_speech_ms: int,
        end_silence_ms: int,
        max_utterance_s: float,
        listen_timeout_s: float,
        preroll_ms: float = _DEFAULT_PREROLL_MS,
        hysteresis_margin: float = _DEFAULT_HYSTERESIS_MARGIN,
    ) -> None:
        self._ring_buffer = ring_buffer
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._end_silence_ms = end_silence_ms
        self._max_utterance_s = max_utterance_s
        self._listen_timeout_s = listen_timeout_s
        self._preroll_ms = preroll_ms
        self._hysteresis_margin = hysteresis_margin

        self._mode = "passive"
        self._state = _State.SILENCE
        self._speech_run_ms = 0.0
        self._silence_run_ms = 0.0
        self._since_listen_start_ms = 0.0
        self._timeout_fired = False
        self._chunks: list[np.ndarray] = []
        self._started_at = 0.0

    def set_mode(self, mode: str) -> None:
        if mode not in ("passive", "active"):
            raise ValueError(f"Unknown endpointer mode: {mode!r}")
        self._mode = mode
        self._since_listen_start_ms = 0.0
        self._timeout_fired = False
        if mode == "active":
            self._reset_segment()

    def set_end_silence_ms(self, value: int) -> None:
        """Allow runtime tuning, e.g. a shorter window for yes/no confirmations."""
        self._end_silence_ms = value

    def reset(self) -> None:
        self._reset_segment()
        self._since_listen_start_ms = 0.0
        self._timeout_fired = False

    def push(self, pcm: np.ndarray, prob: float) -> Utterance | ListenTimeout | None:
        frame_ms = 1000.0 * len(pcm) / self._sample_rate
        hysteresis_threshold = self._threshold - self._hysteresis_margin

        if self._state == _State.SILENCE:
            return self._push_silence(pcm, prob, frame_ms)

        self._chunks.append(pcm)
        if self._segment_duration_ms() >= self._max_utterance_s * 1000:
            return self._emit("max_length")

        if self._state == _State.SPEECH:
            if prob < hysteresis_threshold:
                self._state = _State.TRAILING_SILENCE
                self._silence_run_ms = frame_ms
            return None

        # TRAILING_SILENCE
        if prob >= self._threshold:
            self._state = _State.SPEECH
            self._silence_run_ms = 0.0
            return None
        if prob < hysteresis_threshold:
            self._silence_run_ms += frame_ms
            if self._silence_run_ms >= self._end_silence_ms:
                return self._emit("silence")
        return None

    def _push_silence(self, pcm: np.ndarray, prob: float, frame_ms: float) -> ListenTimeout | None:
        if prob > self._threshold:
            self._speech_run_ms += frame_ms
        else:
            self._speech_run_ms = 0.0

        if self._speech_run_ms >= self._min_speech_ms:
            self._begin_speech(pcm)
            return None

        if self._mode == "active" and not self._timeout_fired:
            self._since_listen_start_ms += frame_ms
            if self._since_listen_start_ms >= self._listen_timeout_s * 1000:
                self._timeout_fired = True
                return ListenTimeout(ts=time.time())
        return None

    def _begin_speech(self, pcm: np.ndarray) -> None:
        preroll = self._ring_buffer.snapshot(self._preroll_ms / 1000.0)
        self._chunks = [preroll] if preroll.size else [pcm.copy()]
        self._state = _State.SPEECH
        self._speech_run_ms = 0.0
        self._silence_run_ms = 0.0
        self._started_at = time.time()

    def _segment_duration_ms(self) -> float:
        total_samples = sum(len(c) for c in self._chunks)
        return 1000.0 * total_samples / self._sample_rate

    def _emit(self, reason: str) -> Utterance:
        pcm = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.int16)
        utt = Utterance(pcm=pcm, started_at=self._started_at, ended_at=time.time(), reason=reason)
        self._reset_segment()
        return utt

    def _reset_segment(self) -> None:
        self._state = _State.SILENCE
        self._chunks = []
        self._speech_run_ms = 0.0
        self._silence_run_ms = 0.0
