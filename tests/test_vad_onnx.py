"""Tests for ultron.audio.vad.SileroVAD against the bundled ONNX model
(docs/tz/02_audio.md §4).

Not a @hardware test: runs entirely on CPU, no microphone needed. Skips if
the model hasn't been installed yet (uv run python scripts/download_models.py --vad).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ultron.audio.vad import SileroVAD, default_model_path
from ultron.config import load_config

_BLOCK = 512
_SR = 16000
_SPEECH_FIXTURE = Path(__file__).parent / "data" / "speech_uz.wav"
_INSTALL_HINT = "scripts/download_models.py --vad"


def _model_path() -> Path:
    config = load_config()
    path = default_model_path(config.app.data_dir)
    if not path.is_file():
        pytest.skip(f"Silero VAD model not installed at {path}; run: {_INSTALL_HINT}")
    return path


def test_one_second_of_silence_scores_low() -> None:
    vad = SileroVAD(_model_path())
    silence = np.zeros(_BLOCK, dtype=np.int16)

    probs = [vad.process(silence) for _ in range(30)]  # ~0.96 s

    assert max(probs) < 0.2


def test_speech_fixture_scores_high() -> None:
    vad = SileroVAD(_model_path())
    pcm, sr = sf.read(_SPEECH_FIXTURE, dtype="int16")
    assert sr == _SR

    probs = [vad.process(pcm[i : i + _BLOCK]) for i in range(0, len(pcm) - _BLOCK + 1, _BLOCK)]

    assert max(probs) > 0.5


def test_reset_clears_internal_state() -> None:
    vad = SileroVAD(_model_path())
    pcm, _ = sf.read(_SPEECH_FIXTURE, dtype="int16")
    for i in range(0, 5 * _BLOCK, _BLOCK):
        vad.process(pcm[i : i + _BLOCK])

    vad.reset()
    prob_after_reset = vad.process(np.zeros(_BLOCK, dtype=np.int16))

    assert prob_after_reset < 0.2
