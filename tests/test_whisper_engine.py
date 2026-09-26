"""Hardware test for ultron.stt.whisper_engine.WhisperEngine (docs/tz/03_stt.md §4).

Requires a CUDA GPU and downloads the configured STT model from Hugging Face
on first run. Not part of the default ``pytest -m "not hardware"`` run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from ultron.config import load_config
from ultron.stt.whisper_engine import WhisperEngine

_SPEECH_FIXTURE = Path(__file__).parent / "data" / "speech_uz.wav"
_SR = 16000

pytestmark = pytest.mark.hardware


@pytest.fixture(scope="module")
def engine() -> WhisperEngine:
    config = load_config()
    eng = WhisperEngine(
        model=config.stt.model,
        device=config.stt.device,
        compute_type=config.stt.compute_type,
        language=config.stt.language,
        beam_size=config.stt.beam_size,
        no_speech_threshold=config.stt.no_speech_threshold,
        log_prob_threshold=config.stt.log_prob_threshold,
        initial_prompt=config.stt.initial_prompt,
        download_root=Path(config.app.data_dir) / "models" / "stt",
    )
    eng.load()
    return eng


def test_transcribes_non_empty_text(engine: WhisperEngine) -> None:
    pcm, sr = sf.read(_SPEECH_FIXTURE, dtype="int16")
    assert sr == _SR

    transcript = engine.transcribe(pcm)

    assert transcript.text != ""
    assert transcript.language == "uz"


def test_latency_under_one_second(engine: WhisperEngine) -> None:
    pcm, _ = sf.read(_SPEECH_FIXTURE, dtype="int16")

    transcript = engine.transcribe(pcm)

    assert transcript.latency_ms < 1000


def test_short_silence_is_filtered_to_empty(engine: WhisperEngine) -> None:
    import numpy as np

    silence = np.zeros(int(0.2 * _SR), dtype=np.int16)  # below the 0.3 s minimum

    transcript = engine.transcribe(silence)

    assert transcript.text == ""
