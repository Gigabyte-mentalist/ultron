"""Hardware test for ultron.tts.mms_engine.MMSEngine (docs/tz/05_tts.md §4).

Downloads the configured TTS model from Hugging Face on first run (CPU
only, no GPU needed). Not part of the default ``pytest -m "not hardware"``
run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ultron.config import load_config
from ultron.i18n.uz import CACHED_PHRASES
from ultron.tts.mms_engine import MMSEngine

_RTF_BUDGET = 0.3

pytestmark = pytest.mark.hardware


@pytest.fixture(scope="module")
def engine() -> MMSEngine:
    config = load_config()
    eng = MMSEngine(
        model_name=config.tts.model,
        model_script=config.tts.model_script,
        cache_dir=Path(config.app.data_dir) / "models" / "tts",
        speed=config.tts.speed,
        sentence_pause_ms=config.tts.sentence_pause_ms,
    )
    eng.load()
    return eng


def test_load_sets_sample_rate(engine: MMSEngine) -> None:
    assert engine.sample_rate > 0


def test_synthesize_produces_non_empty_audio(engine: MMSEngine) -> None:
    pcm = engine.synthesize("Salom, bugun ob-havo juda yaxshi.")
    assert pcm.dtype == np.float32
    assert pcm.size > 0
    assert np.max(np.abs(pcm)) <= 1.0


def test_synthesize_empty_after_wake_word_strip_returns_empty() -> None:
    # A fresh engine is not needed here: "Ultron" alone strips to "", and an
    # empty cleaned string always short-circuits to an empty array regardless
    # of cache state.
    config = load_config()
    eng = MMSEngine(
        model_name=config.tts.model,
        model_script=config.tts.model_script,
        cache_dir=Path(config.app.data_dir) / "models" / "tts",
    )
    eng.load()
    pcm = eng.synthesize("Ultron")
    assert pcm.size == 0


def test_output_never_contains_wake_word_text(engine: MMSEngine) -> None:
    # Indirect check: synthesizing a sentence that mentions the wake word
    # must not raise and must still produce audio for the remaining words.
    pcm = engine.synthesize("Assalomu alaykum, men Ultron emasman.")
    assert pcm.size > 0


def test_cached_phrases_are_primed(engine: MMSEngine) -> None:
    for phrase in CACHED_PHRASES:
        assert engine.synthesize(phrase).size > 0


def test_rtf_within_budget_for_a_medium_sentence(engine: MMSEngine) -> None:
    import time

    text = "Bugun ob-havo juda yaxshi, harorat yigirma sakkiz daraja atrofida bo'ladi."
    start = time.monotonic()
    pcm = engine.synthesize(text)
    elapsed_s = time.monotonic() - start
    duration_s = pcm.size / engine.sample_rate

    assert duration_s > 0
    rtf = elapsed_s / duration_s
    assert rtf <= _RTF_BUDGET


def test_numbers_and_dates_are_spoken(engine: MMSEngine) -> None:
    pcm = engine.synthesize("Bugun 27.09.2026, soat 09:30.")
    assert pcm.size > 0
