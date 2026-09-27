"""End-to-end offline pipeline test (docs/tz/08_orchestrator.md §2).

Requires the real STT (GPU), wake, and TTS (CPU) models -- not part of the
default ``pytest -m "not hardware"`` run. Rather than depending on
pre-recorded human speech (this project's own real recordings under
``data/recordings/`` are the user's personal voice and don't belong in a
git-tracked test fixture), this test *synthesizes* its own prompts with the
already-loaded MMS-TTS engine and feeds that audio into the real wake
detector + STT, exactly as ``core.orchestrator`` would receive an
Endpointer-isolated ``Utterance``. ``ULTRON_TOOLS_MOCK=1`` (docs/tz/07_tools.md
§1) keeps every tool call from touching the real system; the fastpath-covered
commands used below never reach the LLM, so no live Ollama server is needed.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from ultron.config import load_config
from ultron.events import Utterance
from ultron.llm.fastpath import FastPath
from ultron.text.numbers_uz import convert_numbers_in_text
from ultron.text.translit import latin_to_cyrillic
from ultron.tools import configure_all, default_registry
from ultron.tts.mms_engine import MMSEngine
from ultron.tts.preprocess import (  # noqa: PLC2701
    _clean_markup,
    _expand_abbreviations,
    _tidy_whitespace,
)
from ultron.wake import create_wake_detector

pytestmark = pytest.mark.hardware

_SR = 16000


@pytest.fixture(autouse=True)
def _mock_tool_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """``monkeypatch.setenv`` (unlike a bare module-level ``os.environ[...]
    = ...``) is undone after this test, and only takes effect for tests
    that actually run -- a module-level assignment would execute at import
    time even when ``-m "not hardware"`` deselects every test in this file,
    leaking mock mode into whatever other test file pytest collects next.
    """
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")


def _to_pipeline_pcm(float_pcm: np.ndarray, source_sr: int) -> np.ndarray:
    """Synthesized float32 [-1, 1] @ ``source_sr`` -> int16 mono @ 16 kHz,
    the format every stage from the Endpointer onward expects."""
    if source_sr != _SR:
        g = gcd(source_sr, _SR)
        float_pcm = resample_poly(float_pcm, _SR // g, source_sr // g)
    return np.clip(float_pcm * 32767.0, -32768, 32767).astype(np.int16)


def _synthesized_utterance(tts: MMSEngine, text: str) -> Utterance:
    pcm_float = tts.synthesize(text)
    assert pcm_float.size > 0, f"TTS produced no audio for {text!r}"
    pcm = _to_pipeline_pcm(pcm_float, tts.sample_rate)
    return Utterance(pcm=pcm, started_at=0.0, ended_at=0.0, reason="silence")


def _synthesized_utterance_with_wake_word(tts: MMSEngine, text: str) -> Utterance:
    """Like ``_synthesized_utterance``, but for a prompt that must actually
    *contain* "Altron" in the rendered audio.

    ``MMSEngine.synthesize()`` (correctly) always strips any wake-word
    mention before synthesis -- that is the self-trigger guard
    (docs/tz/00_umumiy_arxitektura.md §7 point 6) -- so it can never be used
    to produce audio containing "Altron" at all. This test-only helper
    mirrors ``tts.preprocess.clean_text``'s other steps exactly, skipping
    only the wake-word-stripping one, and calls ``MMSEngine``'s private
    ``_synthesize_raw`` directly (bypassing the public ``synthesize()``,
    which would strip the word again). There's no other way to get real,
    GPU-model-driven audio containing the wake word without a pre-recorded
    human voice sample -- see this module's docstring for why that's
    deliberately avoided.
    """
    cleaned = _clean_markup(text)
    cleaned = convert_numbers_in_text(cleaned)
    cleaned = _expand_abbreviations(cleaned)
    cleaned = _tidy_whitespace(cleaned)
    if tts._model_script == "cyrillic":  # noqa: SLF001
        cleaned = latin_to_cyrillic(cleaned)
    pcm_float = tts._synthesize_raw(cleaned)  # noqa: SLF001
    assert pcm_float.size > 0, f"TTS produced no audio for {text!r} (cleaned: {cleaned!r})"
    pcm = _to_pipeline_pcm(pcm_float, tts.sample_rate)
    return Utterance(pcm=pcm, started_at=0.0, ended_at=0.0, reason="silence")


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def tts(config):
    engine = MMSEngine(
        model_name=config.tts.model,
        model_script=config.tts.model_script,
        cache_dir=Path(config.app.data_dir) / "models" / "tts",
        speed=config.tts.speed,
        sentence_pause_ms=config.tts.sentence_pause_ms,
    )
    engine.load()
    return engine


@pytest.fixture(scope="module")
def wake(config):
    detector = create_wake_detector(config)
    detector.load()
    return detector


@pytest.fixture(autouse=True)
def _reset_wake_between_tests(wake):
    # ``wake`` is module-scoped (loading it is expensive); reset it before
    # each test so one test's successful detection doesn't leave the next
    # one inside ``wake.cooldown_ms`` and silently swallow its WakeEvent.
    wake.reset()
    yield


@pytest.fixture(scope="module")
def registry(config):
    configure_all(config)
    return default_registry


async def test_synthesized_wake_word_alone_is_recognized_without_trailing_text(tts, wake) -> None:
    utt = _synthesized_utterance_with_wake_word(tts, "Altron.")

    event = await wake.process_utterance(utt)

    assert event is not None
    assert event.source == "stt_kws"


async def test_synthesized_wake_word_with_command_extracts_trailing_text(tts, wake) -> None:
    utt = _synthesized_utterance_with_wake_word(tts, "Altron, soat nechi?")

    event = await wake.process_utterance(utt)

    assert event is not None
    assert event.trailing_text
    assert "soat" in event.trailing_text.lower()


async def test_recognized_trailing_text_routes_through_fastpath_to_a_mocked_tool(
    tts, wake, registry
) -> None:
    utt = _synthesized_utterance_with_wake_word(tts, "Altron, soat nechi?")
    event = await wake.process_utterance(utt)
    assert event is not None and event.trailing_text

    fastpath = FastPath(registry)
    match = fastpath.match(event.trailing_text)
    assert match is not None
    assert match.tool_name == "get_time"

    result = await registry.call(match.tool_name, match.args)

    assert result.ok
    assert result.speak


async def test_plain_speech_without_wake_word_is_not_recognized_as_wake(tts, wake) -> None:
    utt = _synthesized_utterance(tts, "Bugun ob-havo juda yaxshi.")

    event = await wake.process_utterance(utt)

    assert event is None
