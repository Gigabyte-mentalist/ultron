"""Tests for ultron.wake.stt_kws.STTKeywordSpotter (docs/tz/04_wake_word.md §4).

Uses small stub STT engines (no real model, no GPU) so this stays a fast,
non-hardware unit test -- only the queueing/matching/gating *logic* is
exercised here.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from ultron.events import Transcript, Utterance
from ultron.wake.matcher import WakeMatcher
from ultron.wake.stt_kws import STTKeywordSpotter

_SR = 16000


class StubEngine:
    """Returns a fixed transcript for every call; records every ``pcm`` seen."""

    def __init__(
        self,
        text: str = "",
        *,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.text = text
        self.calls: list[np.ndarray] = []
        self._started = started
        self._release = release
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, pcm: np.ndarray, *, initial_prompt: str | None = None) -> Transcript:
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            self._release.wait(timeout=2.0)
        self.calls.append(pcm)
        return Transcript(
            text=self.text,
            raw_text=self.text,
            language="uz",
            confidence=1.0,
            duration_s=len(pcm) / _SR,
            latency_ms=1,
        )


def _matcher(fuzzy_threshold: int = 82) -> WakeMatcher:
    return WakeMatcher(
        phrases=["altron", "ultron"], fuzzy_threshold=fuzzy_threshold, max_prefix_words=2
    )


def _utt(seconds: float, *, amplitude: int = 5000, label: int = 0) -> Utterance:
    n = int(seconds * _SR)
    pcm = np.full(n, amplitude + label, dtype=np.int16)
    return Utterance(pcm=pcm, started_at=0.0, ended_at=seconds, reason="silence")


def _spotter(
    kws_text: str, main_text: str = "", **overrides
) -> tuple[STTKeywordSpotter, StubEngine, StubEngine]:
    kws = StubEngine(kws_text)
    main = StubEngine(main_text)
    spotter = STTKeywordSpotter(
        kws_engine=kws,
        main_engine=main,
        matcher=_matcher(),
        cooldown_ms=overrides.pop("cooldown_ms", 0),
        **overrides,
    )
    return spotter, kws, main


async def test_short_segment_with_wake_word_has_no_trailing_text() -> None:
    spotter, kws, main = _spotter(kws_text="altron")

    event = await spotter.process_utterance(_utt(1.0))

    assert event is not None
    assert event.source == "stt_kws"
    assert event.trailing_text is None
    assert len(kws.calls) == 1
    assert len(main.calls) == 0  # short segment: no re-transcription


async def test_long_segment_triggers_main_engine_retranscription() -> None:
    spotter, kws, main = _spotter(kws_text="altron", main_text="altron brauzerni och")

    event = await spotter.process_utterance(_utt(2.0))

    assert event is not None
    assert len(main.calls) == 1
    assert event.trailing_text == "brauzerni och"


async def test_no_match_returns_none_and_skips_main_engine() -> None:
    spotter, kws, main = _spotter(kws_text="salom")

    event = await spotter.process_utterance(_utt(2.0))

    assert event is None
    assert len(main.calls) == 0


async def test_score_is_normalized_to_0_1_range() -> None:
    spotter, _, _ = _spotter(kws_text="altron")

    event = await spotter.process_utterance(_utt(1.0))

    assert event is not None
    assert 0.0 <= event.score <= 1.0


@pytest.mark.parametrize("seconds", [0.1, 0.29, 15.1, 20.0])
async def test_segment_outside_duration_bounds_is_ignored(seconds: float) -> None:
    spotter, kws, _ = _spotter(kws_text="altron")

    event = await spotter.process_utterance(_utt(seconds))

    assert event is None
    assert len(kws.calls) == 0


async def test_quiet_segment_is_skipped_without_calling_stt() -> None:
    spotter, kws, _ = _spotter(kws_text="altron")
    silence = Utterance(
        pcm=np.zeros(int(1.0 * _SR), dtype=np.int16), started_at=0.0, ended_at=1.0, reason="silence"
    )

    event = await spotter.process_utterance(silence)

    assert event is None
    assert len(kws.calls) == 0


async def test_cooldown_suppresses_immediate_retrigger() -> None:
    spotter, kws, _ = _spotter(kws_text="altron", cooldown_ms=10_000)

    first = await spotter.process_utterance(_utt(1.0))
    second = await spotter.process_utterance(_utt(1.0))

    assert first is not None
    assert second is None
    assert len(kws.calls) == 1  # second call short-circuited before ever calling STT


async def test_set_speaking_pauses_entirely_when_barge_in_disabled() -> None:
    spotter, kws, _ = _spotter(kws_text="altron", barge_in_enabled=False)
    spotter.set_speaking(True)

    event = await spotter.process_utterance(_utt(1.0))

    assert event is None
    assert len(kws.calls) == 0


async def test_set_speaking_boosts_rms_gate_when_barge_in_enabled() -> None:
    spotter, kws, _ = _spotter(kws_text="altron", barge_in_enabled=True)
    quietish = _utt(1.0, amplitude=300)  # ~-41 dBFS: above normal gate, below barge-in gate

    not_speaking_event = await spotter.process_utterance(quietish)
    assert not_speaking_event is not None
    assert len(kws.calls) == 1

    spotter.set_speaking(True)
    speaking_event = await spotter.process_utterance(quietish)
    assert speaking_event is None
    assert len(kws.calls) == 1  # unchanged: second call was gated out before STT


async def test_process_frame_is_a_noop_for_stt_kws() -> None:
    spotter, _, _ = _spotter(kws_text="altron")

    assert spotter.process_frame(object()) is None


async def test_queue_drops_oldest_pending_when_full() -> None:
    started = threading.Event()
    release = threading.Event()
    kws = StubEngine("salom", started=started, release=release)
    main = StubEngine("salom")
    spotter = STTKeywordSpotter(
        kws_engine=kws, main_engine=main, matcher=_matcher(), cooldown_ms=0, max_queue=3
    )

    task1 = asyncio.create_task(spotter.process_utterance(_utt(1.0, label=1)))

    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("kws_engine.transcribe was never entered for utterance #1")

    task2 = asyncio.create_task(spotter.process_utterance(_utt(1.0, label=2)))
    task3 = asyncio.create_task(spotter.process_utterance(_utt(1.0, label=3)))
    task4 = asyncio.create_task(spotter.process_utterance(_utt(1.0, label=4)))
    task5 = asyncio.create_task(spotter.process_utterance(_utt(1.0, label=5)))
    await asyncio.sleep(0.05)  # let #2-#5 all reach their "append to queue" point

    release.set()
    results = await asyncio.gather(task1, task2, task3, task4, task5)

    assert results[1] is None  # utterance #2 was dropped while still queued
    seen_labels = sorted(int(call[0]) - 5000 for call in kws.calls)
    assert seen_labels == [1, 3, 4, 5]  # #2 never reached the KWS model at all
