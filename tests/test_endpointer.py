"""Tests for ultron.audio.vad.Endpointer (docs/tz/02_audio.md §4).

Uses a synthetic sequence of VAD probabilities (no ONNX model, no
microphone) to exercise start/end timing, pre-roll, max-length cutoff, and
end-silence hysteresis.
"""

from __future__ import annotations

import numpy as np

from ultron.audio.ring_buffer import RingBuffer
from ultron.audio.vad import Endpointer, ListenTimeout
from ultron.events import Utterance

_SR = 1000
_BLOCK = 10  # 10 ms/frame at this sample rate


def _endpointer(**overrides) -> tuple[Endpointer, RingBuffer]:
    ring = RingBuffer(capacity_seconds=1.0, sample_rate=_SR)
    params = dict(
        sample_rate=_SR,
        threshold=0.5,
        min_speech_ms=30,
        end_silence_ms=50,
        max_utterance_s=1.0,
        listen_timeout_s=5.0,
        preroll_ms=30.0,
    )
    params.update(overrides)
    return Endpointer(ring, **params), ring


def _push_many(ep: Endpointer, ring: RingBuffer, values, probs):
    result = None
    for v, p in zip(values, probs, strict=True):
        frame = np.full(_BLOCK, v, dtype=np.int16)
        ring.write(frame)
        r = ep.push(frame, p)
        if r is not None:
            result = r
    return result


def test_start_end_and_preroll() -> None:
    ep, ring = _endpointer()
    ep.set_mode("passive")

    # settle in silence, filling the ring buffer with recognizable content
    _push_many(ep, ring, range(20), [0.05] * 20)

    # 3 speech frames (30 ms) reach min_speech_ms and confirm speech onset
    r = _push_many(ep, ring, range(20, 23), [0.9, 0.9, 0.9])
    assert r is None

    # 5 confident-silence frames (50 ms) reach end_silence_ms and close it
    utt = _push_many(ep, ring, range(23, 29), [0.05] * 6)

    assert isinstance(utt, Utterance)
    assert utt.reason == "silence"
    assert utt.started_at <= utt.ended_at
    # pre-roll (30 ms = 3 frames) captured [20, 21, 22]; trailing frames append [23..27]
    assert utt.pcm[0] == 20
    assert utt.pcm[-1] == 27
    assert len(utt.pcm) == 8 * _BLOCK


def test_hysteresis_ambiguous_probability_does_not_reset_or_end() -> None:
    ep, ring = _endpointer(threshold=0.5, min_speech_ms=20, end_silence_ms=30)
    ep.set_mode("passive")

    r = _push_many(ep, ring, [0, 1], [0.9, 0.9])  # 20 ms -> speech confirmed
    assert r is None

    r = _push_many(ep, ring, [2], [0.05])  # first confident-silence frame (10 ms)
    assert r is None

    # ambiguous frame (between threshold-0.15=0.35 and threshold=0.5): must hold steady
    r = _push_many(ep, ring, [3], [0.4])
    assert r is None

    r = _push_many(ep, ring, [4], [0.05])  # 20 ms total
    assert r is None

    utt = _push_many(ep, ring, [5], [0.05])  # 30 ms total -> closes
    assert isinstance(utt, Utterance)
    assert utt.reason == "silence"


def test_short_pause_between_words_does_not_split_the_utterance() -> None:
    ep, ring = _endpointer(min_speech_ms=20, end_silence_ms=100)
    ep.set_mode("passive")

    _push_many(ep, ring, [0, 1], [0.9, 0.9])  # speech confirmed

    # a 40 ms pause (< end_silence_ms=100) then speech resumes
    r = _push_many(ep, ring, [2, 3, 4, 5], [0.05, 0.05, 0.05, 0.05])
    assert r is None
    r = _push_many(ep, ring, [6, 7], [0.9, 0.9])
    assert r is None  # back to SPEECH, not a new/closed utterance

    utt = _push_many(ep, ring, range(8, 20), [0.05] * 12)
    assert isinstance(utt, Utterance)
    assert utt.reason == "silence"
    # one continuous utterance, not split into two segments
    assert utt.pcm[0] == 0


def test_max_utterance_length_forces_cutoff() -> None:
    ep, ring = _endpointer(max_utterance_s=0.05, min_speech_ms=10)
    ep.set_mode("passive")

    r = _push_many(ep, ring, [0], [0.9])
    assert r is None

    result = None
    for v in range(1, 10):
        result = _push_many(ep, ring, [v], [0.9])
        if result is not None:
            break

    assert isinstance(result, Utterance)
    assert result.reason == "max_length"


def test_listen_timeout_fires_once_in_active_mode() -> None:
    ep, ring = _endpointer(listen_timeout_s=0.03)
    ep.set_mode("active")

    results = _collect(ep, ring, range(10), [0.05] * 10)

    timeouts = [r for r in results if isinstance(r, ListenTimeout)]
    assert len(timeouts) == 1


def _collect(ep: Endpointer, ring: RingBuffer, values, probs):
    out = []
    for v, p in zip(values, probs, strict=True):
        frame = np.full(_BLOCK, v, dtype=np.int16)
        ring.write(frame)
        out.append(ep.push(frame, p))
    return out
