"""Tests for ultron.core.confirm.VoiceConfirmationProvider (docs/tz/08_orchestrator.md §8B.3).

Every collaborator is injected as a small fake (per the module's own design:
it depends on callables, not on ``core.orchestrator.Orchestrator`` itself),
so the yes/no vocabulary, the retry-once-then-give-up rule, and the timeout
path can all be tested without a real Orchestrator, STT model, or audio I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ultron.core.confirm import CONFIRM_END_SILENCE_MS, VoiceConfirmationProvider
from ultron.events import Utterance


def _utt() -> Utterance:
    return Utterance(
        pcm=np.zeros(1600, dtype=np.int16), started_at=0.0, ended_at=0.0, reason="silence"
    )


@dataclass
class FakeTranscript:
    text: str


class FakeSTT:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    def transcribe(self, pcm: Any, *, initial_prompt: str | None = None) -> FakeTranscript:
        return FakeTranscript(text=self._texts.pop(0) if self._texts else "")


@dataclass
class Collaborators:
    end_silence_values: list[int] = field(default_factory=list)
    utterances: list[Any] = field(default_factory=list)  # queued return values, None = timeout
    enter_calls: int = 0
    leave_calls: int = 0
    earcons: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)

    def set_end_silence_ms(self, value: int) -> None:
        self.end_silence_values.append(value)

    async def wait_for_utterance(self, timeout_s: float) -> Any:
        return self.utterances.pop(0) if self.utterances else None

    async def enter_confirming(self) -> None:
        self.enter_calls += 1

    async def leave_confirming(self) -> None:
        self.leave_calls += 1

    async def play_earcon(self, name: str) -> None:
        self.earcons.append(name)

    async def speak_and_wait(self, text: str) -> None:
        self.spoken.append(text)


def make_provider(
    *, texts: list[str], default_end_silence_ms: int = 700, timeout_s: float = 8.0
) -> tuple[VoiceConfirmationProvider, Collaborators, FakeSTT]:
    collab = Collaborators()
    stt = FakeSTT(texts)
    collab.utterances = [_utt() for _ in texts]  # one non-None utterance per queued transcript
    provider = VoiceConfirmationProvider(
        stt=stt,
        set_end_silence_ms=collab.set_end_silence_ms,
        default_end_silence_ms=default_end_silence_ms,
        wait_for_utterance=collab.wait_for_utterance,
        enter_confirming=collab.enter_confirming,
        leave_confirming=collab.leave_confirming,
        play_earcon=collab.play_earcon,
        speak_and_wait=collab.speak_and_wait,
        timeout_s=timeout_s,
    )
    return provider, collab, stt


async def test_yes_word_confirms() -> None:
    provider, collab, _ = make_provider(texts=["ha"])

    result = await provider.confirm("Kompyuterni o'chirishni tasdiqlaysizmi?")

    assert result is True
    assert collab.enter_calls == 1
    assert collab.leave_calls == 1


async def test_no_word_declines() -> None:
    provider, _, _ = make_provider(texts=["yo'q"])

    result = await provider.confirm("O'chirishni tasdiqlaysizmi?")

    assert result is False


async def test_yes_word_embedded_in_a_longer_answer_still_confirms() -> None:
    provider, _, _ = make_provider(texts=["ha albatta shunday"])

    assert await provider.confirm("Tasdiqlaysizmi?") is True


async def test_end_silence_is_shortened_then_restored() -> None:
    provider, collab, _ = make_provider(texts=["ha"], default_end_silence_ms=700)

    await provider.confirm("Tasdiqlaysizmi?")

    assert collab.end_silence_values == [CONFIRM_END_SILENCE_MS, 700]


async def test_unknown_answer_retries_once_then_confirms() -> None:
    provider, collab, _ = make_provider(texts=["nima gap", "ha"])

    result = await provider.confirm("Tasdiqlaysizmi?")

    assert result is True
    assert collab.earcons == ["confirm", "confirm"]  # asked twice


async def test_unknown_answer_twice_gives_up_as_false() -> None:
    provider, _, _ = make_provider(texts=["nima gap", "boshqa narsa"])

    result = await provider.confirm("Tasdiqlaysizmi?")

    assert result is False


async def test_timeout_declines_and_speaks_cancelled() -> None:
    provider, collab, _ = make_provider(texts=[])  # no utterance ever arrives

    result = await provider.confirm("Tasdiqlaysizmi?")

    assert result is False
    assert any("Bekor qilindi" in text for text in collab.spoken)
    assert collab.leave_calls == 1  # end_silence/state are still restored on timeout
