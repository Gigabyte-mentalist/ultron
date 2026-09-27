"""Voice-based dangerous-tool confirmation (docs/tz/08_orchestrator.md §8B.3).

``VoiceConfirmationProvider`` implements ``tools.base.ConfirmationProvider``
for the live voice pipeline, replacing ``tools.base.ConsoleConfirmationProvider``
(which stays the default for ``--text`` mode). It only depends on small
callables/collaborators injected by ``core.orchestrator.Orchestrator`` rather
than the orchestrator itself, so it can be unit-tested on its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from ultron.events import Utterance
from ultron.stt.base import STTEngine

_YES_WORDS = frozenset(
    {
        "ha",
        "xa",
        "ha ha",
        "albatta",
        "tasdiqlayman",
        "bo'pti",
        "bopti",
        "mayli",
        "ok",
        "davay",
        "да",
    }
)
_NO_WORDS = frozenset({"yo'q", "yoq", "kerak emas", "bekor", "to'xta", "нет"})

_ASK_SUFFIX_UZ = " Ha yoki yo'q deng."
_UNKNOWN_RETRY_UZ = "Tushunmadim. Ha yoki yo'q deng."
_CANCELLED_UZ = "Bekor qilindi."

CONFIRM_END_SILENCE_MS = 400


def _classify(text: str) -> bool | None:
    normalized = text.strip().lower().rstrip(".!?")
    if normalized in _YES_WORDS:
        return True
    if normalized in _NO_WORDS:
        return False
    # Also accept a yes/no word anywhere in a short answer ("ha, albatta").
    words = normalized.split()
    if any(w in _YES_WORDS for w in words):
        return True
    if any(w in _NO_WORDS for w in words):
        return False
    return None


class VoiceConfirmationProvider:
    def __init__(
        self,
        *,
        stt: STTEngine,
        set_end_silence_ms: Callable[[int], None],
        default_end_silence_ms: int,
        wait_for_utterance: Callable[[float], Awaitable[Utterance | None]],
        enter_confirming: Callable[[], Awaitable[None]],
        leave_confirming: Callable[[], Awaitable[None]],
        play_earcon: Callable[[str], Awaitable[None]],
        speak_and_wait: Callable[[str], Awaitable[None]],
        timeout_s: float,
    ) -> None:
        self._stt = stt
        self._set_end_silence_ms = set_end_silence_ms
        self._default_end_silence_ms = default_end_silence_ms
        self._wait_for_utterance = wait_for_utterance
        self._enter_confirming = enter_confirming
        self._leave_confirming = leave_confirming
        self._play_earcon = play_earcon
        self._speak_and_wait = speak_and_wait
        self._timeout_s = timeout_s

    async def confirm(self, summary_uz: str) -> bool:
        await self._enter_confirming()
        self._set_end_silence_ms(CONFIRM_END_SILENCE_MS)
        try:
            answer = await self._ask_once(summary_uz + _ASK_SUFFIX_UZ)
            if answer is None:
                answer = await self._ask_once(_UNKNOWN_RETRY_UZ)
            if answer is None:
                answer = False
            return answer
        finally:
            self._set_end_silence_ms(self._default_end_silence_ms)
            await self._leave_confirming()

    async def _ask_once(self, question_uz: str) -> bool | None:
        await self._play_earcon("confirm")
        await self._speak_and_wait(question_uz)

        utt = await self._wait_for_utterance(self._timeout_s)
        if utt is None:
            await self._speak_and_wait(_CANCELLED_UZ)
            return False

        transcript = await asyncio.to_thread(self._stt.transcribe, utt.pcm)
        answer = _classify(transcript.text)
        logger.info("Ovozli tasdiqlash: '{}' -> {}", transcript.text, answer)
        return answer
