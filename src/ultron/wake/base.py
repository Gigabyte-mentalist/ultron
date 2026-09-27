"""WakeDetector interface (docs/tz/00_umumiy_arxitektura.md §6, docs/tz/04_wake_word.md §3).

``process_frame`` is the fast, always-cheap path used by frame-level
detectors (``oww``); it must never block on STT. ``process_utterance`` is
used by utterance-level detectors (``stt_kws``) and is ``async`` because a
real transcription call sits behind it -- see ``wake/stt_kws.py`` for the
bounded queue that keeps it from piling up work when speech segments arrive
faster than the KWS model can process them.
"""

from __future__ import annotations

from typing import Protocol

from ultron.events import AudioFrame, Utterance, WakeEvent


class WakeDetector(Protocol):
    def load(self) -> None: ...

    def process_frame(self, frame: AudioFrame) -> WakeEvent | None: ...  # oww uchun

    async def process_utterance(self, utt: Utterance) -> WakeEvent | None: ...  # stt_kws uchun

    def reset(self) -> None: ...

    def set_speaking(self, speaking: bool) -> None:
        """Echo/self-trigger guard (docs/tz/04_wake_word.md §3.5).

        Called by the orchestrator when it enters/leaves the ``SPEAKING``
        state. Implementations either pause entirely (``barge_in_enabled``
        is ``False``) or raise their acceptance bar so only a clearly louder
        (real barge-in) utterance can trigger while Ultron is talking.
        """
        ...
