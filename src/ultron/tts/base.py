"""TTS engine interface (docs/tz/00_umumiy_arxitektura.md §6, docs/tz/05_tts.md §3.4)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from ultron.audio.player import AudioPlayer


class TTSEngine(Protocol):
    sample_rate: int

    def load(self) -> None: ...

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize one chunk of already-clean text to float32 PCM in [-1, 1]."""
        ...

    async def speak_stream(self, sentences: AsyncIterator[str], player: AudioPlayer) -> None:
        """Synthesize sentences one at a time and queue them on ``player`` as they're ready.

        The next sentence's synthesis starts as soon as the previous one has
        been queued (not once it has finished playing), so synthesis and
        playback pipeline naturally.
        """
        ...

    def cancel(self) -> None:
        """Stop synthesizing further queued sentences (barge-in)."""
        ...
