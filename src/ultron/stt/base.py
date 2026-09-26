"""STT engine interface (docs/tz/00_umumiy_arxitektura.md §6, docs/tz/03_stt.md §3.1)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ultron.events import Transcript


class STTEngine(Protocol):
    def load(self) -> None: ...

    def transcribe(self, pcm: np.ndarray, *, initial_prompt: str | None = None) -> Transcript: ...
