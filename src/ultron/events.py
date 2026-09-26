"""Cross-module data contracts shared by every stage (docs/tz/00_umumiy_arxitektura.md §6).

Only the dataclasses needed by the stages built so far are defined here.
``WakeEvent`` and ``Transcript`` are added by the wake-word (04) and STT (03)
stages when those modules are built; the field shapes are already fixed by
the 00-TZ reference and must not change without checking with the user first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioFrame:
    """One raw capture block: int16 mono PCM, ``block_size`` samples."""

    pcm: np.ndarray
    ts: float = field(default_factory=time.monotonic)


@dataclass
class Utterance:
    """A complete speech segment isolated by the endpointer, pre-roll included."""

    pcm: np.ndarray
    started_at: float
    ended_at: float
    reason: str  # "silence" | "max_length" | "cancelled"
