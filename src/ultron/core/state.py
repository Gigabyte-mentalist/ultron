"""Orchestrator finite-state machine states (docs/tz/08_orchestrator.md §1.3).

State changes only ever happen through ``Orchestrator._transition(new, reason)``
so every change gets logged in one place -- individual handlers never assign
to the state directly.
"""

from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    FOLLOW_UP = "FOLLOW_UP"
    CONFIRMING = "CONFIRMING"
    MUTED = "MUTED"
    ERROR = "ERROR"
