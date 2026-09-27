"""Cross-module data contracts shared by every stage (docs/tz/00_umumiy_arxitektura.md §6).

``EventBus`` (docs/tz/08_orchestrator.md §1.2) is the orchestrator's pub/sub
backbone for the loosely-coupled, cross-task signals that don't already have
a natural callback home: ``WakeEvent`` (audio_loop -> orchestrator),
``AudioDeviceLost`` (``audio.capture.AudioCapture``'s own callback),
``OllamaDown``/``OllamaRestored`` (health_monitor -> orchestrator),
``UserCommand`` (``tray.app.TrayApp`` -> orchestrator, docs/tz/09_autostart.md
§3.4), and ``SystemSuspending``/``SystemResumed``
(``core.power_watcher.PowerWatcher`` -> orchestrator, §3.5). Both
``TrayApp`` and ``PowerWatcher`` run their own OS-level message loop on a
dedicated thread, so ``publish()``'s thread-safety is exactly what lets them
reach the orchestrator's asyncio loop.

``PlaybackStarted``/``PlaybackEnded`` are wired from
``audio.player.AudioPlayer``'s callbacks too, but ``core.orchestrator``
deliberately does *not* build its own THINKING->SPEAKING transition or
first-audio metric on them: those callbacks fire for *any* clip (an "ack"/
"error" earcon included), so using them there would mistake an earcon for
the response actually starting. The orchestrator marks that moment itself,
at the point it hands the first real sentence to TTS. The two events are
still published, ready for a consumer that only cares about raw playback
activity regardless of content (e.g. a future tray icon animation).

The hot audio-frame/utterance path (``AudioFrame``, ``Utterance``) is
deliberately *not* routed through this generic bus: those are exchanged
directly between ``core.orchestrator``'s own tasks (a plain callback for
each freshly-endpointed ``Utterance``), since that path is latency-sensitive
and each event has exactly one natural consumer at a time -- adding a second,
generic indirection there would only cost overhead for no real decoupling
benefit. ``ReminderDue`` likewise reaches the orchestrator via a direct
listener callback (``ultron.tools.reminder_tools.add_due_listener``), not
this bus, because that callback already runs on the same running event loop
(APScheduler's ``AsyncIOScheduler``) -- no cross-thread handoff is needed.

``publish()`` *is* safe to call from any thread (it hands the event to the
bound loop via ``call_soon_threadsafe``), which is what makes it the right
tool for ``AudioDeviceLost``/``PlaybackStarted``: both originate from
callbacks that run on ``sounddevice``'s realtime PortAudio thread, not the
asyncio loop thread.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


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


@dataclass
class WakeEvent:
    """Emitted by a ``WakeDetector`` when "Altron"/"Ultron" is recognized.

    See docs/tz/04_wake_word.md.
    """

    ts: float
    score: float  # 0..1
    source: str  # "stt_kws" | "oww"
    trailing_text: str | None  # "Altron, brauzerni och" -> "brauzerni och"


@dataclass
class Transcript:
    """STT output for one :class:`Utterance` (docs/tz/03_stt.md)."""

    text: str  # normalized, Latin script
    raw_text: str
    language: str
    confidence: float
    duration_s: float
    latency_ms: int


@dataclass
class ReminderDue:
    """Emitted by :class:`ultron.memory.scheduler.ReminderScheduler` when a
    reminder/timer's due time arrives (docs/tz/07_tools.md §7D.3). The
    orchestrator (docs/tz/08_orchestrator.md) subscribes via
    ``ultron.tools.reminder_tools.add_due_listener`` to play the confirm
    earcon, speak ``text`` via TTS, and queue it if Ultron is already
    speaking; until then, ``reminder_tools`` itself just shows a Windows
    toast.
    """

    id: int
    text: str
    kind: str  # "reminder" | "timer"
    repeat: str | None
    overdue: bool = False  # True if it fired late (laptop was off/asleep)


@dataclass
class UserCommand:
    """From the tray icon (``ultron.tray.app.TrayApp``, docs/tz/09_autostart.md
    §3.4). ``text`` only carries a payload for ``"text_command"`` (the tray's
    "Matnli buyruq..." dialog) -- every other command ignores it."""

    # "mute" | "unmute" | "restart" | "quit" | "text_command" | "gpu_saver" | "gpu_normal"
    command: str
    ts: float = field(default_factory=time.time)
    text: str | None = None


@dataclass
class SystemSuspending:
    """Published by ``ultron.core.power_watcher.PowerWatcher`` on
    ``PBT_APMSUSPEND`` (docs/tz/09_autostart.md §3.5)."""

    ts: float


@dataclass
class SystemResumed:
    """Published by ``ultron.core.power_watcher.PowerWatcher`` on
    ``PBT_APMRESUMEAUTOMATIC`` (docs/tz/09_autostart.md §3.5)."""

    ts: float


@dataclass
class OllamaDown:
    """Published by the orchestrator's health_monitor task when a periodic
    ``OllamaClient.health()`` poll fails (docs/tz/08_orchestrator.md §1.1).
    Named apart from ``llm.client.LLMUnavailable`` (an exception raised per
    request) to avoid confusion between the two."""

    ts: float
    detail: str = ""


@dataclass
class OllamaRestored:
    ts: float


@dataclass
class PlaybackStarted:
    """From ``audio.player.AudioPlayer``'s ``on_playback_start`` callback,
    which runs on the realtime PortAudio thread -- publish from there via
    this bus, never call back into asyncio directly from that thread."""

    ts: float


@dataclass
class PlaybackEnded:
    ts: float


class EventBus:
    """Minimal typed pub/sub (docs/tz/08_orchestrator.md §1.2):
    ``publish(event)`` / ``subscribe(event_type) -> AsyncIterator``.

    ``publish`` is thread-safe: it schedules delivery on the bus's bound
    event loop via ``call_soon_threadsafe`` rather than touching any
    ``asyncio.Queue`` directly, so callbacks running on a non-asyncio thread
    (the audio callback thread) can call it exactly like any coroutine can.
    Each ``subscribe`` call gets its own queue, so multiple independent
    subscribers to the same event type each see every event.
    """

    def __init__(self) -> None:
        self._subscribers: dict[type, list[asyncio.Queue[Any]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()

    def publish(self, event: Any) -> None:
        if self._loop is None:
            raise RuntimeError("EventBus.bind_loop() must be called before publish()")
        self._loop.call_soon_threadsafe(self._deliver, event)

    def _deliver(self, event: Any) -> None:
        for event_type, queues in self._subscribers.items():
            if isinstance(event, event_type):
                for q in queues:
                    q.put_nowait(event)

    def subscribe(self, event_type: type[T]) -> AsyncIterator[T]:
        queue: asyncio.Queue[T] = asyncio.Queue()
        self._subscribers.setdefault(event_type, []).append(queue)
        return self._iterate(queue)

    @staticmethod
    async def _iterate(queue: asyncio.Queue[T]) -> AsyncIterator[T]:
        while True:
            yield await queue.get()
