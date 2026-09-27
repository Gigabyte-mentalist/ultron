"""Tests for ultron.core.orchestrator.Orchestrator's state machine
(docs/tz/08_orchestrator.md §1.3, §8A.2, §8B).

Every collaborator is a small duck-typed fake (mirroring ``tests/test_agent.py``'s
own style) so these run instantly with no GPU/audio hardware and no real
Ollama server -- ``Orchestrator`` itself takes every component through its
constructor and never builds one, which is exactly what makes this possible.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ultron.core.orchestrator as orchestrator_module
from ultron.audio.capture import AudioDeviceLost
from ultron.config import RootConfig
from ultron.core.metrics import RequestMetrics
from ultron.core.orchestrator import Orchestrator
from ultron.core.state import State
from ultron.events import ReminderDue, UserCommand, Utterance, WakeEvent
from ultron.llm.agent import DoneEvent, ErrorEvent, SentenceEvent, ToolEndEvent, ToolStartEvent
from ultron.llm.fastpath import FastPathMatch
from ultron.tools.base import ToolResult

# Every test spawns fire-and-forget background tasks (LISTENING/FOLLOW_UP
# waits, running commands) exactly like the real pipeline does -- an autouse
# fixture cancels whatever is left after each test so a lingering task (e.g.
# a FOLLOW_UP window's 8 s wait) can't survive into the next test's event
# loop and blow up once that loop closes.
_created_orchestrators: list[Orchestrator] = []

# ``RootConfig().app.data_dir``'s pydantic default is the real ``C:/ultron/data``
# (TZ-01). docs/tz/11_uzbek_stt_finetune.md §3.1 made ``_handle_listening_utterance``/
# ``_follow_up_loop`` -- the two most exercised code paths in this whole file --
# write real files under ``<data_dir>/recordings/utterances`` whenever
# ``memory.save_utterances`` (default ``True``) is on, so ``make_orchestrator()``
# below redirects any config still on that default to an ephemeral directory
# instead of trusting each of the 60+ tests here to opt out individually.
_DEFAULT_DATA_DIR = RootConfig().app.data_dir


@pytest.fixture(autouse=True)
async def _cleanup_orchestrators() -> AsyncIterator[None]:
    yield
    while _created_orchestrators:
        orch = _created_orchestrators.pop()
        tasks = list(orch._background_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


_SR = 16000


def _pcm(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(seconds * _SR), dtype=np.int16)


def _utt(seconds: float = 1.0) -> Utterance:
    return Utterance(pcm=_pcm(seconds), started_at=0.0, ended_at=0.0, reason="silence")


class FakeCapture:
    def __init__(self) -> None:
        self.muted = False
        self.started = False
        self.stopped = False
        self.rms_dbfs = -40.0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def mute(self) -> None:
        self.muted = True

    def unmute(self) -> None:
        self.muted = False

    async def frames(self) -> AsyncIterator[Any]:
        if False:  # pragma: no cover -- makes this an async generator; never actually yields
            yield None


class FakePlayer:
    """``is_playing`` flips back to False immediately after ``enqueue`` so
    ``Orchestrator._wait_for_playback_done`` never actually has to poll."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[np.ndarray, int]] = []
        self.stop_calls = 0
        self.started = False
        self.stream_stopped = False
        self.is_playing = False

    def start(self) -> None:
        self.started = True

    def stop_stream(self) -> None:
        self.stream_stopped = True

    def enqueue(self, pcm: np.ndarray, sr: int) -> None:
        self.enqueued.append((pcm, sr))

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_playing = False


class FakeWake:
    def __init__(self) -> None:
        self.speaking_calls: list[bool] = []
        self.reset_calls = 0
        self._next_utterance_event: WakeEvent | None = None

    def load(self) -> None:
        pass

    def reset(self) -> None:
        self.reset_calls += 1

    def set_speaking(self, speaking: bool) -> None:
        self.speaking_calls.append(speaking)

    def process_frame(self, frame: Any) -> WakeEvent | None:
        return None

    async def process_utterance(self, utt: Utterance) -> WakeEvent | None:
        event = self._next_utterance_event
        self._next_utterance_event = None
        return event

    def queue_wake_event(self, event: WakeEvent) -> None:
        self._next_utterance_event = event


@dataclass
class FakeTranscript:
    text: str
    confidence: float = 0.9
    raw_text: str = ""
    language: str = "uz"
    duration_s: float = 1.0
    latency_ms: int = 10


class FakeSTT:
    def __init__(self, transcripts: list[FakeTranscript]) -> None:
        self._transcripts = list(transcripts)

    def load(self) -> None:
        pass

    def transcribe(self, pcm: np.ndarray, *, initial_prompt: str | None = None) -> Any:
        return self._transcripts.pop(0) if self._transcripts else FakeTranscript(text="")


class FakeTTS:
    sample_rate = _SR

    def __init__(self) -> None:
        self.cancel_calls = 0
        self.spoken: list[str] = []

    def load(self) -> None:
        pass

    def synthesize(self, text: str) -> np.ndarray:
        self.spoken.append(text)
        return np.zeros(10, dtype=np.float32)

    async def speak_stream(self, sentences: AsyncIterator[str], player: Any) -> None:
        async for text in sentences:
            self.spoken.append(text)
            player.enqueue(np.zeros(10, dtype=np.float32), self.sample_rate)

    def cancel(self) -> None:
        self.cancel_calls += 1


@dataclass
class FakeAgent:
    responses: list[list[Any]] = field(default_factory=list)
    fastpath_enabled: bool = True
    reset_history_calls: int = 0

    def reset_history(self) -> None:
        self.reset_history_calls += 1

    async def respond(self, text: str) -> AsyncIterator[Any]:
        events = self.responses.pop(0) if self.responses else [DoneEvent()]
        for event in events:
            # A small real await between events (rather than yielding with no
            # awaiting at all) so a test observing an *intermediate* state
            # (e.g. SPEAKING, mid-response) has a reliable window to do so --
            # without it, a no-op fake response can race past every
            # intermediate state within a single event-loop tick.
            await asyncio.sleep(0.02)
            yield event


class FakeFastPath:
    def match(self, text: str) -> FastPathMatch | None:
        return None


class FakeLLMClient:
    def __init__(self, *, healthy: bool = True, model_loaded: bool = True) -> None:
        self.healthy = healthy
        self.model_loaded = model_loaded
        self.keep_alive_calls: list[int | str] = []
        self.unload_calls = 0
        self.ensure_loaded_calls = 0

    async def health(self) -> bool:
        return self.healthy

    async def is_model_loaded(self) -> bool:
        return self.model_loaded

    def set_keep_alive(self, keep_alive: int | str) -> None:
        self.keep_alive_calls.append(keep_alive)

    async def unload(self) -> None:
        self.unload_calls += 1

    async def ensure_loaded(self) -> None:
        self.ensure_loaded_calls += 1


class FakeMetricsWriter:
    def __init__(self) -> None:
        self.records: list[RequestMetrics] = []

    def record(self, metrics: RequestMetrics) -> None:
        self.records.append(metrics)


class FakeRingBuffer:
    """``.snapshot()`` returns empty by default, matching a real ``RingBuffer``
    with nothing written to it yet (most FSM tests never run ``_audio_loop``)."""

    def __init__(self) -> None:
        self._next = np.zeros(0, dtype=np.int16)

    def queue_snapshot(self, pcm: np.ndarray) -> None:
        self._next = pcm

    def snapshot(self, seconds: float) -> np.ndarray:
        return self._next


class FakeTray:
    def __init__(self) -> None:
        self.states: list[State] = []
        self.ollama_healthy: list[bool] = []
        self.last_latency: list[float] = []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def set_state(self, state: State) -> None:
        self.states.append(state)

    def set_ollama_healthy(self, healthy: bool) -> None:
        self.ollama_healthy.append(healthy)

    def set_last_latency(self, seconds: float) -> None:
        self.last_latency.append(seconds)


def make_orchestrator(
    *,
    agent: FakeAgent | None = None,
    wake: FakeWake | None = None,
    stt: FakeSTT | None = None,
    tts: Any = None,
    config: RootConfig | None = None,
    llm_client: FakeLLMClient | None = None,
    tray: FakeTray | None = None,
    ring_buffer: FakeRingBuffer | None = None,
) -> tuple[Orchestrator, dict[str, Any]]:
    resolved_config = config or RootConfig()
    if resolved_config.app.data_dir == _DEFAULT_DATA_DIR:
        resolved_config.app.data_dir = Path(tempfile.mkdtemp(prefix="ultron_test_"))
    parts = {
        "capture": FakeCapture(),
        "player": FakePlayer(),
        "wake": wake or FakeWake(),
        "stt": stt or FakeSTT([]),
        "tts": FakeTTS() if tts is None else tts,
        "agent": agent or FakeAgent(),
        "fastpath": FakeFastPath(),
        "llm_client": llm_client or FakeLLMClient(),
        "metrics": FakeMetricsWriter(),
        "tray": tray,
        "ring_buffer": ring_buffer or FakeRingBuffer(),
    }
    orch = Orchestrator(
        resolved_config,
        bus=orchestrator_module.EventBus(),
        capture=parts["capture"],
        player=parts["player"],
        ring_buffer=parts["ring_buffer"],
        vad=object(),
        endpointer=_FakeEndpointer(),
        wake=parts["wake"],
        stt=parts["stt"],
        tts=parts["tts"],
        agent=parts["agent"],
        fastpath=parts["fastpath"],
        llm_client=parts["llm_client"],
        metrics=parts["metrics"],
        tray=parts["tray"],
    )
    orch._bus.bind_loop()
    _created_orchestrators.append(orch)
    return orch, parts


class _FakeEndpointer:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.end_silence_values: list[int] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def set_end_silence_ms(self, value: int) -> None:
        self.end_silence_values.append(value)


async def _arm_listening(orch: Orchestrator) -> None:
    """``_enter_listening`` spawns ``_listening_loop`` as a background task;
    it only arms the shared utterance future once that task actually gets to
    run, so tests must yield control once before delivering an utterance."""
    await orch._enter_listening()
    await asyncio.sleep(0)


async def _arm_follow_up(orch: Orchestrator) -> None:
    await orch._enter_follow_up()
    await asyncio.sleep(0)


async def _run_until_idle_or_state(
    orch: Orchestrator, states: tuple[State, ...], timeout: float = 2.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while orch.state not in states:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"Timed out waiting for {states}, stuck in {orch.state}")
        await asyncio.sleep(0.01)


async def _wait_for_command_to_finish(orch: Orchestrator, timeout: float = 2.0) -> None:
    """Wait for a command spawned via ``_spawn_command`` to fully finish.

    Unlike ``_run_until_idle_or_state``, this doesn't race against the
    *current* state already matching the target (e.g. FOLLOW_UP, when the
    command was spawned from inside a FOLLOW_UP window that hasn't actually
    progressed yet) -- it waits on the task itself.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while orch._active_agent_task is None:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("No command was ever spawned")
        await asyncio.sleep(0.01)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(orch._active_agent_task, timeout=timeout)


# -- IDLE + WakeEvent --------------------------------------------------------


async def test_wake_event_without_trailing_text_enters_listening() -> None:
    orch, parts = make_orchestrator()

    await orch._on_wake_event(WakeEvent(ts=0.0, score=0.9, source="stt_kws", trailing_text=None))
    await asyncio.sleep(0)  # let the spawned _listening_loop task start and arm the future

    assert orch.state == State.LISTENING
    assert len(parts["player"].enqueued) == 1  # "wake" earcon


async def test_wake_event_with_short_trailing_text_still_enters_listening() -> None:
    orch, _ = make_orchestrator()

    await orch._on_wake_event(WakeEvent(ts=0.0, score=0.9, source="stt_kws", trailing_text="och"))
    await asyncio.sleep(0)

    assert orch.state == State.LISTENING  # single word: not "trailing_text >= 2 words"


async def test_wake_event_with_trailing_text_goes_straight_to_thinking() -> None:
    agent = FakeAgent(responses=[[SentenceEvent("Bajarildi."), DoneEvent()]])
    orch, parts = make_orchestrator(agent=agent)

    await orch._on_wake_event(
        WakeEvent(ts=0.0, score=0.9, source="stt_kws", trailing_text="brauzerni och")
    )
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP, State.IDLE))

    assert parts["tts"].spoken == ["Bajarildi."]
    assert len(parts["metrics"].records) == 1
    assert parts["metrics"].records[0].wake_source == "stt_kws"


# -- wake_auto auto-collection (docs/tz/10_wake_word_training.md §3.2) -------


async def test_save_wake_auto_sample_writes_wav_when_configured(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    ring_buffer = FakeRingBuffer()
    ring_buffer.queue_snapshot(np.ones(8000, dtype=np.int16))
    orch, _ = make_orchestrator(config=config, ring_buffer=ring_buffer)

    await orch._save_wake_auto_sample()

    saved = sorted((tmp_path / "recordings" / "wake_auto").glob("*.wav"))
    assert len(saved) == 1


async def test_save_wake_auto_sample_skipped_when_save_utterances_disabled(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    config.memory.save_utterances = False
    ring_buffer = FakeRingBuffer()
    ring_buffer.queue_snapshot(np.ones(8000, dtype=np.int16))
    orch, _ = make_orchestrator(config=config, ring_buffer=ring_buffer)

    await orch._save_wake_auto_sample()

    assert not (tmp_path / "recordings" / "wake_auto").exists()


async def test_save_wake_auto_sample_skipped_when_ring_buffer_empty(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    orch, _ = make_orchestrator(config=config)  # FakeRingBuffer's default snapshot is empty

    await orch._save_wake_auto_sample()

    assert not (tmp_path / "recordings" / "wake_auto").exists()


async def test_wake_event_schedules_wake_auto_save(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    ring_buffer = FakeRingBuffer()
    ring_buffer.queue_snapshot(np.ones(8000, dtype=np.int16))
    orch, _ = make_orchestrator(config=config, ring_buffer=ring_buffer)

    await orch._on_wake_event(WakeEvent(ts=0.0, score=0.9, source="stt_kws", trailing_text=None))
    save_task = next(
        t for t in orch._background_tasks if t.get_coro().__name__ == "_save_wake_auto_sample"
    )
    await save_task

    saved = sorted((tmp_path / "recordings" / "wake_auto").glob("*.wav"))
    assert len(saved) == 1


# -- LISTENING ----------------------------------------------------------------


async def test_listening_low_confidence_retries_once_then_gives_up() -> None:
    stt = FakeSTT(
        [FakeTranscript(text="", confidence=0.0), FakeTranscript(text="", confidence=0.0)]
    )
    orch, parts = make_orchestrator(stt=stt)
    orch._config.vad.listen_timeout_s = 1

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await asyncio.sleep(0.05)  # first low-confidence transcript -> retry, re-armed
    assert orch.state == State.LISTENING
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.IDLE,))

    assert orch.state == State.IDLE


async def test_listening_cancel_phrase_returns_to_idle() -> None:
    stt = FakeSTT([FakeTranscript(text="bekor qil")])
    orch, _ = make_orchestrator(stt=stt)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.IDLE,))

    assert orch.state == State.IDLE


async def test_listening_timeout_returns_to_idle_with_done_earcon() -> None:
    orch, parts = make_orchestrator()
    orch._config.vad.listen_timeout_s = 0.05

    await _arm_listening(orch)
    await _run_until_idle_or_state(orch, (State.IDLE,))

    assert orch.state == State.IDLE
    # the "wake" earcon is played by the caller (_on_wake_event), which this
    # test bypasses by calling _enter_listening directly -- only "done" here.
    assert len(parts["player"].enqueued) == 1


async def test_listening_good_command_reaches_follow_up() -> None:
    stt = FakeSTT([FakeTranscript(text="soat nechi", confidence=0.9)])
    agent = FakeAgent(responses=[[SentenceEvent("Hozir soat 14:05."), DoneEvent()]])
    orch, parts = make_orchestrator(stt=stt, agent=agent)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP,))

    assert orch.state == State.FOLLOW_UP
    assert parts["tts"].spoken == ["Hozir soat 14:05."]


# -- FOLLOW_UP ------------------------------------------------------------------


async def test_follow_up_end_phrase_returns_to_idle() -> None:
    stt = FakeSTT([FakeTranscript(text="rahmat", confidence=0.9)])
    orch, parts = make_orchestrator(stt=stt)

    await _arm_follow_up(orch)
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.IDLE,))

    assert orch.state == State.IDLE
    assert "Xizmatingizdaman" in parts["tts"].spoken


async def test_follow_up_timeout_returns_to_idle_silently() -> None:
    config = RootConfig()
    config.conversation.follow_up_window_s = 0.05
    orch, parts = make_orchestrator(config=config)

    await _arm_follow_up(orch)
    await _run_until_idle_or_state(orch, (State.IDLE,))

    assert orch.state == State.IDLE
    assert parts["tts"].spoken == []


async def test_follow_up_ignores_too_short_utterance() -> None:
    config = RootConfig()
    config.conversation.follow_up_window_s = 0.2
    orch, _ = make_orchestrator(config=config)

    await _arm_follow_up(orch)
    orch._deliver_utterance(_utt(seconds=0.1))  # below the 0.5 s floor
    await _run_until_idle_or_state(orch, (State.IDLE,), timeout=1.0)

    assert orch.state == State.IDLE  # timed out normally, the short blip was ignored


async def test_follow_up_ignores_low_confidence_utterance() -> None:
    config = RootConfig()
    config.conversation.follow_up_window_s = 0.2
    stt = FakeSTT([FakeTranscript(text="nimadir", confidence=0.1)])
    orch, _ = make_orchestrator(config=config, stt=stt)

    await _arm_follow_up(orch)
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.IDLE,), timeout=1.0)

    assert orch.state == State.IDLE


async def test_follow_up_command_goes_to_thinking() -> None:
    stt = FakeSTT([FakeTranscript(text="ovozni balandlat", confidence=0.9)])
    agent = FakeAgent(responses=[[SentenceEvent("Ovoz oshirildi."), DoneEvent()]])
    orch, parts = make_orchestrator(stt=stt, agent=agent)

    await _arm_follow_up(orch)
    orch._deliver_utterance(_utt())
    await _wait_for_command_to_finish(orch)

    assert parts["tts"].spoken == ["Ovoz oshirildi."]


# -- barge-in -------------------------------------------------------------------


async def test_barge_in_during_speaking_stops_player_and_tts_and_reenters_listening() -> None:
    agent = FakeAgent(
        responses=[[SentenceEvent("Uzoq javob boshlanmoqda."), ToolStartEvent("noop")]]
    )
    orch, parts = make_orchestrator(agent=agent)

    await orch._transition(State.THINKING, "test_setup")
    orch._spawn_command("uzun buyruq", wake_source="stt_kws", confidence=0.9, vad_end=0.0)
    await _run_until_idle_or_state(orch, (State.SPEAKING,))

    await orch._on_wake_event(WakeEvent(ts=0.0, score=0.9, source="stt_kws", trailing_text=None))
    await asyncio.sleep(0)

    assert parts["player"].stop_calls >= 1
    assert parts["tts"].cancel_calls >= 1
    assert orch.state == State.LISTENING


# -- CONFIRMING -----------------------------------------------------------------


async def test_enter_and_leave_confirming_restores_previous_state() -> None:
    orch, _ = make_orchestrator()
    await orch._transition(State.THINKING, "test_setup")

    await orch._enter_confirming()
    assert orch.state == State.CONFIRMING

    await orch._leave_confirming()
    assert orch.state == State.THINKING


# -- reminders ------------------------------------------------------------------


async def test_reminder_due_while_idle_is_spoken_immediately() -> None:
    orch, parts = make_orchestrator()

    await orch._on_reminder_due(ReminderDue(id=1, text="Choy tayyor", kind="reminder", repeat=None))

    assert any("Choy tayyor" in text for text in parts["tts"].spoken) or parts["player"].enqueued


async def test_reminder_due_while_thinking_is_queued_then_flushed() -> None:
    agent = FakeAgent(responses=[[SentenceEvent("Javob."), DoneEvent()]])
    orch, parts = make_orchestrator(agent=agent)
    await orch._transition(State.THINKING, "test_setup")

    await orch._on_reminder_due(ReminderDue(id=1, text="Choy tayyor", kind="reminder", repeat=None))
    assert orch._pending_reminders == ["Eslatma: Choy tayyor"]

    orch._spawn_command("davom et", wake_source="stt_kws", confidence=0.9, vad_end=0.0)
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP,))

    assert orch._pending_reminders == []
    assert "Eslatma: Choy tayyor" in parts["tts"].spoken


# -- user commands / device loss -------------------------------------------------


async def test_user_command_mute_and_unmute() -> None:
    orch, parts = make_orchestrator()

    await orch._on_user_command(UserCommand(command="mute"))
    assert orch.state == State.MUTED
    assert parts["capture"].muted is True

    await orch._on_user_command(UserCommand(command="unmute"))
    assert orch.state == State.IDLE
    assert parts["capture"].muted is False


async def test_user_command_restart_stops_with_exit_code_ten() -> None:
    """docs/tz/09_autostart.md: tray "Qayta ishga tushirish" is a *full*
    process restart (supervisor sees exit 10 and relaunches immediately),
    deliberately distinct from the lighter in-process ``_soft_restart()``
    that repeated-error recovery uses."""
    orch, _ = make_orchestrator()

    await orch._on_user_command(UserCommand(command="restart"))

    assert orch._exit_code == 10
    assert orch._stop_event.is_set()


async def test_user_command_quit_stops_with_exit_code_zero() -> None:
    orch, _ = make_orchestrator()

    await orch._on_user_command(UserCommand(command="quit"))

    assert orch._exit_code == 0
    assert orch._stop_event.is_set()


async def test_user_command_text_command_runs_like_a_listening_utterance() -> None:
    agent = FakeAgent(responses=[[SentenceEvent("Bajarildi."), DoneEvent()]])
    orch, parts = make_orchestrator(agent=agent)

    await orch._on_user_command(UserCommand(command="text_command", text="soat nechi"))
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP, State.IDLE))

    assert parts["tts"].spoken == ["Bajarildi."]
    assert parts["metrics"].records[0].wake_source == "tray_text"


async def test_user_command_text_command_ignored_while_busy() -> None:
    agent = FakeAgent(responses=[[SentenceEvent("Bajarildi."), DoneEvent()]])
    orch, parts = make_orchestrator(agent=agent)
    await orch._transition(State.THINKING, "test_setup")
    orch._spawn_command("uzun buyruq", wake_source="stt_kws", confidence=0.9, vad_end=0.0)
    await asyncio.sleep(0)
    assert orch.state == State.THINKING

    await orch._on_user_command(UserCommand(command="text_command", text="boshqa buyruq"))

    # Only the original command ran -- the tray text command was dropped.
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP, State.IDLE))
    assert parts["tts"].spoken == ["Bajarildi."]
    assert len(parts["metrics"].records) == 1


async def test_user_command_gpu_saver_unloads_model_and_pins_keep_alive_zero() -> None:
    llm_client = FakeLLMClient()
    orch, _ = make_orchestrator(llm_client=llm_client)

    await orch._on_user_command(UserCommand(command="gpu_saver"))

    assert orch._gpu_saver_active is True
    assert llm_client.unload_calls == 1
    assert llm_client.keep_alive_calls == [0]


async def test_user_command_gpu_normal_restores_configured_keep_alive() -> None:
    llm_client = FakeLLMClient()
    orch, _ = make_orchestrator(llm_client=llm_client)
    await orch._on_user_command(UserCommand(command="gpu_saver"))

    await orch._on_user_command(UserCommand(command="gpu_normal"))

    assert orch._gpu_saver_active is False
    assert llm_client.ensure_loaded_calls == 1
    assert llm_client.keep_alive_calls == [0, orch._config.llm.keep_alive]


async def test_transition_updates_tray_state() -> None:
    tray = FakeTray()
    orch, _ = make_orchestrator(tray=tray)

    await orch._on_user_command(UserCommand(command="mute"))

    assert State.MUTED in tray.states


async def test_ollama_down_and_restored_update_tray() -> None:
    tray = FakeTray()
    orch, _ = make_orchestrator(tray=tray)

    await orch._on_ollama_down(orchestrator_module.OllamaDown(ts=0.0, detail="x"))
    assert tray.ollama_healthy == [False]

    await orch._on_ollama_restored(orchestrator_module.OllamaRestored(ts=0.0))
    assert tray.ollama_healthy == [False, True]


# -- suspend / resume (docs/tz/09_autostart.md §3.5) -----------------------


async def test_system_suspend_mutes_capture_and_enters_muted() -> None:
    orch, parts = make_orchestrator()

    await orch._on_system_suspending(orchestrator_module.SystemSuspending(ts=0.0))

    assert parts["capture"].muted is True
    assert orch.state == State.MUTED


async def test_system_resume_unmutes_and_returns_to_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch, parts = make_orchestrator()
    monkeypatch.setattr(orchestrator_module, "_RESUME_SETTLE_S", 0.0)
    await orch._on_system_suspending(orchestrator_module.SystemSuspending(ts=0.0))

    await orch._on_system_resumed(orchestrator_module.SystemResumed(ts=0.0))

    assert parts["capture"].muted is False
    assert orch.state == State.IDLE


async def test_system_resume_keeps_manual_mute(monkeypatch: pytest.MonkeyPatch) -> None:
    """A suspend that happens while the user had already muted manually must
    not be auto-unmuted just because the laptop woke back up."""
    orch, parts = make_orchestrator()
    monkeypatch.setattr(orchestrator_module, "_RESUME_SETTLE_S", 0.0)
    await orch._on_user_command(UserCommand(command="mute"))
    await orch._on_system_suspending(orchestrator_module.SystemSuspending(ts=0.0))

    await orch._on_system_resumed(orchestrator_module.SystemResumed(ts=0.0))

    assert parts["capture"].muted is True
    assert orch.state == State.MUTED


async def test_system_resume_stops_with_exit_ten_when_stt_warmup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenSTT(FakeSTT):
        def transcribe(self, pcm: Any, *, initial_prompt: str | None = None) -> Any:
            raise RuntimeError("CUDA osilib qoldi")

    orch, _ = make_orchestrator(stt=_BrokenSTT([]))
    monkeypatch.setattr(orchestrator_module, "_RESUME_SETTLE_S", 0.0)

    await orch._on_system_resumed(orchestrator_module.SystemResumed(ts=0.0))

    assert orch._exit_code == 10
    assert orch._stop_event.is_set()


async def test_audio_device_lost_enters_error_state() -> None:
    orch, _ = make_orchestrator()

    await orch._on_audio_device_lost(AudioDeviceLost(ts=0.0, detail="unplugged"))

    assert orch.state == State.ERROR


# -- error handling / soft restart ------------------------------------------------


async def test_five_errors_in_window_trigger_soft_restart() -> None:
    orch, _ = make_orchestrator()

    for _ in range(5):
        await orch._on_handler_error()

    assert orch._endpointer.reset_calls == 1
    assert orch._wake.reset_calls == 1
    assert orch._agent.reset_history_calls == 1
    assert orch.state == State.IDLE  # soft_restart itself finishes back at IDLE


async def test_soft_restart_twice_within_window_gives_up_and_stops() -> None:
    orch, _ = make_orchestrator()

    for _ in range(5):
        await orch._on_handler_error()
    for _ in range(5):
        await orch._on_handler_error()

    assert orch._exit_code == 3
    assert orch._stop_event.is_set()


# -- metrics ----------------------------------------------------------------------


async def test_successful_command_records_metrics_with_expected_fields() -> None:
    stt = FakeSTT([FakeTranscript(text="soat nechi", confidence=0.87)])
    agent = FakeAgent(
        responses=[
            [
                ToolStartEvent("get_time"),
                ToolEndEvent(ToolResult(ok=True, speak="Hozir 14:05.")),
                DoneEvent(),
            ]
        ]
    )
    orch, parts = make_orchestrator(stt=stt, agent=agent)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.THINKING, State.FOLLOW_UP), timeout=2.0)
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP,))

    assert len(parts["metrics"].records) == 1
    record = parts["metrics"].records[0]
    assert record.stt_text == "soat nechi"
    assert record.confidence == 0.87
    assert record.tools == ["get_time"]
    assert record.error is None


async def test_error_event_from_agent_is_spoken_and_recorded() -> None:
    agent = FakeAgent(
        responses=[[ErrorEvent("Hozir o'ylay olmayapman, bir oz kuting"), DoneEvent()]]
    )
    orch, parts = make_orchestrator(agent=agent)

    await orch._transition(State.THINKING, "test_setup")
    orch._spawn_command("nimadir", wake_source="stt_kws", confidence=0.9, vad_end=0.0)
    await _wait_for_command_to_finish(orch)

    assert parts["metrics"].records[0].error == "Hozir o'ylay olmayapman, bir oz kuting"


# -- personal STT dataset capture (docs/tz/11_uzbek_stt_finetune.md §3.1) -----


def _utterance_index_rows(data_dir: Path) -> list[dict[str, Any]]:
    day_dir = data_dir / "recordings" / "utterances" / date.today().isoformat()
    index_path = day_dir / "index.jsonl"
    if not index_path.is_file():
        return []
    lines = index_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


async def test_save_utterances_disabled_skips_capture(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    config.memory.save_utterances = False
    stt = FakeSTT([FakeTranscript(text="bekor qil", confidence=0.9)])
    orch, _ = make_orchestrator(config=config, stt=stt)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.IDLE,))

    assert not (tmp_path / "recordings" / "utterances").exists()


async def test_listening_low_confidence_utterance_is_captured_unlabeled(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    stt = FakeSTT([FakeTranscript(text="xira gap", confidence=0.1)])
    orch, _ = make_orchestrator(config=config, stt=stt)
    orch._config.vad.listen_timeout_s = 1

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await asyncio.sleep(0.05)  # low confidence -> LISTEN_RETRY, re-armed for a 2nd try

    rows = _utterance_index_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["stt_text"] == "xira gap"
    assert rows[0]["confidence"] == 0.1
    assert rows[0]["label"] == "unlabeled"
    assert rows[0]["tool_called"] is False
    assert rows[0]["tool_ok"] is None


async def test_listening_cancel_phrase_utterance_is_captured_unlabeled(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    stt = FakeSTT([FakeTranscript(text="bekor qil", confidence=0.9)])
    orch, _ = make_orchestrator(config=config, stt=stt)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.IDLE,))

    rows = _utterance_index_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["stt_text"] == "bekor qil"
    assert rows[0]["label"] == "unlabeled"
    assert rows[0]["tool_called"] is False


async def test_successful_tool_command_eventually_saves_silver_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator_module, "SILVER_WINDOW_S", 0.02)
    config = RootConfig()
    config.app.data_dir = tmp_path
    stt = FakeSTT([FakeTranscript(text="telegramni och", confidence=0.9)])
    agent = FakeAgent(
        responses=[[ToolStartEvent("open_app"), ToolEndEvent(ToolResult(ok=True)), DoneEvent()]]
    )
    orch, _ = make_orchestrator(config=config, stt=stt, agent=agent)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await _run_until_idle_or_state(orch, (State.THINKING, State.FOLLOW_UP), timeout=2.0)
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP,))

    pending_task = next(
        t
        for t in orch._background_tasks
        if t.get_coro().__name__ == "_resolve_silver_window_after_delay"
    )
    await pending_task

    rows = _utterance_index_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["stt_text"] == "telegramni och"
    assert rows[0]["label"] == "silver"
    assert rows[0]["tool_called"] is True
    assert rows[0]["tool_ok"] is True


async def test_failed_tool_command_saves_unlabeled_immediately(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    stt = FakeSTT([FakeTranscript(text="telegramni och", confidence=0.9)])
    agent = FakeAgent(
        responses=[
            [
                ToolStartEvent("open_app"),
                ToolEndEvent(ToolResult(ok=False, error="topilmadi")),
                DoneEvent(),
            ]
        ]
    )
    orch, _ = make_orchestrator(config=config, stt=stt, agent=agent)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    # ``_finalize_utterance_immediate`` writes via ``asyncio.to_thread`` *inside*
    # ``_run_command``'s ``finally`` -- state already flips to FOLLOW_UP before
    # that write lands, so waiting on the state alone (like most tests here do)
    # races the write; waiting for the task itself doesn't.
    await _wait_for_command_to_finish(orch)

    rows = _utterance_index_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["label"] == "unlabeled"
    assert rows[0]["tool_called"] is True
    assert rows[0]["tool_ok"] is False


async def test_command_without_any_tool_call_saves_unlabeled(tmp_path: Path) -> None:
    config = RootConfig()
    config.app.data_dir = tmp_path
    stt = FakeSTT([FakeTranscript(text="soat nechi", confidence=0.9)])
    agent = FakeAgent(responses=[[SentenceEvent("Hozir soat 14:05."), DoneEvent()]])
    orch, _ = make_orchestrator(config=config, stt=stt, agent=agent)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await _wait_for_command_to_finish(orch)

    rows = _utterance_index_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["label"] == "unlabeled"
    assert rows[0]["tool_called"] is False
    assert rows[0]["tool_ok"] is None


async def test_repeating_command_within_silver_window_keeps_it_unlabeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator_module, "SILVER_WINDOW_S", 1.0)
    config = RootConfig()
    config.app.data_dir = tmp_path
    stt = FakeSTT(
        [
            FakeTranscript(text="telegramni och", confidence=0.9),
            FakeTranscript(text="telegramni och", confidence=0.9),
        ]
    )
    agent = FakeAgent(
        responses=[
            [ToolStartEvent("open_app"), ToolEndEvent(ToolResult(ok=True)), DoneEvent()],
            [ToolStartEvent("open_app"), ToolEndEvent(ToolResult(ok=True)), DoneEvent()],
        ]
    )
    orch, _ = make_orchestrator(config=config, stt=stt, agent=agent)

    await _arm_listening(orch)
    orch._deliver_utterance(_utt())
    await _wait_for_command_to_finish(orch)
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP,))
    # ``_follow_up_loop`` (spawned by the just-finished command's own
    # ``_enter_follow_up()``) needs one scheduling round to reach its
    # ``_wait_for_utterance`` call and arm the shared future -- same reason
    # ``_arm_follow_up`` yields once before delivering (see its docstring).
    await asyncio.sleep(0)

    # Said again, right away, while the first command's silver window is
    # still open -- docs/tz/11's "qayta aytmagan" signal.
    orch._deliver_utterance(_utt())
    await _wait_for_command_to_finish(orch)
    await _run_until_idle_or_state(orch, (State.FOLLOW_UP,))

    pending_tasks = [
        t
        for t in orch._background_tasks
        if t.get_coro().__name__ == "_resolve_silver_window_after_delay"
    ]
    assert len(pending_tasks) == 2  # both commands succeeded -> both opened a window
    await asyncio.gather(*pending_tasks)

    rows = _utterance_index_rows(tmp_path)
    assert len(rows) == 2
    assert rows[0]["label"] == "unlabeled"  # invalidated by the repeat below
    assert rows[1]["label"] == "silver"  # nothing said/repeated after it
