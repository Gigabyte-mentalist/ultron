"""The full voice pipeline's state machine (docs/tz/08_orchestrator.md).

``build_orchestrator()`` does the real, hardware-touching startup work (§8A.1:
loading STT/wake/TTS models, wiring the tool registry, connecting to Ollama)
and returns a fully-wired :class:`Orchestrator`. The class itself takes every
collaborator through its constructor and never constructs one itself, so a
test can build an ``Orchestrator`` directly out of fakes without touching any
real model or audio device (see ``tests/test_orchestrator_fsm.py``).

See the module docstring of ``ultron.events`` for why only a handful of
signals (``WakeEvent``, ``AudioDeviceLost``, ``UserCommand``, ``OllamaDown``/
``OllamaRestored``) travel over the generic ``EventBus`` while the hot
``Utterance`` path and ``ReminderDue`` are wired directly, and why
``PlaybackStarted``/``PlaybackEnded`` are published but not subscribed to
here.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf
from loguru import logger

from ultron.audio import earcons
from ultron.audio.capture import AudioCapture, AudioDeviceLost
from ultron.audio.player import AudioPlayer
from ultron.audio.ring_buffer import RingBuffer
from ultron.audio.vad import Endpointer, SileroVAD, default_model_path
from ultron.companion.widget import CompanionWidget
from ultron.config import RootConfig
from ultron.core.confirm import VoiceConfirmationProvider
from ultron.core.metrics import MetricsWriter, RequestTimer
from ultron.core.ollama_launcher import launch_ollama_hidden
from ultron.core.power_watcher import PowerWatcher
from ultron.core.state import State
from ultron.events import (
    EventBus,
    OllamaDown,
    OllamaRestored,
    PlaybackEnded,
    PlaybackStarted,
    ReminderDue,
    SystemResumed,
    SystemSuspending,
    UserCommand,
    Utterance,
    WakeEvent,
)
from ultron.i18n.uz import (
    FOLLOW_UP_CLOSING,
    GENERIC_ERROR,
    LISTEN_RETRY,
    OLLAMA_RECONNECTING,
    READY,
    REMINDER_OVERDUE_PREFIX,
    REMINDER_PREFIX,
    WAIT_A_MOMENT,
)
from ultron.llm.agent import (
    Agent,
    DoneEvent,
    ErrorEvent,
    SentenceEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from ultron.llm.client import OllamaClient
from ultron.llm.fastpath import FastPath
from ultron.memory.utterance_log import (
    SILVER_WINDOW_S,
    PendingSilver,
    UtteranceLog,
    UtteranceRecord,
)
from ultron.stt.base import STTEngine
from ultron.tools import (
    async_init_all,
    async_shutdown_all,
    configure_all,
    default_registry,
    reminder_tools,
    system_tools,
)
from ultron.tools.file_tools import FileIndexer
from ultron.tools.memory_tools import get_facts_for_prompt
from ultron.tray.app import TrayApp, start_hotkey_listener
from ultron.tts.base import TTSEngine
from ultron.tts.mms_engine import MMSEngine
from ultron.wake import create_wake_detector
from ultron.wake.base import WakeDetector

_SAMPLE_RATE = 16000

# docs/tz/10_wake_word_training.md §3.2: har bir WakeEvent'ning boshidagi shuncha
# soniyasi ``memory.save_utterances=True`` bo'lsa ``recordings/wake_auto/``ga saqlanadi.
_WAKE_AUTO_SECONDS = 1.5

# 8A.2: bo'sh / past ishonchli transkript LISTENING'da bir marta qayta so'raladi.
_LISTEN_MIN_CONFIDENCE = 0.35
# 8A.2: FOLLOW_UP'da yolg'on trigger'ni kamaytirish.
_FOLLOW_UP_MIN_DURATION_S = 0.5
_FOLLOW_UP_MIN_CONFIDENCE = 0.45

# 8B.1: streaming ack/"Bir daqiqa..." vaqt chegaralari.
_TOOL_ACK_DELAY_S = 0.4
_TOOL_SLOW_NOTICE_S = 2.0
_AGENT_POLL_S = 0.1
_PLAYBACK_POLL_S = 0.05

# 8A.3: 1 daqiqada 5 ta xato -> soft_restart; ikkinchi marta -> exit(3).
_ERROR_WINDOW_S = 60.0
_ERROR_THRESHOLD = 5

_HEALTH_CHECK_INTERVAL_S = 30.0
# docs/tz/00_umumiy_arxitektura.md §3: ~7.0-7.5 GB jami VRAM byudjeti.
_VRAM_BUDGET_MB = 7600
_EXIT_UNRECOVERABLE = 3

# 09-TZ §3.2.5: supervisor kills the process if this file goes 30 s stale.
_HEARTBEAT_INTERVAL_S = 5.0
_HEARTBEAT_PATH_PARTS = ("tmp", "heartbeat")

# 09-TZ §3.6: Ollama down for 3 consecutive 30 s checks (~90 s) without the
# health monitor's own relaunch attempt (fired on the *first* failure, well
# before this) succeeding -> tell the user out loud/via toast.
_OLLAMA_ANNOUNCE_AFTER_FAILURES = 3
# 09-TZ §3.6: disk/RSS health thresholds.
_DISK_FREE_MIN_GB = 5.0
_RSS_SOFT_RESTART_MB = 4096
# Files an old-data cleanup must never delete out from under the process
# that's actively writing them (mirrors ultron.supervisor.ACTIVE_LOG_NAMES,
# duplicated rather than imported so core.orchestrator never has to import
# the separate supervisor *process*'s module just for one constant).
_ACTIVE_LOG_NAMES = frozenset(
    {"ultron.log", "tools_audit.log", "supervisor.log", "metrics.jsonl", "ultron_stdout.log"}
)

# 09-TZ §3.5: resume settle time before re-opening audio/probing Ollama --
# CUDA's context sometimes needs a moment after a suspend/resume cycle.
_RESUME_SETTLE_S = 3.0


class CriticalStartupError(Exception):
    """A component docs/tz/08_orchestrator.md §8A.1 marks critical (audio,
    STT/wake) failed to load. The caller (``__main__.py``) retries every 30 s."""


def _matches_any_phrase(normalized_text: str, phrases: list[str]) -> bool:
    return any(phrase in normalized_text for phrase in phrases)


def _resolve_main_stt_engine(detector: WakeDetector) -> STTEngine:
    """Reuse the wake detector's already-loaded main STT engine for general
    command transcription too, rather than loading a second whisper model
    into VRAM (docs/tz/00_umumiy_arxitektura.md §3's ~7-7.5 GB budget has no
    room for two). Works for ``stt_kws`` -- today's only functional mode,
    since ``oww``/``hybrid`` need docs/tz/10_wake_word_training.md's model,
    not yet trained, and ``wake.create_wake_detector`` falls back to
    ``stt_kws`` automatically until then."""
    main_engine = getattr(detector, "main_engine", None)
    if main_engine is not None:
        return main_engine
    raise RuntimeError(
        f"Wake detector {type(detector).__name__} o'zining asosiy STT modelini "
        "ochib bermaydi; buyruqlarni transkripsiya qilish uchun kerak "
        "(hybrid/oww rejimlari hali to'liq qo'llab-quvvatlanmaydi, TZ-10 kutilmoqda)"
    )


def _gpu_vram_used_mb() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return int(proc.stdout.strip().splitlines()[0])
    except ValueError:
        return None


class Orchestrator:
    """The state machine described in docs/tz/08_orchestrator.md §1.3/§8A.2/§8B.

    Every collaborator is injected; the class does no model loading or device
    I/O setup of its own -- see :func:`build_orchestrator` for that.
    """

    def __init__(
        self,
        config: RootConfig,
        *,
        bus: EventBus,
        capture: AudioCapture,
        player: AudioPlayer,
        ring_buffer: RingBuffer,
        vad: SileroVAD,
        endpointer: Endpointer,
        wake: WakeDetector,
        stt: STTEngine,
        tts: TTSEngine | None,
        agent: Agent,
        fastpath: FastPath,
        llm_client: OllamaClient,
        metrics: MetricsWriter,
        debug: bool = False,
        tray: TrayApp | None = None,
        power_watcher: PowerWatcher | None = None,
        hotkey_listener: Any | None = None,
        companion: CompanionWidget | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._capture = capture
        self._player = player
        self._ring_buffer = ring_buffer
        self._vad = vad
        self._endpointer = endpointer
        self._wake = wake
        self._stt = stt
        self._tts = tts
        self._agent = agent
        self._fastpath = fastpath
        self._llm_client = llm_client
        self._metrics = metrics
        self._debug = debug
        # docs/tz/09_autostart.md: all optional so tests/test_orchestrator_fsm.py's
        # fakes-only fixture (no tray/power hardware) keeps working unmodified.
        self._tray = tray
        self._power_watcher = power_watcher
        self._hotkey_listener = hotkey_listener
        self._companion = companion

        self._state = State.IDLE
        self._pre_confirm_state: State | None = None
        self._utterance_future: asyncio.Future[Utterance] | None = None
        self._active_agent_task: asyncio.Task[None] | None = None
        # Every task spawned fire-and-forget outside ``self._tasks`` (a
        # LISTENING/FOLLOW_UP wait, a wake-check, a running command) is
        # tracked here too, purely so ``_shutdown()`` can cancel *all* of
        # them -- without this, one could keep running past shutdown
        # (dangling GPU/Ollama calls) since none of them is any other task's
        # child.
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._active_timer: RequestTimer | None = None
        self._pending_reminders: list[str] = []
        self._pending_wake_ms: int | None = None
        self._listening_wake_source = "stt_kws"
        self._listening_wake_ms: int | None = None
        self._error_timestamps: list[float] = []
        self._last_soft_restart_at: float | None = None
        self._audio_device_down = False
        self._file_indexer: FileIndexer | None = None
        self._debug_rms = float("-inf")
        self._debug_vad_prob = 0.0
        self._tasks: list[asyncio.Task[Any]] = []
        self._exit_code = 0
        self._wake_auto_index: int | None = None
        self._stop_event = asyncio.Event()
        # docs/tz/11_uzbek_stt_finetune.md §3.1 -- gated per-call on
        # ``config.memory.save_utterances``, same flag as ``_save_wake_auto_sample``.
        self._utterance_log = UtteranceLog(Path(config.app.data_dir) / "recordings" / "utterances")
        self._active_utterance_record: UtteranceRecord | None = None

        # docs/tz/09_autostart.md §3.6/§3.7.
        self._gpu_saver_active = False
        self._ollama_down_since: float | None = None
        self._ollama_consecutive_failures = 0
        self._ollama_announced = False
        # docs/tz/09_autostart.md §3.5: a suspend that happens while the user
        # had already muted manually must not be auto-unmuted on resume.
        self._muted_before_suspend = False

    @property
    def state(self) -> State:
        return self._state

    # -- lifecycle ----------------------------------------------------

    async def run(self) -> int:
        self._bus.bind_loop()
        self._capture.start()
        self._player.start()
        self._track_background(self._build_file_index_background())
        if self._config.memory.save_utterances:
            self._track_background(self._cleanup_stale_utterances())
        if self._tray is not None:
            self._tray.start()
        if self._power_watcher is not None:
            self._power_watcher.start()
        if self._companion is not None:
            self._companion.start()

        if self._config.autostart.announce_ready:
            await self._speak_and_wait(READY)

        self._tasks = [
            asyncio.ensure_future(self._audio_loop()),
            asyncio.ensure_future(self._orchestrator_loop()),
            asyncio.ensure_future(self._health_monitor_loop()),
            asyncio.ensure_future(self._heartbeat_loop()),
        ]
        if self._debug:
            self._tasks.append(asyncio.ensure_future(self._debug_status_loop()))

        try:
            await self._stop_event.wait()
        finally:
            await self._shutdown()
        return self._exit_code

    async def stop(self, exit_code: int = 0) -> None:
        self._exit_code = exit_code
        self._stop_event.set()

    def _track_background(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _shutdown(self) -> None:
        logger.info("Ultron to'xtatilmoqda...")
        all_tasks = [*self._tasks, *self._background_tasks]
        for task in all_tasks:
            task.cancel()
        for task in all_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._file_indexer is not None:
            self._file_indexer.stop_watching()
        self._capture.stop()
        self._player.stop_stream()
        if self._tray is not None:
            self._tray.stop()
        if self._power_watcher is not None:
            self._power_watcher.stop()
        if self._companion is not None:
            self._companion.stop()
        if self._hotkey_listener is not None:
            with contextlib.suppress(Exception):
                self._hotkey_listener.stop()
        await async_shutdown_all()

    async def _build_file_index_background(self) -> None:
        """docs/tz/07_tools.md §7B explicitly deferred calling
        ``FileIndexer.build_index()``/``start_watching()`` to this stage.
        Runs in the background (not awaited by ``run()``) since
        ``allowed_roots`` can include a whole drive -- scanning it must not
        block the ~40 s startup budget in §8A.1."""
        try:
            from ultron.memory import db as memory_db

            db = await memory_db.get_db()
            self._file_indexer = FileIndexer(db, self._config.tools.files.allowed_roots)
            count = await self._file_indexer.build_index()
            self._file_indexer.start_watching()
            logger.info("Fayl indeksi qurildi: {} fayl, kuzatuv yoqildi", count)
        except Exception:  # noqa: BLE001 -- indexing is a nice-to-have, not critical
            logger.exception("Fayl indeksini qurishda xato")

    # -- FSM primitives -------------------------------------------------

    async def _transition(self, new: State, reason: str) -> None:
        old = self._state
        if old == new:
            return
        logger.info("FSM: {} -> {} ({})", old.value, new.value, reason)
        self._state = new
        if self._tray is not None:
            self._tray.set_state(new)
        if new in (State.IDLE, State.FOLLOW_UP):
            # Centralized here rather than at each call site landing on IDLE/
            # FOLLOW_UP (timeouts, cancel/end phrases, a finished command,
            # handler-error recovery, ...) -- one reminder that arrived
            # during THINKING/SPEAKING/LISTENING/CONFIRMING must be spoken
            # the moment we're next quiet, no matter which path got us there
            # (docs/tz/08_orchestrator.md §8B.4).
            await self._flush_pending_reminders()

    async def _enter_confirming(self) -> None:
        self._pre_confirm_state = self._state
        await self._transition(State.CONFIRMING, "tool_dangerous")

    async def _leave_confirming(self) -> None:
        restore = self._pre_confirm_state or State.THINKING
        self._pre_confirm_state = None
        await self._transition(restore, "confirm_resolved")

    def _leave_speaking(self) -> None:
        self._wake.set_speaking(False)

    async def _play_earcon(self, name: str) -> None:
        self._player.enqueue(earcons.generate(name), 16000)

    async def _toast_fallback(self, text: str) -> None:
        try:
            import win11toast

            await asyncio.to_thread(win11toast.toast, "Ultron", text)
        except Exception as exc:  # noqa: BLE001 -- a toast failure must not break anything
            logger.warning("Toast ko'rsatib bo'lmadi: {}", exc)

    async def _speak_and_wait(self, text: str) -> None:
        """Speak one short, already-clean canned phrase and wait for it to
        finish playing (docs/tz/08_orchestrator.md §8A.1: TTS unavailable ->
        the response is shown as a toast instead)."""
        if self._tts is None:
            await self._toast_fallback(text)
            return
        self._wake.set_speaking(True)
        try:
            pcm = await asyncio.to_thread(self._tts.synthesize, text)
            if pcm.size:
                self._player.enqueue(pcm, self._tts.sample_rate)
                await self._wait_for_playback_done()
        finally:
            self._wake.set_speaking(False)

    async def _wait_for_playback_done(self) -> None:
        while self._player.is_playing:
            await asyncio.sleep(_PLAYBACK_POLL_S)

    async def _wait_for_utterance(self, timeout_s: float) -> Utterance | None:
        """Arm the single-slot utterance channel and wait up to ``timeout_s``
        for the Endpointer's next segment. Used by LISTENING, FOLLOW_UP, and
        ``core.confirm.VoiceConfirmationProvider`` -- exactly one of these is
        ever "armed" at a time, so there is no contention."""
        future: asyncio.Future[Utterance] = asyncio.get_running_loop().create_future()
        self._utterance_future = future
        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        except TimeoutError:
            return None
        finally:
            if self._utterance_future is future:
                self._utterance_future = None

    def _deliver_utterance(self, utt: Utterance) -> None:
        """Called synchronously from ``_audio_loop`` for every Endpointer-
        emitted ``Utterance``. Endpointer is kept in "passive" mode always
        (see ``_audio_loop``'s docstring); LISTENING/FOLLOW_UP/CONFIRMING's
        own timeouts are implemented here rather than via the Endpointer's
        active-mode ``listen_timeout_s``, since those three states each need
        a different timeout duration."""
        future = self._utterance_future
        if future is not None and not future.done():
            future.set_result(utt)
            return
        if self._state in (State.IDLE, State.THINKING, State.SPEAKING):
            self._track_background(self._check_wake(utt))

    async def _check_wake(self, utt: Utterance) -> None:
        started = time.monotonic()
        try:
            event = await self._wake.process_utterance(utt)
        except Exception:  # noqa: BLE001 -- one bad segment must not kill the pipeline
            logger.exception("Wake detektorida xato")
            return
        if event is not None:
            self._pending_wake_ms = int((time.monotonic() - started) * 1000)
            self._bus.publish(event)

    # -- background tasks -----------------------------------------------

    async def _audio_loop(self) -> None:
        """capture.frames() -> ring_buffer, VAD, wake(frame), Endpointer (§1.1)."""
        async for frame in self._capture.frames():
            if self._audio_device_down:
                self._audio_device_down = False
                logger.info("Audio qurilma tiklandi")
                await self._transition(State.IDLE, "audio_device_restored")

            self._ring_buffer.write(frame.pcm)
            prob = self._vad.process(frame.pcm)
            self._debug_rms = self._capture.rms_dbfs
            self._debug_vad_prob = prob

            wake_event = self._wake.process_frame(frame)
            if wake_event is not None:
                self._bus.publish(wake_event)

            result = self._endpointer.push(frame.pcm, prob)
            if isinstance(result, Utterance):
                self._deliver_utterance(result)

    async def _orchestrator_loop(self) -> None:
        """Reads WakeEvent/AudioDeviceLost/UserCommand/OllamaDown/Restored/
        SystemSuspending/SystemResumed off the EventBus and dispatches to
        handlers (§1.1, docs/tz/09_autostart.md §3.5)."""
        await asyncio.gather(
            self._pump(self._bus.subscribe(WakeEvent), self._on_wake_event),
            self._pump(self._bus.subscribe(AudioDeviceLost), self._on_audio_device_lost),
            self._pump(self._bus.subscribe(UserCommand), self._on_user_command),
            self._pump(self._bus.subscribe(OllamaDown), self._on_ollama_down),
            self._pump(self._bus.subscribe(OllamaRestored), self._on_ollama_restored),
            self._pump(self._bus.subscribe(SystemSuspending), self._on_system_suspending),
            self._pump(self._bus.subscribe(SystemResumed), self._on_system_resumed),
        )

    async def _pump(self, subscription: AsyncIterator[Any], handler: Any) -> None:
        async for event in subscription:
            try:
                await handler(event)
            except Exception:  # noqa: BLE001 -- one bad event must not kill the pump
                logger.exception("Hodisa handler'ida kutilmagan xato")

    async def _health_monitor_loop(self) -> None:
        """Ollama (reachability + model-loaded), audio device (already
        self-healing at the ``AudioCapture`` layer, docs/tz/02_audio.md's own
        watchdog -- nothing to add here), VRAM, disk, RSS -- every 30 s
        (§1.1, docs/tz/09_autostart.md §3.6)."""
        ollama_was_down = False
        while True:
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL_S)
            try:
                healthy = await self._llm_client.health()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ollama health-check xatosi: {}", exc)
                healthy = False

            if healthy:
                if ollama_was_down:
                    ollama_was_down = False
                    self._bus.publish(OllamaRestored(ts=time.time()))
                self._ollama_consecutive_failures = 0
                self._ollama_announced = False
                if not self._gpu_saver_active:
                    with contextlib.suppress(Exception):
                        if not await self._llm_client.is_model_loaded():
                            logger.info("Model VRAM'da emas, qayta yuklanmoqda")
                            await self._llm_client.ensure_loaded()
            else:
                self._ollama_consecutive_failures += 1
                if not ollama_was_down:
                    ollama_was_down = True
                    self._bus.publish(OllamaDown(ts=time.time(), detail="health() false"))
                    # Fired on the *first* failed check (not the 3rd, below)
                    # so a manually-killed Ollama has the best chance of
                    # being back up before docs/tz/09_autostart.md §5's
                    # "Ollama'ni o'chirish -> <= 90 s ichida tiklanadi".
                    if self._config.autostart.ollama_start_if_missing:
                        launched = await asyncio.to_thread(launch_ollama_hidden)
                        if launched:
                            logger.info("Ollama qayta ishga tushirildi (health monitor)")
                if (
                    self._ollama_consecutive_failures >= _OLLAMA_ANNOUNCE_AFTER_FAILURES
                    and not self._ollama_announced
                ):
                    self._ollama_announced = True
                    if self._state in (State.IDLE, State.FOLLOW_UP):
                        await self._speak_and_wait(OLLAMA_RECONNECTING)
                    else:
                        await self._toast_fallback(OLLAMA_RECONNECTING)

            vram_mb = await asyncio.to_thread(_gpu_vram_used_mb)
            if vram_mb is not None and vram_mb > _VRAM_BUDGET_MB:
                logger.warning(
                    "VRAM byudjetdan oshdi: {} MB (limit {} MB)", vram_mb, _VRAM_BUDGET_MB
                )

            await self._check_disk_space()
            await self._check_process_rss()

    async def _heartbeat_loop(self) -> None:
        """docs/tz/09_autostart.md §3.2 step 5: written every 5 s so
        ``ultron.supervisor`` can tell a wedged process (no more heartbeats)
        apart from one that's merely busy."""
        path = Path(self._config.app.data_dir).joinpath(*_HEARTBEAT_PATH_PARTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                path.write_text(str(time.time()), encoding="utf-8")
            except OSError as exc:  # noqa: BLE001 -- must never kill the process
                logger.warning("Heartbeat yozib bo'lmadi: {}", exc)
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)

    async def _check_disk_space(self) -> None:
        data_dir = Path(self._config.app.data_dir)
        try:
            free_gb = await asyncio.to_thread(lambda: shutil.disk_usage(data_dir).free / 1e9)
        except OSError:
            return
        if free_gb >= _DISK_FREE_MIN_GB:
            return
        logger.warning(
            "Disk bo'sh joyi {:.1f} GB'dan kam ({:.1f} GB), eski fayllar tozalanmoqda",
            _DISK_FREE_MIN_GB,
            free_gb,
        )
        await asyncio.to_thread(self._cleanup_old_files, data_dir)

    def _cleanup_old_files(self, data_dir: Path) -> None:
        """Best-effort: deletes the oldest logs/recordings first, stopping as
        soon as free space clears the threshold (docs/tz/09_autostart.md
        §3.6's "Eski log va utterance yozuvlarini tozalash"). Loguru's own
        rotation/retention already bounds ``ultron.log``/etc under normal
        conditions -- this is the backstop for when that isn't enough."""
        candidates: list[Path] = []
        logs_dir = data_dir / "logs"
        if logs_dir.is_dir():
            candidates += [
                p for p in logs_dir.iterdir() if p.is_file() and p.name not in _ACTIVE_LOG_NAMES
            ]
        recordings_dir = data_dir / "recordings"
        if recordings_dir.is_dir():
            candidates += [p for p in recordings_dir.rglob("*") if p.is_file()]
        candidates.sort(key=lambda p: p.stat().st_mtime)

        for path in candidates:
            if shutil.disk_usage(data_dir).free / 1e9 >= _DISK_FREE_MIN_GB:
                break
            with contextlib.suppress(OSError):
                path.unlink()
        logger.info(
            "Tozalashdan keyin bo'sh joy: {:.1f} GB", shutil.disk_usage(data_dir).free / 1e9
        )

    async def _check_process_rss(self) -> None:
        try:
            rss_mb = await asyncio.to_thread(
                lambda: psutil.Process().memory_info().rss / (1024 * 1024)
            )
        except psutil.Error:
            return
        if rss_mb <= _RSS_SOFT_RESTART_MB:
            return
        logger.warning(
            "Jarayon xotirasi {} MB byudjetdan oshdi ({:.0f} MB), qayta yuklanmoqda",
            _RSS_SOFT_RESTART_MB,
            rss_mb,
        )
        await self._soft_restart()

    async def _debug_status_loop(self) -> None:
        """--debug rejimi (§8B.6): jonli holat, DEBUG log qatori sifatida
        (CLAUDE.md: hech qachon ``print`` ishlatilmaydi, faqat logger)."""
        while True:
            await asyncio.sleep(1.0)
            vram_mb = await asyncio.to_thread(_gpu_vram_used_mb)
            vram_txt = f"{vram_mb / 1024:.1f} GB" if vram_mb is not None else "?"
            logger.debug(
                "[{}] mic {:.0f} dBFS | VAD {:.2f} | VRAM {}",
                self._state.value,
                self._debug_rms,
                self._debug_vad_prob,
                vram_txt,
            )

    # -- WakeEvent / barge-in --------------------------------------------

    async def _on_wake_event(self, event: WakeEvent) -> None:
        self._track_background(self._save_wake_auto_sample())
        if self._state in (State.MUTED, State.ERROR, State.CONFIRMING):
            return
        if self._state in (State.SPEAKING, State.THINKING):
            await self._barge_in(event)
            return
        if self._state not in (State.IDLE, State.FOLLOW_UP):
            return
        await self._react_to_wake(event)

    async def _save_wake_auto_sample(self) -> None:
        """docs/tz/10_wake_word_training.md §3.2: auto-collect real "Altron"
        samples from live usage, for later training -- opt-out via
        ``memory.save_utterances`` since it writes short voice clips to disk."""
        if not self._config.memory.save_utterances:
            return
        try:
            snapshot = self._ring_buffer.snapshot(_WAKE_AUTO_SECONDS)
            if snapshot.size == 0:
                return
            out_dir = Path(self._config.app.data_dir) / "recordings" / "wake_auto"
            await asyncio.to_thread(self._write_wake_auto_wav, out_dir, snapshot)
        except Exception:  # noqa: BLE001 -- best-effort data collection, never fatal
            logger.exception("wake_auto namunasini saqlab bo'lmadi")

    def _write_wake_auto_wav(self, out_dir: Path, pcm: np.ndarray) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._wake_auto_index is None:
            existing = [int(p.stem) for p in out_dir.glob("*.wav") if p.stem.isdigit()]
            self._wake_auto_index = max(existing, default=-1) + 1
        target = out_dir / f"{self._wake_auto_index:04d}.wav"
        sf.write(target, pcm, self._config.audio.sample_rate, subtype="PCM_16")
        self._wake_auto_index += 1

    # -- docs/tz/11_uzbek_stt_finetune.md §3.1: personal STT dataset capture -

    async def _cleanup_stale_utterances(self) -> None:
        """Runs once per process start (autostart/supervisor already restart
        often enough that a periodic loop would add complexity for little
        gain) -- best-effort, never blocks startup on failure."""
        try:
            await asyncio.to_thread(self._utterance_log.cleanup_stale)
        except Exception:  # noqa: BLE001 -- disk hygiene must not crash the orchestrator
            logger.exception("Eski utterance yozuvlarini tozalashda xato")

    async def _capture_utterance(
        self, pcm: np.ndarray, *, text: str, confidence: float, wake_source: str
    ) -> UtteranceRecord | None:
        return await asyncio.to_thread(
            self._utterance_log.capture,
            pcm,
            text=text,
            confidence=confidence,
            model=self._config.stt.model,
            wake_source=wake_source,
        )

    async def _finalize_utterance_immediate(
        self, record: UtteranceRecord | None, *, tool_called: bool, tool_ok: bool | None
    ) -> None:
        if record is None:
            return
        await asyncio.to_thread(
            self._utterance_log.finalize_immediate, record, tool_called=tool_called, tool_ok=tool_ok
        )

    async def _resolve_silver_window_after_delay(self, pending: PendingSilver) -> None:
        await asyncio.sleep(SILVER_WINDOW_S)
        await asyncio.to_thread(self._utterance_log.resolve_silver_window, pending)

    async def _barge_in(self, event: WakeEvent) -> None:
        """docs/tz/08_orchestrator.md §8B.2."""
        logger.info("Barge-in: wake source={}", event.source)
        self._player.stop()
        if self._tts is not None:
            self._tts.cancel()
        task = self._active_agent_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._leave_speaking()
        await self._react_to_wake(event)

    async def _react_to_wake(self, event: WakeEvent) -> None:
        wake_ms = self._pending_wake_ms
        vad_end = time.monotonic() - ((wake_ms or 0) / 1000)
        trailing = (event.trailing_text or "").strip()

        if trailing and len(trailing.split()) >= 2:
            await self._play_earcon("ack")
            await self._transition(State.THINKING, "wake_with_trailing")
            self._spawn_command(
                trailing,
                wake_source=event.source,
                confidence=None,
                vad_end=vad_end,
                wake_ms=wake_ms,
            )
        else:
            await self._play_earcon("wake")
            self._listening_wake_source = event.source
            self._listening_wake_ms = wake_ms
            await self._enter_listening()

    # -- LISTENING --------------------------------------------------------

    async def _enter_listening(self) -> None:
        await self._transition(State.LISTENING, "await_command")
        self._track_background(self._listening_loop())

    async def _listening_loop(self, *, retried: bool = False) -> None:
        utt = await self._wait_for_utterance(self._config.vad.listen_timeout_s)
        vad_end = time.monotonic()
        if self._state != State.LISTENING:
            return  # superseded (barge-in/mute/etc.) while we were waiting
        if utt is None:
            await self._play_earcon("done")
            await self._transition(State.IDLE, "listen_timeout")
            return
        await self._handle_listening_utterance(utt, vad_end=vad_end, retried=retried)

    async def _handle_listening_utterance(
        self, utt: Utterance, *, vad_end: float, retried: bool
    ) -> None:
        transcript = await asyncio.to_thread(self._stt.transcribe, utt.pcm)
        text = transcript.text.strip()
        normalized = text.lower().rstrip(".!?")
        is_cancel = _matches_any_phrase(normalized, self._config.conversation.cancel_phrases)

        record: UtteranceRecord | None = None
        if self._config.memory.save_utterances:
            self._utterance_log.note_new_utterance(normalized, is_cancel=is_cancel)
            record = await self._capture_utterance(
                utt.pcm,
                text=text,
                confidence=transcript.confidence,
                wake_source=self._listening_wake_source,
            )

        if not text or transcript.confidence < _LISTEN_MIN_CONFIDENCE:
            await self._finalize_utterance_immediate(record, tool_called=False, tool_ok=None)
            if not retried:
                await self._speak_and_wait(LISTEN_RETRY)
                await self._listening_loop(retried=True)
            else:
                await self._play_earcon("done")
                await self._transition(State.IDLE, "listen_retry_exhausted")
            return

        if is_cancel:
            await self._finalize_utterance_immediate(record, tool_called=False, tool_ok=None)
            await self._play_earcon("done")
            await self._transition(State.IDLE, "cancel_phrase")
            return

        self._active_utterance_record = record
        await self._transition(State.THINKING, "command_understood")
        self._spawn_command(
            text,
            wake_source=self._listening_wake_source,
            confidence=transcript.confidence,
            vad_end=vad_end,
            wake_ms=self._listening_wake_ms,
        )

    # -- FOLLOW_UP ----------------------------------------------------------

    async def _enter_follow_up(self) -> None:
        await self._transition(State.FOLLOW_UP, "await_follow_up")
        self._track_background(self._follow_up_loop())

    async def _follow_up_loop(self) -> None:
        deadline = time.monotonic() + self._config.conversation.follow_up_window_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await self._transition(State.IDLE, "follow_up_timeout")
                return

            utt = await self._wait_for_utterance(remaining)
            vad_end = time.monotonic()
            if self._state != State.FOLLOW_UP:
                return
            if utt is None:
                await self._transition(State.IDLE, "follow_up_timeout")
                return

            duration_s = len(utt.pcm) / _SAMPLE_RATE
            if duration_s < _FOLLOW_UP_MIN_DURATION_S:
                continue  # too short -- ignored, keep waiting out the window (§8A.2)

            transcript = await asyncio.to_thread(self._stt.transcribe, utt.pcm)
            text = transcript.text.strip()
            normalized = text.lower().rstrip(".!?")
            is_cancel = _matches_any_phrase(normalized, self._config.conversation.cancel_phrases)

            record: UtteranceRecord | None = None
            if self._config.memory.save_utterances:
                self._utterance_log.note_new_utterance(normalized, is_cancel=is_cancel)
                record = await self._capture_utterance(
                    utt.pcm,
                    text=text,
                    confidence=transcript.confidence,
                    wake_source="follow_up",
                )

            if transcript.confidence < _FOLLOW_UP_MIN_CONFIDENCE or not text:
                await self._finalize_utterance_immediate(record, tool_called=False, tool_ok=None)
                continue

            if _matches_any_phrase(normalized, self._config.conversation.end_phrases):
                await self._finalize_utterance_immediate(record, tool_called=False, tool_ok=None)
                await self._speak_and_wait(FOLLOW_UP_CLOSING)
                await self._transition(State.IDLE, "follow_up_end_phrase")
                return

            self._active_utterance_record = record
            await self._transition(State.THINKING, "follow_up_command")
            self._spawn_command(
                text, wake_source="follow_up", confidence=transcript.confidence, vad_end=vad_end
            )
            return

    # -- THINKING / SPEAKING: running one command ------------------------

    def _spawn_command(
        self,
        text: str,
        *,
        wake_source: str,
        confidence: float | None,
        vad_end: float,
        wake_ms: int | None = None,
    ) -> None:
        self._active_agent_task = self._track_background(
            self._run_command(
                text,
                wake_source=wake_source,
                confidence=confidence,
                vad_end=vad_end,
                wake_ms=wake_ms,
            )
        )

    async def _run_command(
        self,
        text: str,
        *,
        wake_source: str,
        confidence: float | None,
        vad_end: float,
        wake_ms: int | None,
    ) -> None:
        timer = RequestTimer(wake_source=wake_source, vad_end=vad_end)
        timer.stt_text = text
        if confidence is not None:
            timer.confidence = confidence
        if wake_ms is not None:
            timer.set_wake_ms(wake_ms)
        timer.path = (
            "fastpath"
            if (self._agent.fastpath_enabled and self._fastpath.match(text) is not None)
            else "llm"
        )
        self._active_timer = timer

        try:
            await self._consume_agent_stream(text, timer)
            await self._wait_for_playback_done()
            self._leave_speaking()
            self._on_request_success()
            if self._state in (State.THINKING, State.SPEAKING):
                await self._enter_follow_up()
        except asyncio.CancelledError:
            timer.error = "barge_in"
            raise
        except Exception as exc:  # noqa: BLE001 -- a request bug must not crash the orchestrator
            logger.exception("Buyruqni bajarishda xato")
            timer.error = str(exc)
            await self._on_handler_error()
        finally:
            record = timer.finalize()
            self._metrics.record(record)
            if self._tray is not None and record.total_to_first_audio_ms is not None:
                self._tray.set_last_latency(record.total_to_first_audio_ms / 1000)
            await self._finalize_command_utterance(timer)
            self._active_timer = None
            self._active_agent_task = None

    async def _finalize_command_utterance(self, timer: RequestTimer) -> None:
        """docs/tz/11_uzbek_stt_finetune.md §3.1: decides the just-finished
        command's utterance label. A successful tool call opens a 10s
        "silver" window (see ``UtteranceLog``); anything else (no tool,
        a failed tool, a barge-in/exception) is written as "unlabeled"
        right away -- there's nothing left to wait for."""
        utterance_record = self._active_utterance_record
        self._active_utterance_record = None
        if utterance_record is None:
            return
        tool_called = bool(timer.tools)
        if tool_called and timer.tool_ok and timer.error is None:
            pending = self._utterance_log.begin_silver_window(utterance_record)
            self._track_background(self._resolve_silver_window_after_delay(pending))
        else:
            await self._finalize_utterance_immediate(
                utterance_record,
                tool_called=tool_called,
                tool_ok=(timer.tool_ok if tool_called else None),
            )

    async def _consume_agent_stream(self, text: str, timer: RequestTimer) -> None:
        """docs/tz/08_orchestrator.md §8B.1: stream sentences to TTS as soon
        as they're ready; play "ack" if a tool runs >400 ms with nothing said
        yet, and a cached "Bir daqiqa..." if it runs >2 s."""
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _sentence_source() -> AsyncIterator[str]:
            while True:
                item = await sentence_queue.get()
                if item is None:
                    return
                yield item

        speak_task = (
            asyncio.ensure_future(self._tts.speak_stream(_sentence_source(), self._player))
            if self._tts is not None
            else None
        )

        said_anything = False
        ack_played = False
        wait_notice_sent = False
        tool_started_at: float | None = None
        first_event_seen = False

        agent_iter = self._agent.respond(text)
        next_task: asyncio.Task[Any] | None = None
        try:
            while True:
                next_task = asyncio.ensure_future(agent_iter.__anext__())
                while True:
                    done, _pending = await asyncio.wait({next_task}, timeout=_AGENT_POLL_S)
                    if next_task in done:
                        break
                    if tool_started_at is not None and not said_anything:
                        elapsed = time.monotonic() - tool_started_at
                        if not ack_played and elapsed >= _TOOL_ACK_DELAY_S:
                            await self._play_earcon("ack")
                            ack_played = True
                        if not wait_notice_sent and elapsed >= _TOOL_SLOW_NOTICE_S:
                            said_anything = True
                            wait_notice_sent = True
                            if speak_task is not None:
                                await sentence_queue.put(WAIT_A_MOMENT)
                            else:
                                await self._toast_fallback(WAIT_A_MOMENT)

                try:
                    event = next_task.result()
                except StopAsyncIteration:
                    break

                if not first_event_seen:
                    timer.mark_first_token()
                    first_event_seen = True

                if isinstance(event, SentenceEvent):
                    said_anything = True
                    if speak_task is not None:
                        await sentence_queue.put(event.text)
                    else:
                        await self._toast_fallback(event.text)
                    if self._state == State.THINKING:
                        # First real spoken sentence handed off for synthesis
                        # -- marked here (not via AudioPlayer's own
                        # ``on_playback_start`` callback) so an "ack"/"error"
                        # earcon played earlier in this same turn can't be
                        # mistaken for the response actually starting.
                        timer.mark_first_audio()
                        await self._transition(State.SPEAKING, "first_sentence_queued")
                        self._wake.set_speaking(True)
                elif isinstance(event, ToolStartEvent):
                    tool_started_at = time.monotonic()
                    ack_played = False
                    timer.add_tool(event.name)
                elif isinstance(event, ToolEndEvent):
                    tool_started_at = None
                    timer.mark_tool_result(event.result.ok)
                elif isinstance(event, ErrorEvent):
                    said_anything = True
                    timer.error = event.message
                    if speak_task is not None:
                        await sentence_queue.put(event.message)
                    else:
                        await self._toast_fallback(event.message)
                elif isinstance(event, DoneEvent):
                    break
        finally:
            # Barge-in cancels *this* coroutine, but ``asyncio.wait`` above
            # does not automatically cancel ``next_task`` just because the
            # wait on it was cancelled -- without this, the agent's in-flight
            # Ollama/tool call would keep running orphaned in the background
            # (docs/tz/08_orchestrator.md §8B.2: "agent_task.cancel() (Ollama
            # so'rovi ham to'xtaydi)").
            if next_task is not None and not next_task.done():
                next_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                    await next_task
            with contextlib.suppress(Exception):
                await agent_iter.aclose()
            await sentence_queue.put(None)
            if speak_task is not None:
                await speak_task

    # -- error handling (§8A.3) ------------------------------------------

    def _on_request_success(self) -> None:
        self._error_timestamps.clear()

    async def _on_handler_error(self) -> None:
        await self._play_earcon("error")
        await self._speak_and_wait(GENERIC_ERROR)
        await self._transition(State.IDLE, "handler_error")

        now = time.monotonic()
        self._error_timestamps = [t for t in self._error_timestamps if now - t < _ERROR_WINDOW_S]
        self._error_timestamps.append(now)
        if len(self._error_timestamps) < _ERROR_THRESHOLD:
            return

        restarted_recently = (
            self._last_soft_restart_at is not None
            and (now - self._last_soft_restart_at) < _ERROR_WINDOW_S
        )
        if restarted_recently:
            logger.critical(
                "Qayta yuklashdan keyin ham xatolar davom etmoqda, jarayon chiqadi (exit {})",
                _EXIT_UNRECOVERABLE,
            )
            await self.stop(_EXIT_UNRECOVERABLE)
            return

        logger.warning(
            "1 daqiqada {} ta xato, komponentlar qayta yuklanmoqda (soft restart)",
            len(self._error_timestamps),
        )
        self._error_timestamps.clear()
        self._last_soft_restart_at = now
        await self._soft_restart()

    async def _soft_restart(self) -> None:
        """Resets the wake detector/endpointer/agent-history state (§8A.3's
        "komponentlarni qayta yuklash"). The loaded STT/TTS models themselves
        are *not* reloaded: their own ``load()`` is already a no-op once
        loaded (docs/tz/03_stt.md, docs/tz/05_tts.md), and they're stateless,
        thread-safe, deterministic per call -- a run of repeated errors is
        far more likely to be Ollama, a tool bug, or bad input than a wedged
        model, which is exactly what resetting detector/agent state and
        re-probing Ollama addresses."""
        await self._transition(State.ERROR, "soft_restart")
        self._wake.reset()
        self._endpointer.reset()
        self._agent.reset_history()
        with contextlib.suppress(Exception):
            await self._llm_client.health()
        await self._transition(State.IDLE, "soft_restart_done")

    # -- reminders (§8B.4) ------------------------------------------------

    async def _on_reminder_due(self, event: ReminderDue) -> None:
        prefix = REMINDER_OVERDUE_PREFIX if event.overdue else REMINDER_PREFIX
        text = prefix + event.text
        if self._state in (State.IDLE, State.FOLLOW_UP):
            await self._speak_and_wait(text)
        else:
            self._pending_reminders.append(text)

    async def _flush_pending_reminders(self) -> None:
        while self._pending_reminders and self._state in (State.IDLE, State.FOLLOW_UP):
            text = self._pending_reminders.pop(0)
            await self._speak_and_wait(text)

    # -- device / health / tray hooks ------------------------------------

    async def _on_audio_device_lost(self, event: AudioDeviceLost) -> None:
        logger.error("Audio qurilma yo'qoldi: {}", event.detail)
        self._audio_device_down = True
        await self._transition(State.ERROR, "audio_device_lost")

    async def _on_ollama_down(self, event: OllamaDown) -> None:
        logger.warning("Ollama ishlamayapti: {}", event.detail)
        if self._tray is not None:
            self._tray.set_ollama_healthy(False)

    async def _on_ollama_restored(self, event: OllamaRestored) -> None:
        logger.info("Ollama tiklandi")
        if self._tray is not None:
            self._tray.set_ollama_healthy(True)

    async def _on_user_command(self, cmd: UserCommand) -> None:
        """docs/tz/09_autostart.md §3.4: everything the tray icon's menu can
        trigger."""
        if cmd.command == "mute":
            self._capture.mute()
            await self._transition(State.MUTED, "user_command_mute")
        elif cmd.command == "unmute":
            self._capture.unmute()
            await self._transition(State.IDLE, "user_command_unmute")
        elif cmd.command == "restart":
            # A *full* process restart (supervisor relaunches from scratch,
            # exit code 10, §3.2 step 8) -- deliberately not the lighter
            # in-process ``_soft_restart()`` that repeated-error recovery
            # uses (§8A.3): a user reaching for "restart" from the tray is
            # very likely trying to clear a problem that a mere state reset
            # (wake/endpointer/agent history) won't touch, e.g. a wedged
            # model or driver.
            await self.stop(10)
        elif cmd.command == "quit":
            await self.stop(0)
        elif cmd.command == "text_command":
            await self._on_text_command(cmd.text or "")
        elif cmd.command == "gpu_saver":
            await self._on_gpu_saver(True)
        elif cmd.command == "gpu_normal":
            await self._on_gpu_saver(False)

    async def _on_text_command(self, text: str) -> None:
        """docs/tz/09_autostart.md §3.4's "Matnli buyruq..." dialog -- treated
        exactly like an already-transcribed LISTENING utterance, minus the
        wake word. Ignored while a command is already running/being
        confirmed so it can't stomp on an in-flight turn."""
        text = text.strip()
        if not text:
            return
        if self._state not in (State.IDLE, State.FOLLOW_UP):
            logger.warning(
                "Tray matnli buyrug'i e'tiborsiz qoldirildi, band holat: {}", self._state
            )
            return
        await self._transition(State.THINKING, "tray_text_command")
        self._spawn_command(text, wake_source="tray_text", confidence=1.0, vad_end=time.monotonic())

    async def _on_gpu_saver(self, enabled: bool) -> None:
        """docs/tz/09_autostart.md §3.7 ("GPU'ni bo'shatish", optional):
        unloads the Ollama model from VRAM and stops the health monitor from
        silently reloading it. The STT-to-CPU half of §3.7 is deliberately
        NOT implemented -- ``WhisperEngine``'s device is fixed at
        construction (docs/tz/03_stt.md), so swapping it live would mean
        tearing down and reloading the very engine the wake detector also
        reuses (``_resolve_main_stt_engine``) while the pipeline keeps
        running; §3.7 itself marks the whole feature "ixtiyoriy, lekin
        tavsiya" (optional) and it isn't part of §5's acceptance criteria."""
        self._gpu_saver_active = enabled
        if enabled:
            logger.info("GPU'ni bo'shatish: Ollama modeli VRAM'dan tushirilmoqda")
            self._llm_client.set_keep_alive(0)
            await self._llm_client.unload()
        else:
            logger.info("Normal rejim: Ollama modeli qayta yuklanmoqda")
            self._llm_client.set_keep_alive(self._config.llm.keep_alive)
            await self._llm_client.ensure_loaded()

    # -- suspend / resume (docs/tz/09_autostart.md §3.5) -------------------

    async def _on_system_suspending(self, event: SystemSuspending) -> None:
        logger.info("Tizim uyquga ketmoqda")
        self._muted_before_suspend = self._state == State.MUTED
        self._capture.mute()
        await self._transition(State.MUTED, "system_suspend")

    async def _on_system_resumed(self, event: SystemResumed) -> None:
        logger.info("Tizim uyg'ondi, {:.0f} s kutilmoqda", _RESUME_SETTLE_S)
        await asyncio.sleep(_RESUME_SETTLE_S)

        if not self._muted_before_suspend:
            self._capture.unmute()

        with contextlib.suppress(Exception):
            healthy = await self._llm_client.health()
            if not healthy:
                logger.warning("Uyg'onishdan keyin Ollama javob bermayapti")

        # STT warmup: a real inference call (not just ``load()``, a no-op
        # once already loaded) to catch a CUDA context left wedged by the
        # suspend/resume cycle itself (§6 xavflar) before it can surface as
        # a confusing mid-command failure.
        try:
            warmup_pcm = np.zeros(_SAMPLE_RATE, dtype=np.int16)
            await asyncio.to_thread(self._stt.transcribe, warmup_pcm)
        except Exception:
            logger.exception(
                "Uyg'onishdan keyin STT warmup muvaffaqiyatsiz, jarayon qayta ishga tushadi"
            )
            await self.stop(10)
            return

        if not self._muted_before_suspend:
            await self._transition(State.IDLE, "system_resumed")


async def build_orchestrator(config: RootConfig, *, debug: bool = False) -> Orchestrator:
    """Construct every real component and wire them into an ``Orchestrator``
    (docs/tz/08_orchestrator.md §8A.1, steps 1-6; step 7's "Tayyorman"
    announcement happens in ``Orchestrator.run()`` instead, once tasks are
    about to start). Raises :class:`CriticalStartupError` if audio/STT/wake
    fail to load; TTS failing is non-critical (falls back to toasts)."""
    data_dir = Path(config.app.data_dir)
    bus = EventBus()

    # Registered before anything else so an overdue reminder that fires
    # during startup (rare, but ``ReminderScheduler.load_pending()`` starts
    # APScheduler as part of tool init below) is still forwarded once the
    # orchestrator exists -- see the module docstring on ``_forward_reminder``.
    orchestrator_ref: list[Orchestrator] = []

    async def _forward_reminder(event: ReminderDue) -> None:
        if orchestrator_ref:
            await orchestrator_ref[0]._on_reminder_due(event)  # noqa: SLF001

    reminder_tools.add_due_listener(_forward_reminder)

    ring_buffer = RingBuffer(config.audio.preroll_seconds + 1.0, config.audio.sample_rate)

    async def _load_critical() -> tuple[SileroVAD, WakeDetector, STTEngine]:
        vad = SileroVAD(default_model_path(data_dir))
        wake = create_wake_detector(config, ring_buffer=ring_buffer)
        await asyncio.to_thread(wake.load)
        stt = _resolve_main_stt_engine(wake)
        return vad, wake, stt

    async def _load_tts() -> TTSEngine | None:
        try:
            tts = MMSEngine(
                model_name=config.tts.model,
                model_script=config.tts.model_script,
                cache_dir=data_dir / "models" / "tts",
                speed=config.tts.speed,
                sentence_pause_ms=config.tts.sentence_pause_ms,
            )
            await asyncio.to_thread(tts.load)
            return tts
        except Exception as exc:  # noqa: BLE001 -- TTS is non-critical (§8A.1)
            logger.warning("TTS yuklanmadi, javoblar toast orqali ko'rsatiladi: {}", exc)
            return None

    async def _init_tools() -> None:
        configure_all(config)
        await async_init_all()
        await system_tools.ensure_app_index_scanned()

    try:
        (vad, wake, stt), tts, _unused = await asyncio.gather(
            _load_critical(), _load_tts(), _init_tools()
        )
    except Exception as exc:
        raise CriticalStartupError(str(exc)) from exc

    endpointer = Endpointer(
        ring_buffer,
        sample_rate=config.audio.sample_rate,
        threshold=config.vad.threshold,
        min_speech_ms=config.vad.min_speech_ms,
        end_silence_ms=config.vad.end_silence_ms,
        max_utterance_s=config.vad.max_utterance_s,
        listen_timeout_s=config.vad.listen_timeout_s,
    )

    capture = AudioCapture(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        block_size=config.audio.block_size,
        input_device=config.audio.input_device,
        on_device_lost=bus.publish,
    )
    player = AudioPlayer(
        sample_rate=config.audio.sample_rate,
        output_device=config.audio.output_device,
        output_volume=config.audio.output_volume,
        on_playback_start=lambda: bus.publish(PlaybackStarted(ts=time.time())),
        on_playback_end=lambda: bus.publish(PlaybackEnded(ts=time.time())),
    )

    llm_client = OllamaClient(config.llm)
    if not await llm_client.health():
        logger.warning(
            "Ollama serverga ulanib bo'lmadi yoki '{}' modeli topilmadi -- javoblar "
            "'Hozir o'ylay olmayapman' bilan tugaydi, toki server tiklanmaguncha",
            config.llm.model,
        )
    else:
        await llm_client.ensure_loaded()

    fastpath = FastPath(default_registry)
    agent = Agent(
        llm_client,
        default_registry,
        fastpath,
        config.llm,
        city=config.tools.weather.city,
        facts_provider=get_facts_for_prompt,
        fastpath_enabled=config.fastpath.enabled,
    )

    metrics = MetricsWriter(data_dir / "logs" / "metrics.jsonl")

    tray = TrayApp(config, bus)
    power_watcher = PowerWatcher(
        on_suspend=lambda: bus.publish(SystemSuspending(ts=time.time())),
        on_resume=lambda: bus.publish(SystemResumed(ts=time.time())),
    )
    hotkey_listener = None
    if config.autostart.hotkey_enabled:
        try:
            hotkey_listener = start_hotkey_listener(bus, config.autostart.hotkey)
        except Exception as exc:  # noqa: BLE001 -- optional (§3.4), must not block startup
            logger.warning("Global hotkey ishga tushmadi: {}", exc)

    companion = None
    if config.companion.enabled:
        try:
            companion = CompanionWidget(config, bus)
        except Exception as exc:  # noqa: BLE001 -- purely cosmetic, must not block startup
            logger.warning("Vizual hamroh tayyorlanmadi: {}", exc)

    orchestrator = Orchestrator(
        config,
        bus=bus,
        capture=capture,
        player=player,
        ring_buffer=ring_buffer,
        vad=vad,
        endpointer=endpointer,
        wake=wake,
        stt=stt,
        tts=tts,
        agent=agent,
        fastpath=fastpath,
        llm_client=llm_client,
        metrics=metrics,
        debug=debug,
        tray=tray,
        power_watcher=power_watcher,
        hotkey_listener=hotkey_listener,
        companion=companion,
    )
    orchestrator_ref.append(orchestrator)

    voice_confirm = VoiceConfirmationProvider(
        stt=stt,
        set_end_silence_ms=endpointer.set_end_silence_ms,
        default_end_silence_ms=config.vad.end_silence_ms,
        wait_for_utterance=orchestrator._wait_for_utterance,  # noqa: SLF001
        enter_confirming=orchestrator._enter_confirming,  # noqa: SLF001
        leave_confirming=orchestrator._leave_confirming,  # noqa: SLF001
        play_earcon=orchestrator._play_earcon,  # noqa: SLF001
        speak_and_wait=orchestrator._speak_and_wait,  # noqa: SLF001
        timeout_s=config.tools.confirm_timeout_s,
    )
    default_registry.confirmation = voice_confirm

    return orchestrator
