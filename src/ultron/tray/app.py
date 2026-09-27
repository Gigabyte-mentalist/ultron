"""Tray icon, menu, and typed-command dialog (docs/tz/09_autostart.md §3.4).

``TrayApp`` never touches the orchestrator directly -- every action it takes
(mute, restart, quit, a typed command, GPU-saver toggle) is published onto
the shared ``EventBus`` as a ``UserCommand``, the same loose coupling
``core.orchestrator`` already uses for ``WakeEvent``/``OllamaDown``/etc (see
``events.py``'s module docstring). ``core.orchestrator`` calls
``set_state``/``set_ollama_healthy``/``set_last_latency`` the other way, to
keep the icon's color and tooltip current.

pystray's Win32 backend runs its message loop on its own thread via
``Icon.run_detached()`` -- this process's actual main thread is owned by
``asyncio.run()`` (``__main__.py``), so ``run()`` (which the library's
docstring says "must be called from the main thread") is never an option
here; ``run_detached()`` is the library's own documented way to integrate
with a foreign event loop and, on the Win32 backend, does exactly what a
manual ``threading.Thread(target=icon.run)`` would.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image, ImageDraw

from ultron.config import RootConfig
from ultron.core.state import State
from ultron.events import EventBus, UserCommand
from ultron.i18n.uz import (
    TRAY_COMPANION_HIDE,
    TRAY_COMPANION_SHOW,
    TRAY_GPU_SAVER_OFF,
    TRAY_GPU_SAVER_ON,
    TRAY_MUTE_MIC,
    TRAY_OPEN_CONFIG,
    TRAY_OPEN_LOGS,
    TRAY_QUIT,
    TRAY_RESTART,
    TRAY_TEXT_COMMAND,
    TRAY_TEXT_COMMAND_LABEL,
    TRAY_TEXT_COMMAND_SUBMIT,
    TRAY_TEXT_COMMAND_TITLE,
    TRAY_UNMUTE_MIC,
)

# tray/app.py -> tray -> ultron -> src -> project root (matches __main__.py's
# own "-- check" resolution: Path(__file__).resolve().parent.parent.parent).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ICON_SIZE = 64

# docs/tz/09_autostart.md §3.4's color legend. FOLLOW_UP/CONFIRMING aren't
# separately listed there; mapped to the closest state they behave like
# (still passively listening / actively awaiting a spoken answer).
_STATE_COLORS: dict[State, tuple[int, int, int]] = {
    State.IDLE: (110, 170, 130),
    State.LISTENING: (60, 130, 240),
    State.THINKING: (150, 90, 220),
    State.SPEAKING: (150, 90, 220),
    State.FOLLOW_UP: (110, 170, 130),
    State.CONFIRMING: (60, 130, 240),
    State.MUTED: (150, 150, 150),
    State.ERROR: (220, 60, 60),
}

_STATE_LABELS_UZ: dict[State, str] = {
    State.IDLE: "tinglayapti",
    State.LISTENING: "tinglayapti",
    State.THINKING: "o'ylayapti",
    State.SPEAKING: "gapiryapti",
    State.FOLLOW_UP: "tinglayapti",
    State.CONFIRMING: "tasdiq kutmoqda",
    State.MUTED: "mikrofon o'chirilgan",
    State.ERROR: "xatolik",
}

_ERROR_COLOR = _STATE_COLORS[State.ERROR]


def _make_icon_image(color: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse([margin, margin, _ICON_SIZE - margin, _ICON_SIZE - margin], fill=(*color, 255))
    return img


class TrayApp:
    def __init__(self, config: RootConfig, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self._muted = False
        self._gpu_saver = False
        self._ollama_healthy = True
        self._state = State.IDLE
        self._companion_visible = True
        self._last_latency_s: float | None = None
        self._icon: Any = None

    # -- lifecycle, called from the orchestrator's asyncio thread -----------

    def start(self) -> None:
        import pystray

        self._icon = pystray.Icon(
            "Ultron",
            icon=_make_icon_image(_STATE_COLORS[State.IDLE]),
            title=self._tooltip(),
            menu=self._build_menu(),
        )
        self._icon.run_detached()

    def stop(self) -> None:
        if self._icon is not None:
            with contextlib.suppress(Exception):
                self._icon.stop()
            self._icon = None

    # -- orchestrator -> tray -------------------------------------------

    def set_state(self, state: State) -> None:
        self._state = state
        self._refresh()

    def set_ollama_healthy(self, healthy: bool) -> None:
        self._ollama_healthy = healthy
        self._refresh()

    def set_last_latency(self, seconds: float) -> None:
        self._last_latency_s = seconds
        self._refresh()

    def _refresh(self) -> None:
        if self._icon is None:
            return
        color = (
            _ERROR_COLOR
            if not self._ollama_healthy
            else _STATE_COLORS.get(self._state, _ERROR_COLOR)
        )
        with contextlib.suppress(Exception):
            self._icon.icon = _make_icon_image(color)
            self._icon.title = self._tooltip()

    def _tooltip(self) -> str:
        label = _STATE_LABELS_UZ.get(self._state, self._state.value)
        if not self._ollama_healthy:
            label = "miyam ishlamayapti"
        if self._last_latency_s is not None:
            return f"Ultron — {label} | oxirgi javob {self._last_latency_s:.1f} s"
        return f"Ultron — {label}"

    # -- menu -------------------------------------------------------------

    def _build_menu(self) -> Any:
        import pystray

        return pystray.Menu(
            pystray.MenuItem(self._mute_label, self._on_toggle_mute),
            pystray.MenuItem(TRAY_TEXT_COMMAND, self._on_text_command),
            pystray.MenuItem(TRAY_OPEN_LOGS, self._on_open_logs),
            pystray.MenuItem(TRAY_OPEN_CONFIG, self._on_open_config),
            pystray.MenuItem(self._gpu_saver_label, self._on_toggle_gpu_saver),
            pystray.MenuItem(self._companion_label, self._on_toggle_companion),
            pystray.MenuItem(TRAY_RESTART, self._on_restart),
            pystray.MenuItem(TRAY_QUIT, self._on_quit),
        )

    def _mute_label(self, _item: Any) -> str:
        return TRAY_UNMUTE_MIC if self._muted else TRAY_MUTE_MIC

    def _gpu_saver_label(self, _item: Any) -> str:
        return TRAY_GPU_SAVER_OFF if self._gpu_saver else TRAY_GPU_SAVER_ON

    def _companion_label(self, _item: Any) -> str:
        return TRAY_COMPANION_HIDE if self._companion_visible else TRAY_COMPANION_SHOW

    # -- menu callbacks (run on pystray's own thread) ----------------------

    def _on_toggle_mute(self, _icon: Any, _item: Any) -> None:
        self._muted = not self._muted
        self._bus.publish(UserCommand(command="mute" if self._muted else "unmute"))

    def _on_toggle_gpu_saver(self, _icon: Any, _item: Any) -> None:
        self._gpu_saver = not self._gpu_saver
        self._bus.publish(UserCommand(command="gpu_saver" if self._gpu_saver else "gpu_normal"))

    def _on_toggle_companion(self, _icon: Any, _item: Any) -> None:
        # ultron.companion.widget.CompanionWidget consumes this itself,
        # straight off the bus -- the orchestrator's own UserCommand handler
        # has no case for it and just ignores it, by design (see that
        # widget's module docstring).
        self._companion_visible = not self._companion_visible
        self._bus.publish(UserCommand(command="companion_toggle"))

    def _on_restart(self, _icon: Any, _item: Any) -> None:
        self._bus.publish(UserCommand(command="restart"))

    def _on_quit(self, _icon: Any, _item: Any) -> None:
        self._bus.publish(UserCommand(command="quit"))

    def _on_open_logs(self, _icon: Any, _item: Any) -> None:
        path = Path(self._config.app.data_dir) / "logs"
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as exc:
            logger.warning("Log papkasini ochib bo'lmadi: {}", exc)

    def _on_open_config(self, _icon: Any, _item: Any) -> None:
        path = _PROJECT_ROOT / "config" / "config.yaml"
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as exc:
            logger.warning("Config faylini ochib bo'lmadi: {}", exc)

    def _on_text_command(self, _icon: Any, _item: Any) -> None:
        threading.Thread(
            target=self._text_command_dialog, name="TrayTextCommand", daemon=True
        ).start()

    def _text_command_dialog(self) -> None:
        result: list[str] = []

        def _submit(_event: Any = None) -> None:
            result.append(entry.get())
            root.destroy()

        try:
            root = tk.Tk()
            root.title(TRAY_TEXT_COMMAND_TITLE)
            root.attributes("-topmost", True)
            root.resizable(False, False)
            tk.Label(root, text=TRAY_TEXT_COMMAND_LABEL).pack(padx=10, pady=(10, 0))
            entry = tk.Entry(root, width=50)
            entry.pack(padx=10, pady=10)
            entry.focus_force()
            entry.bind("<Return>", _submit)
            tk.Button(root, text=TRAY_TEXT_COMMAND_SUBMIT, command=_submit).pack(pady=(0, 10))
            root.protocol("WM_DELETE_WINDOW", root.destroy)
            root.mainloop()
        except Exception:  # noqa: BLE001 -- a broken dialog must not take the tray down
            logger.exception("Matnli buyruq oynasida xato")
            return

        text = result[0].strip() if result else ""
        if text:
            self._bus.publish(UserCommand(command="text_command", text=text))


def start_hotkey_listener(bus: EventBus, hotkey: str) -> Any:
    """docs/tz/09_autostart.md §3.4: an optional global ``Ctrl+Alt+U`` (config-
    driven) fallback for when the wake word itself fails to trigger. Publishes
    a synthetic ``WakeEvent`` (``source="hotkey"``, no trailing text -- always
    goes through LISTENING, never treated as a full "wake + command" combo).
    Returns the running listener so the caller can ``.stop()`` it at shutdown;
    ``None`` if ``pynput`` isn't available."""
    from pynput import keyboard

    from ultron.events import WakeEvent

    def _on_activate() -> None:
        logger.info("Global hotkey ({}) bosildi", hotkey)
        bus.publish(WakeEvent(ts=time.time(), score=1.0, source="hotkey", trailing_text=None))

    listener = keyboard.GlobalHotKeys({hotkey: _on_activate})
    listener.start()
    return listener
