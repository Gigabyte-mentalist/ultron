"""Visual companion overlay -- not part of any TZ, added after MVP+TZ-10/11
directly at the user's request: a small always-on-top face widget (see
``companion.face``) whose eyes glow while Ultron is speaking, purely for
visual/emotional feedback.

Lives entirely outside the orchestrator's FSM: it subscribes to the same
``EventBus`` every other loosely-coupled consumer uses. ``events.py``'s own
module docstring anticipated exactly this -- "``PlaybackStarted``/
``PlaybackEnded`` are ... ready for a consumer that only cares about raw
playback activity regardless of content (e.g. a future tray icon
animation)". ``core.orchestrator`` only holds this long enough to call
``start()``/``stop()`` at the right lifecycle points, exactly like
``tray.app.TrayApp``/``core.power_watcher.PowerWatcher`` -- it never sees a
``companion_toggle`` ``UserCommand`` do anything (there's no ``elif`` for it
in ``_on_user_command``), because this widget consumes that event itself,
straight off the bus.

Tkinter's mainloop owns its own dedicated thread (the same pattern
``TrayApp``'s "Matnli buyruq..." dialog already uses); events cross from the
asyncio side via a plain thread-safe ``queue.Queue``, drained from inside
that mainloop's own ``after()`` timer -- the standard safe way to touch a Tk
widget from another thread is to never touch it directly at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from typing import Any

from loguru import logger
from PIL import ImageTk

from ultron.companion.face import idle_frames, speaking_frames
from ultron.config import RootConfig
from ultron.events import EventBus, PlaybackEnded, PlaybackStarted, UserCommand

_IDLE_FRAME_MS = 90
_SPEAKING_FRAME_MS = 60

_Placer = Callable[[int, int, int, int, int], tuple[int, int]]
_POSITIONS: dict[str, _Placer] = {
    "bottom_right": lambda sw, sh, w, h, m: (sw - w - m, sh - h - m),
    "bottom_left": lambda sw, sh, w, h, m: (m, sh - h - m),
    "top_right": lambda sw, sh, w, h, m: (sw - w - m, m),
    "top_left": lambda sw, sh, w, h, m: (m, m),
}


class CompanionWidget:
    def __init__(self, config: RootConfig, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self._tk_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._consumer_task: asyncio.Task[Any] | None = None

    # -- lifecycle, called from the orchestrator's asyncio thread -----------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_tk, name="CompanionWidget", daemon=True)
        self._thread.start()
        self._consumer_task = asyncio.ensure_future(self._consume_events())

    def stop(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            self._consumer_task = None
        self._tk_queue.put(("quit", None))
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- asyncio side: EventBus -> thread-safe queue ------------------------

    async def _consume_events(self) -> None:
        async def _pump_started() -> None:
            async for _event in self._bus.subscribe(PlaybackStarted):
                self._tk_queue.put(("speaking", True))

        async def _pump_ended() -> None:
            async for _event in self._bus.subscribe(PlaybackEnded):
                self._tk_queue.put(("speaking", False))

        async def _pump_commands() -> None:
            async for cmd in self._bus.subscribe(UserCommand):
                if cmd.command == "companion_toggle":
                    self._tk_queue.put(("toggle_visibility", None))

        await asyncio.gather(_pump_started(), _pump_ended(), _pump_commands())

    # -- Tk thread ------------------------------------------------------

    def _run_tk(self) -> None:
        try:
            self._build_and_run()
        except Exception:  # noqa: BLE001 -- a cosmetic widget must never take the process down
            logger.exception("Vizual hamroh oynasida xato")

    def _build_and_run(self) -> None:
        size = self._config.companion.size
        idle_photos_src = idle_frames(size)
        speaking_photos_src = speaking_frames(size)

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg="black")
        with contextlib.suppress(tk.TclError):
            # Windows-only color-key transparency -- punches the head
            # polygon's transparent (alpha=0) surroundings through to the
            # desktop instead of showing a black square.
            root.wm_attributes("-transparentcolor", "black")

        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        place = _POSITIONS.get(self._config.companion.position, _POSITIONS["bottom_right"])
        x, y = place(screen_w, screen_h, size, size, self._config.companion.margin)
        root.geometry(f"{size}x{size}+{x}+{y}")

        # Must be created only after `root` exists -- PhotoImage needs a live
        # Tcl interpreter, hence rendering the PIL frames themselves happens
        # earlier (thread-safe, no Tk involved) and only the Tk-side wrapping
        # happens here.
        idle_photos = [ImageTk.PhotoImage(f) for f in idle_photos_src]
        speaking_photos = [ImageTk.PhotoImage(f) for f in speaking_photos_src]

        label = tk.Label(root, bd=0, bg="black")
        label.pack()

        state = {"speaking": False, "frame": 0, "visible": True}

        def poll() -> None:
            try:
                while True:
                    kind, value = self._tk_queue.get_nowait()
                    if kind == "speaking":
                        state["speaking"] = value
                    elif kind == "toggle_visibility":
                        state["visible"] = not state["visible"]
                        (root.deiconify() if state["visible"] else root.withdraw())
                    elif kind == "quit":
                        root.quit()
                        return
            except queue.Empty:
                pass

            photos = speaking_photos if state["speaking"] else idle_photos
            state["frame"] = (state["frame"] + 1) % len(photos)
            photo = photos[state["frame"]]
            label.configure(image=photo)
            label.image = photo  # type: ignore[attr-defined]  # keep a reference alive
            delay = _SPEAKING_FRAME_MS if state["speaking"] else _IDLE_FRAME_MS
            root.after(delay, poll)

        root.after(0, poll)
        root.mainloop()
        with contextlib.suppress(Exception):
            root.destroy()
