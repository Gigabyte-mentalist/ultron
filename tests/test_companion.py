"""Tests for ultron.companion (visual companion overlay -- not part of any
TZ, added directly at the user's request; see companion/widget.py's module
docstring for why it lives entirely outside the orchestrator's FSM).

``face.py`` is pure ``PIL`` rendering, fully testable headlessly.
``CompanionWidget``'s bus-consumption logic is tested the same way; actually
opening a Tk window is exercised manually (screenshot), not here, matching
how ``tray.app.TrayApp``'s real pystray icon isn't unit-tested either.
"""

from __future__ import annotations

import asyncio
import contextlib

from ultron.companion.face import idle_frames, render_face, speaking_frames
from ultron.companion.widget import CompanionWidget
from ultron.config import RootConfig
from ultron.events import EventBus, PlaybackEnded, PlaybackStarted, UserCommand

# The eye slits always cover this point at every tested size (see
# face._eye_polygon's fractions) -- used to sample eye brightness.
_LEFT_EYE_SAMPLE = (0.35, 0.40)


def _sample(img, size: int, frac_xy: tuple[float, float]) -> tuple[int, int, int, int]:
    x, y = int(frac_xy[0] * size), int(frac_xy[1] * size)
    return img.getpixel((x, y))


def test_render_face_returns_rgba_of_requested_size() -> None:
    img = render_face(128, 0.5)
    assert img.size == (128, 128)
    assert img.mode == "RGBA"


def test_render_face_glow_controls_eye_brightness() -> None:
    dim = render_face(200, 0.0)
    bright = render_face(200, 1.0)

    dim_r = _sample(dim, 200, _LEFT_EYE_SAMPLE)[0]
    bright_r = _sample(bright, 200, _LEFT_EYE_SAMPLE)[0]
    assert bright_r > dim_r


def test_render_face_clamps_out_of_range_glow() -> None:
    # must not raise, and must clamp rather than produce invalid color math
    low = render_face(64, -5.0)
    high = render_face(64, 5.0)
    assert low.size == high.size == (64, 64)


def test_idle_and_speaking_frames_counts_and_relative_brightness() -> None:
    idle = idle_frames(120, count=8)
    speaking = speaking_frames(120, count=8)
    assert len(idle) == 8
    assert len(speaking) == 8

    idle_avg = sum(_sample(f, 120, _LEFT_EYE_SAMPLE)[0] for f in idle) / len(idle)
    speaking_avg = sum(_sample(f, 120, _LEFT_EYE_SAMPLE)[0] for f in speaking) / len(speaking)
    assert speaking_avg > idle_avg  # speaking pulse is brighter on average than idle breathing


async def test_consume_events_forwards_playback_started_and_ended() -> None:
    bus = EventBus()
    bus.bind_loop()
    widget = CompanionWidget(RootConfig(), bus)

    task = asyncio.ensure_future(widget._consume_events())  # noqa: SLF001
    await asyncio.sleep(0)  # let the pump coroutines reach their subscribe() calls

    bus.publish(PlaybackStarted(ts=0.0))
    bus.publish(PlaybackEnded(ts=0.0))
    await asyncio.sleep(0.05)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    items = []
    while not widget._tk_queue.empty():  # noqa: SLF001
        items.append(widget._tk_queue.get_nowait())  # noqa: SLF001
    assert items == [("speaking", True), ("speaking", False)]


async def test_consume_events_forwards_companion_toggle_only() -> None:
    bus = EventBus()
    bus.bind_loop()
    widget = CompanionWidget(RootConfig(), bus)

    task = asyncio.ensure_future(widget._consume_events())  # noqa: SLF001
    await asyncio.sleep(0)

    bus.publish(UserCommand(command="companion_toggle"))
    bus.publish(UserCommand(command="mute"))  # must be ignored by this widget
    await asyncio.sleep(0.05)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    items = []
    while not widget._tk_queue.empty():  # noqa: SLF001
        items.append(widget._tk_queue.get_nowait())  # noqa: SLF001
    assert items == [("toggle_visibility", None)]


def test_stop_without_start_does_not_raise() -> None:
    widget = CompanionWidget(RootConfig(), EventBus())
    widget.stop()  # no thread/task was ever started
