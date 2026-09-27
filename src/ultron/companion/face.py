"""Procedural "Altron" face rendering for the visual companion widget.

Not a copy of Marvel's Ultron artwork (that's someone else's copyrighted
character design) -- a small, original, angular gunmetal head with glowing
red eye slits, drawn entirely with ``PIL.ImageDraw`` primitives, in the same
spirit ``ultron.tray.app``'s tray icon is already procedurally generated
rather than loaded from an asset file.

``render_face(size, glow)`` draws one frame; ``idle_frames``/``speaking_frames``
each precompute a full pulse cycle (a slow, dim "breathing" glow at rest, a
faster, brighter pulse while Ultron is actually speaking) so
``companion.widget.CompanionWidget`` only ever has to cycle through
already-rendered frames, never render on the Tk thread's own timer tick.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFilter

_HEAD_COLOR = (32, 32, 38, 255)
_OUTLINE_COLOR = (58, 58, 66, 255)
_SEAM_COLOR = (52, 52, 60, 200)
_BROW_COLOR = (48, 48, 55, 180)
_GRILLE_COLOR = (18, 18, 22, 220)
_EYE_DIM = (90, 8, 8, 255)
_EYE_BRIGHT = (255, 60, 40, 255)


def _head_polygon(size: int) -> list[tuple[float, float]]:
    return [
        (0.28 * size, 0.05 * size),
        (0.72 * size, 0.05 * size),
        (0.92 * size, 0.32 * size),
        (0.80 * size, 0.86 * size),
        (0.60 * size, 0.98 * size),
        (0.40 * size, 0.98 * size),
        (0.20 * size, 0.86 * size),
        (0.08 * size, 0.32 * size),
    ]


def _eye_polygon(size: int, *, side: str) -> list[tuple[float, float]]:
    """An angled slit, narrower and higher at the outer edge -- a menacing,
    frowning tilt rather than a neutral horizontal eye."""
    outer_top = 0.37 * size
    inner_top = 0.42 * size
    height = 0.075 * size
    if side == "left":
        outer_x, inner_x = 0.24 * size, 0.47 * size
    else:
        outer_x, inner_x = 0.76 * size, 0.53 * size
    return [
        (outer_x, outer_top),
        (inner_x, inner_top),
        (inner_x, inner_top + height),
        (outer_x, outer_top + height * 1.3),
    ]


def _lerp_color(dim: tuple[int, int, int, int], bright: tuple[int, int, int, int], t: float):
    return tuple(int(d + (b - d) * t) for d, b in zip(dim, bright, strict=True))


def _draw_eye(
    img: Image.Image, draw: ImageDraw.ImageDraw, size: int, glow: float, *, side: str
) -> None:
    points = _eye_polygon(size, side=side)
    core_color = _lerp_color(_EYE_DIM, _EYE_BRIGHT, glow)

    if glow > 0.05:
        glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)
        glow_draw.polygon(points, fill=(255, 30, 20, int(180 * glow)))
        blur_radius = max(1, round(size * 0.05 * glow))
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
        img.alpha_composite(glow_layer)

    draw.polygon(points, fill=core_color)  # crisp core on top of the soft glow


def render_face(size: int, glow: float) -> Image.Image:
    """Renders one frame. ``glow`` is the eye brightness, 0.0 (near-off) to
    1.0 (fully lit) -- callers pick it from wherever they are in a pulse
    cycle, see ``idle_frames``/``speaking_frames``."""
    glow = max(0.0, min(1.0, glow))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.polygon(
        _head_polygon(size), fill=_HEAD_COLOR, outline=_OUTLINE_COLOR, width=max(2, size // 70)
    )
    draw.line(
        [(0.5 * size, 0.07 * size), (0.5 * size, 0.96 * size)],
        fill=_SEAM_COLOR,
        width=max(1, size // 110),
    )
    draw.line(
        [(0.14 * size, 0.33 * size), (0.86 * size, 0.33 * size)],
        fill=_BROW_COLOR,
        width=max(1, size // 130),
    )
    for i in range(3):
        y = (0.72 + i * 0.05) * size
        draw.line(
            [(0.34 * size, y), (0.66 * size, y)], fill=_GRILLE_COLOR, width=max(1, size // 100)
        )

    _draw_eye(img, draw, size, glow, side="left")
    _draw_eye(img, draw, size, glow, side="right")
    return img


def idle_frames(size: int, count: int = 24) -> list[Image.Image]:
    """A slow, dim "breathing" pulse (~2s cycle at the widget's frame delay)."""
    return [
        render_face(size, 0.05 + 0.10 * (0.5 + 0.5 * math.sin(2 * math.pi * i / count)))
        for i in range(count)
    ]


def speaking_frames(size: int, count: int = 14) -> list[Image.Image]:
    """A faster, brighter pulse while Ultron is actually talking."""
    return [
        render_face(size, 0.55 + 0.45 * (0.5 + 0.5 * math.sin(2 * math.pi * i / count)))
        for i in range(count)
    ]
