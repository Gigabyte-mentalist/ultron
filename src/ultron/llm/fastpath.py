"""LLM-free fast-path: ~15 common intents answered by regex alone
(docs/tz/06_llm.md §3.5).

Only a full pattern match ("to'liq regex moslik") counts -- anything even
slightly off falls through to the LLM agent loop, which is why this uses
plain ``re.match`` against the whole normalized utterance rather than fuzzy
scoring. If a matched intent's target tool isn't registered (e.g.
``get_weather`` when ``internet_tools_enabled=False``), ``match()`` returns
``None`` and the caller falls through to the LLM (``tool_name not in
registry``), per the TZ-06 scope note ("registry'da yo'q intent o'chirilgan
hisoblanadi"). The tool/argument names used here (``delta``, ``level``,
``name``, ``query``, ``muted``) are taken directly from the table in
docs/tz/06_llm.md §3.5 where given; ``mute``'s ``muted`` argument name is
this module's own choice (the TZ-06 table only showed ``mute(True/False)``),
kept because it matches ``tools/system_tools.py``'s actual signature
(docs/tz/07_tools.md §7A.2) -- reconciled per that TZ's own note. The
generic "X ni och" pattern also matched folder references ("Yuklamalar
papkasini och") and bare site names/domains ("kun.uz ni och") until
docs/tz/07_tools.md added ``open_folder``/``open_url``; ``is_folder_reference()``
and ``is_url_reference()`` route those to the right tool instead of ``open_app``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ultron.text.numbers_uz import words_to_number
from ultron.tools.base import ToolRegistry
from ultron.tools.browser_tools import is_url_reference
from ultron.tools.file_tools import is_folder_reference

_VOLUME_STEP = 10
_BRIGHTNESS_STEP = 10

_TIME_RE = re.compile(r"^(soat nechi|soat necha bo'ldi|nima vaqt)\??$")
_DATE_RE = re.compile(r"^(bugun (nechanchi sana|necha(nchi)? sana)|qaysi sana(dir)?)\??$")
_LOCK_RE = re.compile(r"^(kompyuterni|ekranni) qulfla$")
_SCREENSHOT_RE = re.compile(r"^(skrinshot ol|ekranni rasmga ol)$")
_MUTE_RE = re.compile(r"^ovozni o'chir$")
_UNMUTE_RE = re.compile(r"^ovozni yoq$")
_VOLUME_SET_RE = re.compile(r"^ovoz(ni)? (.+?)( foiz)? qil$|^ovoz (\d{1,3})$")
_VOLUME_UP_RE = re.compile(r"^ovozni (balandlat|ko'tar|oshir|qo'sh)$")
_VOLUME_DOWN_RE = re.compile(r"^ovozni (pasaytir|kamaytir|tushir)$")
_BRIGHTNESS_SET_RE = re.compile(r"^yorug'lik(ni)? (.+?)( foiz)? qil$|^yorug'lik (\d{1,3})$")
_BRIGHTNESS_UP_RE = re.compile(r"^yorug'likni (oshir|ko'tar)$")
_BRIGHTNESS_DOWN_RE = re.compile(r"^yorug'likni (kamaytir|pasaytir)$")
_YOUTUBE_RE = re.compile(r"^youtube'?da (.+?) (qo'y|och)$")
_CLOSE_APP_RE = re.compile(r"^(.+?)ni yop$")
_OPEN_APP_RE = re.compile(r"^(.+?)ni (och|ishga tushir)$|^(.+?) och$")


@dataclass
class FastPathMatch:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)


def _percent_from_text(text: str) -> int | None:
    digit_match = re.search(r"\d+", text)
    if digit_match:
        return int(digit_match.group())
    return words_to_number(text.strip())


class FastPath:
    """Matches normalized text against the known intent table (docs/tz/06_llm.md §3.5)."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def match(self, text: str) -> FastPathMatch | None:
        normalized = text.strip().lower().rstrip(".!")
        if not normalized:
            return None

        candidate = self._match_intent(normalized)
        if candidate is None or candidate.tool_name not in self._registry:
            return None
        return candidate

    def _match_intent(self, text: str) -> FastPathMatch | None:  # noqa: PLR0911
        if _TIME_RE.match(text):
            return FastPathMatch("get_time")
        if _DATE_RE.match(text):
            return FastPathMatch("get_date")
        if _LOCK_RE.match(text):
            return FastPathMatch("lock_screen")
        if _SCREENSHOT_RE.match(text):
            return FastPathMatch("screenshot")
        if _MUTE_RE.match(text):
            return FastPathMatch("mute", {"muted": True})
        if _UNMUTE_RE.match(text):
            return FastPathMatch("mute", {"muted": False})

        match = _VOLUME_SET_RE.match(text)
        if match:
            level = _percent_from_text(match.group(2) or match.group(4) or "")
            return FastPathMatch("set_volume", {"level": level}) if level is not None else None
        if _VOLUME_UP_RE.match(text):
            return FastPathMatch("set_volume", {"delta": _VOLUME_STEP})
        if _VOLUME_DOWN_RE.match(text):
            return FastPathMatch("set_volume", {"delta": -_VOLUME_STEP})

        match = _BRIGHTNESS_SET_RE.match(text)
        if match:
            level = _percent_from_text(match.group(2) or match.group(4) or "")
            return FastPathMatch("set_brightness", {"level": level}) if level is not None else None
        if _BRIGHTNESS_UP_RE.match(text):
            return FastPathMatch("set_brightness", {"delta": _BRIGHTNESS_STEP})
        if _BRIGHTNESS_DOWN_RE.match(text):
            return FastPathMatch("set_brightness", {"delta": -_BRIGHTNESS_STEP})

        match = _YOUTUBE_RE.match(text)
        if match:
            return FastPathMatch("youtube_search", {"query": match.group(1).strip()})

        match = _CLOSE_APP_RE.match(text)
        if match:
            return FastPathMatch("close_app", {"name": match.group(1).strip()})

        match = _OPEN_APP_RE.match(text)
        if match:
            name = (match.group(1) or match.group(3) or "").strip()
            if name:
                if is_url_reference(name):
                    return FastPathMatch("open_url", {"url": name})
                if is_folder_reference(name):
                    return FastPathMatch("open_folder", {"name_or_path": name})
                return FastPathMatch("open_app", {"name": name})

        return None
