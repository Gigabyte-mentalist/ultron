"""Tests for ultron.llm.fastpath.FastPath (docs/tz/06_llm.md §3.5, §4).

40+ phrases across every intent in the §3.5 table map to the expected tool
and args; ambiguous/unrelated phrases resolve to ``None``. Cases whose tool
isn't registered (everything except ``get_time``/``get_date`` until
docs/tz/07_tools.md) are asserted to fall through to ``None`` too, per the
TZ-06 scope note.
"""

from __future__ import annotations

import pytest

from ultron.llm.fastpath import FastPath
from ultron.tools.base import Tool, ToolRegistry, ToolResult, _build_arg_model, _build_schema


async def _noop() -> ToolResult:
    return ToolResult(ok=True)


def _stub_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description="stub",
        dangerous=False,
        func=_noop,
        schema=_build_schema(_noop),
        arg_model=_build_arg_model(_noop),
    )


@pytest.fixture
def registry_with_demo_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_stub_tool("get_time"))
    registry.register(_stub_tool("get_date"))
    return registry


@pytest.fixture
def full_registry() -> ToolRegistry:
    """Every intent's target tool registered, for testing the matching logic
    itself independent of the TZ-06 registry-gating behavior."""
    registry = ToolRegistry()
    for name in (
        "get_time",
        "get_date",
        "lock_screen",
        "screenshot",
        "mute",
        "set_volume",
        "set_brightness",
        "youtube_search",
        "close_app",
        "open_app",
        "open_folder",
        "open_url",
    ):
        registry.register(_stub_tool(name))
    return registry


_MATCH_CASES = [
    ("Soat nechi?", "get_time", {}),
    ("soat nechi", "get_time", {}),
    ("Nima vaqt?", "get_time", {}),
    ("Bugun nechanchi sana?", "get_date", {}),
    ("bugun nechanchi sana", "get_date", {}),
    ("qaysi sana", "get_date", {}),
    ("Kompyuterni qulfla", "lock_screen", {}),
    ("ekranni qulfla", "lock_screen", {}),
    ("Skrinshot ol", "screenshot", {}),
    ("ekranni rasmga ol", "screenshot", {}),
    ("Ovozni o'chir", "mute", {"muted": True}),
    ("ovozni yoq", "mute", {"muted": False}),
    ("Ovozni balandlat", "set_volume", {"delta": 10}),
    ("ovozni ko'tar", "set_volume", {"delta": 10}),
    ("ovozni oshir", "set_volume", {"delta": 10}),
    ("ovozni qo'sh", "set_volume", {"delta": 10}),
    ("Ovozni pasaytir", "set_volume", {"delta": -10}),
    ("ovozni kamaytir", "set_volume", {"delta": -10}),
    ("ovozni tushir", "set_volume", {"delta": -10}),
    ("Ovozni ellik foiz qil", "set_volume", {"level": 50}),
    ("ovoz 30", "set_volume", {"level": 30}),
    ("ovozni o'ttiz besh foiz qil", "set_volume", {"level": 35}),
    ("Yorug'likni oshir", "set_brightness", {"delta": 10}),
    ("yorug'likni ko'tar", "set_brightness", {"delta": 10}),
    ("Yorug'likni kamaytir", "set_brightness", {"delta": -10}),
    ("yorug'likni pasaytir", "set_brightness", {"delta": -10}),
    ("yorug'lik 70", "set_brightness", {"level": 70}),
    ("Yorug'likni yetmish foiz qil", "set_brightness", {"level": 70}),
    ("YouTube'da lofi musiqa qo'y", "youtube_search", {"query": "lofi musiqa"}),
    ("youtube'da mashhur qoshiqlar och", "youtube_search", {"query": "mashhur qoshiqlar"}),
    ("Telegramni och", "open_app", {"name": "telegram"}),
    ("chrome och", "open_app", {"name": "chrome"}),
    ("brauzerni ishga tushir", "open_app", {"name": "brauzer"}),
    ("Telegramni yop", "close_app", {"name": "telegram"}),
    ("chromeni yop", "close_app", {"name": "chrome"}),
    ("Yuklamalar papkasini och", "open_folder", {"name_or_path": "yuklamalar papkasi"}),
    ("hujjatlar papkasini och", "open_folder", {"name_or_path": "hujjatlar papkasi"}),
    ("ish stolini och", "open_folder", {"name_or_path": "ish stoli"}),
    ("kun.uz ni och", "open_url", {"url": "kun.uz"}),
    ("github och", "open_url", {"url": "github"}),
]


@pytest.mark.parametrize("text,expected_tool,expected_args", _MATCH_CASES)
def test_match_known_intents(
    full_registry: ToolRegistry, text: str, expected_tool: str, expected_args: dict
) -> None:
    fastpath = FastPath(full_registry)

    match = fastpath.match(text)

    assert match is not None
    assert match.tool_name == expected_tool
    assert match.args == expected_args


_AMBIGUOUS_CASES = [
    "Bugun ob-havo qanday?",
    "Menga bir latifa ayt",
    "Ovoz haqida gapir",
    "Nima uchun kompyuter sekin ishlayapti?",
    "Salom, qandaysiz?",
    "",
    "   ",
]


@pytest.mark.parametrize("text", _AMBIGUOUS_CASES)
def test_ambiguous_phrases_return_none(full_registry: ToolRegistry, text: str) -> None:
    fastpath = FastPath(full_registry)

    assert fastpath.match(text) is None


def test_unregistered_tool_intents_fall_through_to_none(
    registry_with_demo_tools: ToolRegistry,
) -> None:
    fastpath = FastPath(registry_with_demo_tools)

    # regex matches "set_volume"/"open_app" etc., but this registry only
    # has get_time/get_date -- match() must fall through to None either way.
    assert fastpath.match("Ovozni balandlat") is None
    assert fastpath.match("Telegramni och") is None
    assert fastpath.match("Kompyuterni qulfla") is None


def test_registered_demo_tools_still_match(registry_with_demo_tools: ToolRegistry) -> None:
    fastpath = FastPath(registry_with_demo_tools)

    match = fastpath.match("Soat nechi?")

    assert match is not None
    assert match.tool_name == "get_time"
