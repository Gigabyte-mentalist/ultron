"""Tests for ultron.tools.base.ToolRegistry (docs/tz/06_llm.md §3.2, §4)."""

from __future__ import annotations

import asyncio
from typing import Annotated

from ultron.tools.base import Tool, ToolRegistry, ToolResult, _build_arg_model, _build_schema


def _make_tool(func, *, dangerous: bool = False) -> Tool:
    return Tool(
        name=func.__name__,
        description="test tool",
        dangerous=dangerous,
        func=func,
        schema=_build_schema(func),
        arg_model=_build_arg_model(func),
    )


class _AlwaysConfirm:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.calls: list[str] = []

    async def confirm(self, summary_uz: str) -> bool:
        self.calls.append(summary_uz)
        return self.answer


async def test_call_unknown_tool_returns_error() -> None:
    registry = ToolRegistry()

    result = await registry.call("no_such_tool", {})

    assert result.ok is False
    assert "no_such_tool" in result.error


async def test_validation_error_returns_ok_false_without_calling_func() -> None:
    called = False

    async def greet(name: Annotated[str, "name"]) -> ToolResult:
        nonlocal called
        called = True
        return ToolResult(ok=True)

    registry = ToolRegistry()
    registry.register(_make_tool(greet))

    result = await registry.call("greet", {})  # missing required "name"

    assert result.ok is False
    assert result.error is not None
    assert called is False


async def test_dangerous_tool_calls_confirmation_provider_before_running() -> None:
    ran = False

    async def delete_stuff() -> ToolResult:
        nonlocal ran
        ran = True
        return ToolResult(ok=True)

    confirmation = _AlwaysConfirm(answer=True)
    registry = ToolRegistry(confirmation=confirmation)
    registry.register(_make_tool(delete_stuff, dangerous=True))

    result = await registry.call("delete_stuff", {})

    assert len(confirmation.calls) == 1
    assert ran is True
    assert result.ok is True


async def test_dangerous_tool_not_run_when_confirmation_declined() -> None:
    ran = False

    async def delete_stuff() -> ToolResult:
        nonlocal ran
        ran = True
        return ToolResult(ok=True)

    confirmation = _AlwaysConfirm(answer=False)
    registry = ToolRegistry(confirmation=confirmation)
    registry.register(_make_tool(delete_stuff, dangerous=True))

    result = await registry.call("delete_stuff", {})

    assert result.ok is False
    assert ran is False


async def test_tool_timeout_returns_ok_false() -> None:
    async def slow() -> ToolResult:
        await asyncio.sleep(999)
        return ToolResult(ok=True)

    registry = ToolRegistry()
    registry.register(_make_tool(slow))

    import ultron.tools.base as base_module

    original_timeout = base_module.TOOL_TIMEOUT_S
    base_module.TOOL_TIMEOUT_S = 0.05
    try:
        result = await registry.call("slow", {})
    finally:
        base_module.TOOL_TIMEOUT_S = original_timeout

    assert result.ok is False
    assert "timeout" in result.error.lower()


async def test_tool_exception_is_caught_and_returned_as_error() -> None:
    async def boom() -> ToolResult:
        raise RuntimeError("kaboom")

    registry = ToolRegistry()
    registry.register(_make_tool(boom))

    result = await registry.call("boom", {})

    assert result.ok is False
    assert "kaboom" in result.error


def test_schemas_only_includes_enabled_tools() -> None:
    async def a() -> ToolResult:
        return ToolResult(ok=True)

    async def b() -> ToolResult:
        return ToolResult(ok=True)

    registry = ToolRegistry()
    tool_a = _make_tool(a)
    tool_b = _make_tool(b)
    tool_b.enabled = False
    registry.register(tool_a)
    registry.register(tool_b)

    names = {s["function"]["name"] for s in registry.schemas()}
    assert names == {"a"}
    assert "a" in registry
    assert "b" not in registry
