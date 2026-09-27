"""Tool framework: schema generation from type hints, registry, argument
validation, and the confirmation gate for dangerous tools
(docs/tz/06_llm.md §3.2, docs/tz/00_umumiy_arxitektura.md §6).
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import UnionType
from typing import Any, Literal, Protocol, Union, get_args, get_origin

from loguru import logger
from pydantic import BaseModel, ValidationError, create_model

TOOL_TIMEOUT_S = 15.0


def is_mock_mode() -> bool:
    """``ULTRON_TOOLS_MOCK=1`` -- state-changing tools log what they would do
    instead of performing the real action (docs/tz/07_tools.md §1: eval and
    tests run with this set so a buggy/mistaken tool call never touches the
    real system)."""
    return os.environ.get("ULTRON_TOOLS_MOCK") == "1"


class ToolError(Exception):
    """Raised when a ``@tool``-decorated function's signature can't be turned
    into a JSON schema (unsupported type hint)."""


@dataclass
class ToolResult:
    ok: bool
    data: dict | str | None = None
    speak: str | None = None  # short Uzbek text the user hears, if any
    error: str | None = None


class ConfirmationProvider(Protocol):
    async def confirm(self, summary_uz: str) -> bool: ...


class ConsoleConfirmationProvider:
    """Confirms ``dangerous=True`` tool calls on stdin (docs/tz/06_llm.md §3.7).

    docs/tz/08_orchestrator.md replaces this with a voice-based provider for
    the real pipeline; this one keeps the ``--text`` REPL usable on its own.
    """

    async def confirm(self, summary_uz: str) -> bool:
        answer = await asyncio.to_thread(input, f"{summary_uz} [ha/yo'q]: ")
        return answer.strip().lower() in {"ha", "ha.", "h", "y", "yes"}


def _unwrap(annotation: Any) -> tuple[Any, str | None, bool]:
    """Fully unwrap ``Annotated``/``Optional`` regardless of nesting order
    (``Optional[Annotated[X, d]]`` and ``Annotated[Optional[X], d]`` both
    reduce to ``(X, d, True)``) -> ``(bare_type, description, optional)``."""
    description: str | None = None
    optional = False
    changed = True
    while changed:
        changed = False
        metadata = getattr(annotation, "__metadata__", None)
        if metadata:
            found = next((m for m in metadata if isinstance(m, str)), None)
            description = found or description
            annotation = annotation.__origin__
            changed = True
            continue
        if get_origin(annotation) in (Union, UnionType):
            args = get_args(annotation)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1 and len(args) == 2:
                annotation = non_none[0]
                optional = True
                changed = True
                continue
    return annotation, description, optional


def _json_type_schema(annotation: Any) -> dict[str, Any]:
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if get_origin(annotation) is Literal:
        choices = get_args(annotation)
        schema = _json_type_schema(type(choices[0]))
        schema["enum"] = list(choices)
        return schema
    raise ToolError(f"Qo'llab-quvvatlanmaydigan tool argument turi: {annotation!r}")


def _iter_params(
    func: Callable[..., Any],
) -> list[tuple[str, inspect.Parameter, Any, str | None, bool]]:
    sig = inspect.signature(func)
    hints = typing.get_type_hints(func, include_extras=True)
    out = []
    for name, param in sig.parameters.items():
        annotation = hints.get(name, str)
        bare, description, optional = _unwrap(annotation)
        out.append((name, param, bare, description, optional))
    return out


def _build_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Function signature + type hints -> Ollama/OpenAI-style JSON schema
    (str/int/float/bool, ``Literal`` -> enum, ``Optional``, defaults)."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param, bare, description, optional in _iter_params(func):
        prop = _json_type_schema(bare)
        if description:
            prop["description"] = description
        has_default = param.default is not inspect.Parameter.empty
        if has_default and param.default is not None:
            prop["default"] = param.default
        properties[name] = prop
        if not has_default and not optional:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _build_arg_model(func: Callable[..., Any]) -> type[BaseModel]:
    """Function signature -> a pydantic model used to validate incoming
    LLM-supplied arguments before the function is ever called."""
    fields: dict[str, Any] = {}
    for name, param, bare, _description, optional in _iter_params(func):
        python_type = (bare | None) if optional else bare
        if param.default is not inspect.Parameter.empty:
            default = param.default
        elif optional:
            default = None
        else:
            default = ...
        fields[name] = (python_type, default)
    return create_model(f"{func.__name__}_Args", **fields)  # type: ignore[call-overload]


@dataclass
class Tool:
    name: str
    description: str
    dangerous: bool
    func: Callable[..., Awaitable[ToolResult]]
    schema: dict[str, Any]
    arg_model: type[BaseModel]
    enabled: bool = True
    # docs/tz/08_orchestrator.md §8B.3: the Uzbek question spoken by
    # VoiceConfirmationProvider for a dangerous tool. ``description`` is
    # English (LLM-facing, per CLAUDE.md's "kod ingliz tilida" rule), so it
    # is never itself something to speak aloud -- this is the integration
    # point the confirmation flow actually needs. Falls back to
    # ``description`` (English) for any dangerous tool that doesn't set one,
    # which only matters for ``ConsoleConfirmationProvider``'s text-mode REPL.
    confirm_question_uz: str | None = None

    def ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


class ToolRegistry:
    """Holds every ``@tool``-decorated function; validates, confirms, times
    out, and audit-logs each call (docs/tz/06_llm.md §3.2)."""

    def __init__(self, *, confirmation: ConfirmationProvider | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.confirmation: ConfirmationProvider = confirmation or ConsoleConfirmationProvider()

    def register(self, tool_obj: Tool) -> None:
        self._tools[tool_obj.name] = tool_obj

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> dict[str, Tool]:
        return dict(self._tools)

    def __contains__(self, name: str) -> bool:
        tool_obj = self._tools.get(name)
        return tool_obj is not None and tool_obj.enabled

    def schemas(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        tools = self._tools.values()
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return [t.ollama_schema() for t in tools]

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool_obj = self._tools.get(name)
        if tool_obj is None or not tool_obj.enabled:
            result = ToolResult(ok=False, error=f"Noma'lum tool: {name}")
            self._audit(name, args, result, 0)
            return result

        started = time.monotonic()

        try:
            validated = tool_obj.arg_model.model_validate(args)
        except ValidationError as exc:
            result = ToolResult(ok=False, error=f"Argument xatosi: {exc}")
            self._audit(name, args, result, self._elapsed_ms(started))
            return result

        if tool_obj.dangerous:
            summary = tool_obj.confirm_question_uz or (
                f"{tool_obj.description}: {validated.model_dump()}"
            )
            confirmed = await self.confirmation.confirm(summary)
            if not confirmed:
                result = ToolResult(ok=False, error="Foydalanuvchi tasdiqlamadi")
                self._audit(name, args, result, self._elapsed_ms(started))
                return result

        try:
            result = await asyncio.wait_for(
                tool_obj.func(**validated.model_dump()), timeout=TOOL_TIMEOUT_S
            )
        except TimeoutError:
            result = ToolResult(ok=False, error="Tool javob bermadi (timeout)")
        except Exception as exc:  # noqa: BLE001 -- a buggy tool must not crash the agent loop
            logger.exception("Tool '{}' kutilmagan xato bilan yiqildi", name)
            result = ToolResult(ok=False, error=f"Tool xatosi: {exc}")

        self._audit(name, args, result, self._elapsed_ms(started))
        return result

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    def _audit(self, name: str, args: dict[str, Any], result: ToolResult, duration_ms: int) -> None:
        logger.bind(audit=True).info(
            "tool={} args={} ok={} error={} duration_ms={}",
            name,
            args,
            result.ok,
            result.error,
            duration_ms,
        )


default_registry = ToolRegistry()


def tool(
    *,
    name: str,
    description: str,
    dangerous: bool = False,
    confirm_question_uz: str | None = None,
) -> Callable[[Callable[..., Awaitable[ToolResult]]], Callable[..., Awaitable[ToolResult]]]:
    """Decorator: builds a JSON schema + pydantic validator from the function
    signature and registers it in ``default_registry`` (docs/tz/06_llm.md §3.2).

        @tool(name="open_app", description="...", dangerous=False)
        async def open_app(name: Annotated[str, "..."]) -> ToolResult: ...

    ``confirm_question_uz`` is the Uzbek question spoken for a
    ``dangerous=True`` tool (docs/tz/08_orchestrator.md §8B.3); irrelevant
    otherwise.
    """

    def decorator(
        func: Callable[..., Awaitable[ToolResult]],
    ) -> Callable[..., Awaitable[ToolResult]]:
        schema = _build_schema(func)
        arg_model = _build_arg_model(func)
        default_registry.register(
            Tool(
                name=name,
                description=description,
                dangerous=dangerous,
                func=func,
                schema=schema,
                arg_model=arg_model,
                confirm_question_uz=confirm_question_uz,
            )
        )
        return func

    return decorator
