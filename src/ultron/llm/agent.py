"""Tool-calling agent loop (docs/tz/06_llm.md §3.4).

``Agent.respond()`` is an async generator: fast-path first, then the
system+history+user -> stream -> sentence-split -> tool-call -> repeat loop,
bounded by ``max_tool_iterations``. Cancelling it (``aclose()``) cancels the
in-flight Ollama request too, for barge-in.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ultron.config import LlmConfig
from ultron.i18n.uz import LLM_UNAVAILABLE, TOOL_ITERATIONS_EXCEEDED
from ultron.llm.client import LLMUnavailable, OllamaClient
from ultron.llm.fastpath import FastPath, FastPathMatch
from ultron.llm.prompts import build_system_prompt
from ultron.llm.sentence_splitter import SentenceSplitter
from ultron.tools.base import ToolRegistry, ToolResult

# docs/tz/06_llm.md §3.3: 5 daqiqa faolsizlikdan keyin tarix tozalanadi
_HISTORY_IDLE_RESET_S = 300.0
_TOOL_HISTORY_MAX_CHARS = 500


@dataclass
class SentenceEvent:
    text: str


@dataclass
class ToolStartEvent:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEndEvent:
    result: ToolResult


@dataclass
class DoneEvent:
    pass


@dataclass
class ErrorEvent:
    message: str


AgentEvent = SentenceEvent | ToolStartEvent | ToolEndEvent | DoneEvent | ErrorEvent


def _tool_result_history_text(result: ToolResult) -> str:
    if result.ok:
        payload: Any = result.data if result.data is not None else (result.speak or "ok")
    else:
        payload = f"xato: {result.error}"
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return text[:_TOOL_HISTORY_MAX_CHARS]


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        registry: ToolRegistry,
        fastpath: FastPath,
        config: LlmConfig,
        *,
        city: str = "Tashkent",
        facts_provider: Callable[[], list[str]] | None = None,
        fastpath_enabled: bool = True,
    ) -> None:
        self._client = client
        self._registry = registry
        self._fastpath = fastpath
        self._config = config
        self._city = city
        self._facts_provider = facts_provider or (lambda: [])
        self.fastpath_enabled = fastpath_enabled
        self._history: list[dict[str, Any]] = []
        self._last_active_at = time.monotonic()

    def reset_history(self) -> None:
        self._history.clear()

    async def respond(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self._maybe_reset_idle_history()
        self._last_active_at = time.monotonic()

        if self.fastpath_enabled:
            match = self._fastpath.match(user_text)
            if match is not None:
                async for event in self._run_fastpath(match):
                    yield event
                return

        async for event in self._run_llm_loop(user_text):
            yield event

    def _maybe_reset_idle_history(self) -> None:
        if self._history and (time.monotonic() - self._last_active_at) > _HISTORY_IDLE_RESET_S:
            self._history.clear()

    async def _run_fastpath(self, match: FastPathMatch) -> AsyncIterator[AgentEvent]:
        yield ToolStartEvent(match.tool_name, match.args)
        result = await self._registry.call(match.tool_name, match.args)
        yield ToolEndEvent(result)
        if result.speak:
            yield SentenceEvent(result.speak)
        elif not result.ok and result.error:
            yield SentenceEvent(result.error)
        yield DoneEvent()

    def _trim_history(self) -> None:
        max_turns = self._config.history_turns
        if max_turns <= 0:
            self._history.clear()
            return
        user_indices = [i for i, m in enumerate(self._history) if m["role"] == "user"]
        if len(user_indices) > max_turns:
            self._history = self._history[user_indices[-max_turns] :]

    async def _run_llm_loop(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        system_prompt = build_system_prompt(city=self._city, facts=self._facts_provider())
        tools = self._registry.schemas()

        last_tool_result: ToolResult | None = None
        said_anything = False

        for _iteration in range(self._config.max_tool_iterations):
            messages = [{"role": "system", "content": system_prompt}, *self._history]
            splitter = SentenceSplitter()
            assistant_text = ""
            tool_calls: list[dict[str, Any]] = []

            try:
                async with contextlib.aclosing(
                    self._client.chat_stream(messages, tools=tools)
                ) as stream:
                    async for chunk in stream:
                        if chunk.content:
                            assistant_text += chunk.content
                            for sentence in splitter.feed(chunk.content):
                                said_anything = True
                                yield SentenceEvent(sentence)
                        if chunk.tool_calls:
                            tool_calls.extend(chunk.tool_calls)
            except LLMUnavailable as exc:
                logger.warning("LLM javob bermadi: {}", exc)
                yield ErrorEvent(LLM_UNAVAILABLE)
                yield DoneEvent()
                return

            trailing = splitter.flush()
            if trailing:
                said_anything = True
                yield SentenceEvent(trailing)

            assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_text}
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls
                ]
            self._history.append(assistant_message)

            if not tool_calls:
                break

            for tc in tool_calls:
                yield ToolStartEvent(tc["name"], tc["arguments"])

            results = await asyncio.gather(
                *(self._registry.call(tc["name"], tc["arguments"]) for tc in tool_calls)
            )

            for _tc, result in zip(tool_calls, results, strict=True):
                last_tool_result = result
                yield ToolEndEvent(result)
                self._history.append({"role": "tool", "content": _tool_result_history_text(result)})
        else:
            yield SentenceEvent(TOOL_ITERATIONS_EXCEEDED)
            yield DoneEvent()
            return

        if not said_anything and last_tool_result and last_tool_result.speak:
            yield SentenceEvent(last_tool_result.speak)

        yield DoneEvent()
