"""Tests for ultron.llm.agent.Agent (docs/tz/06_llm.md §3.4, §4).

Uses a fake Ollama client (a duck-typed stand-in for ``OllamaClient`` that
only implements ``chat_stream``) so the tool-calling loop, max-iteration
cutoff, and cancellation/barge-in behavior can be tested without a real
Ollama server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from ultron.config import LlmConfig
from ultron.llm.agent import (
    Agent,
    DoneEvent,
    ErrorEvent,
    SentenceEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from ultron.llm.client import ChatChunk, LLMUnavailable
from ultron.llm.fastpath import FastPath
from ultron.tools.base import Tool, ToolRegistry, ToolResult, _build_arg_model, _build_schema


class _FakeOllamaClient:
    """Each ``chat_stream`` call pops the next queued response list."""

    def __init__(self, responses: list[list[ChatChunk]]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        self.yielded_count = 0

    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(messages)
        response = self._responses.pop(0) if self._responses else [ChatChunk(done=True)]
        for chunk in response:
            yield chunk
            self.yielded_count += 1


class _FailingOllamaClient:
    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[ChatChunk]:
        raise LLMUnavailable("Ollama server o'chirilgan")
        yield  # pragma: no cover -- makes this an async generator function


async def _get_time() -> ToolResult:
    return ToolResult(ok=True, data={"hour": 14, "minute": 5}, speak="Hozir soat 14:05.")


def _registry_with_get_time() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_time",
            description="Return current time",
            dangerous=False,
            func=_get_time,
            schema=_build_schema(_get_time),
            arg_model=_build_arg_model(_get_time),
        )
    )
    return registry


def _make_agent(
    client: Any,
    registry: ToolRegistry | None = None,
    *,
    max_tool_iterations: int = 5,
    fastpath_enabled: bool = False,
) -> Agent:
    registry = registry or ToolRegistry()
    config = LlmConfig(max_tool_iterations=max_tool_iterations)
    return Agent(client, registry, FastPath(registry), config, fastpath_enabled=fastpath_enabled)


async def _collect(agent: Agent, text: str) -> list[Any]:
    return [event async for event in agent.respond(text)]


async def test_plain_conversational_response_streams_sentences() -> None:
    client = _FakeOllamaClient(
        [[ChatChunk(content="Salom, "), ChatChunk(content="qandaysiz? "), ChatChunk(done=True)]]
    )
    agent = _make_agent(client)

    events = await _collect(agent, "salom")

    sentence_events = [e for e in events if isinstance(e, SentenceEvent)]
    assert [e.text for e in sentence_events] == ["Salom, qandaysiz?"]
    assert isinstance(events[-1], DoneEvent)
    assert len(client.calls) == 1


async def test_tool_call_then_final_answer() -> None:
    client = _FakeOllamaClient(
        [
            [ChatChunk(tool_calls=[{"name": "get_time", "arguments": {}}], done=True)],
            [ChatChunk(content="Hozir soat 14:05."), ChatChunk(done=True)],
        ]
    )
    registry = _registry_with_get_time()
    agent = _make_agent(client, registry)

    events = await _collect(agent, "soat nechi")

    assert any(isinstance(e, ToolStartEvent) and e.name == "get_time" for e in events)
    assert any(isinstance(e, ToolEndEvent) and e.result.ok for e in events)
    sentence_events = [e for e in events if isinstance(e, SentenceEvent)]
    assert sentence_events[-1].text == "Hozir soat 14:05."
    assert len(client.calls) == 2

    # the second LLM call must include the assistant tool_calls turn + the
    # tool result turn so the model can see what it asked for and got back.
    second_call_messages = client.calls[1]
    roles = [m["role"] for m in second_call_messages]
    assert "tool" in roles


async def test_tool_result_speak_used_when_llm_says_nothing_after() -> None:
    client = _FakeOllamaClient(
        [
            [ChatChunk(tool_calls=[{"name": "get_time", "arguments": {}}], done=True)],
            [ChatChunk(done=True)],  # LLM has nothing further to add
        ]
    )
    registry = _registry_with_get_time()
    agent = _make_agent(client, registry)

    events = await _collect(agent, "soat nechi")

    sentence_events = [e for e in events if isinstance(e, SentenceEvent)]
    assert [e.text for e in sentence_events] == ["Hozir soat 14:05."]


async def test_max_tool_iterations_exceeded_gives_up_gracefully() -> None:
    always_calls_tool = [ChatChunk(tool_calls=[{"name": "get_time", "arguments": {}}], done=True)]
    client = _FakeOllamaClient([always_calls_tool, always_calls_tool, always_calls_tool])
    registry = _registry_with_get_time()
    agent = _make_agent(client, registry, max_tool_iterations=3)

    events = await _collect(agent, "soat nechi")

    assert len(client.calls) == 3
    sentence_events = [e for e in events if isinstance(e, SentenceEvent)]
    assert sentence_events[-1].text == "Kechirasiz, buni bajara olmadim"
    assert isinstance(events[-1], DoneEvent)


async def test_llm_unavailable_yields_error_event_and_does_not_raise() -> None:
    agent = _make_agent(_FailingOllamaClient())

    events = await _collect(agent, "salom")

    assert any(isinstance(e, ErrorEvent) for e in events)
    assert isinstance(events[-1], DoneEvent)


async def test_cancelling_respond_stops_the_underlying_stream_early() -> None:
    client = _FakeOllamaClient(
        [
            [
                ChatChunk(content="Bir. "),
                ChatChunk(content="Ikki. "),
                ChatChunk(content="Uch. "),
                ChatChunk(content="To'rt."),
                ChatChunk(done=True),
            ]
        ]
    )
    agent = _make_agent(client)

    gen = agent.respond("uzoq gapir")
    first_event = await gen.__anext__()
    assert isinstance(first_event, SentenceEvent)

    await gen.aclose()

    assert client.yielded_count < 5


async def test_reset_history_clears_prior_turns() -> None:
    client = _FakeOllamaClient(
        [
            [ChatChunk(content="Birinchi javob."), ChatChunk(done=True)],
            [ChatChunk(content="Ikkinchi javob."), ChatChunk(done=True)],
        ]
    )
    agent = _make_agent(client)

    await _collect(agent, "birinchi")
    agent.reset_history()
    await _collect(agent, "ikkinchi")

    second_call_messages = client.calls[1]
    user_messages = [m for m in second_call_messages if m["role"] == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "ikkinchi"


@pytest.mark.parametrize(
    "text,expected_speak",
    [
        ("soat nechi", "Hozir soat 14:05."),
    ],
)
async def test_fastpath_short_circuits_the_llm(text: str, expected_speak: str) -> None:
    registry = _registry_with_get_time()
    client = _FakeOllamaClient([])  # if the LLM were called, there'd be nothing queued
    agent = _make_agent(client, registry, fastpath_enabled=True)

    events = await _collect(agent, text)

    assert not client.calls  # LLM never invoked
    sentence_events = [e for e in events if isinstance(e, SentenceEvent)]
    assert sentence_events[-1].text == expected_speak
