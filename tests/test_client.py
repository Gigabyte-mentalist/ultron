"""Tests for ultron.llm.client.OllamaClient (docs/tz/06_llm.md §3.1).

Uses fakes standing in for ``ollama.AsyncClient`` (assigned directly onto
``OllamaClient._client``) rather than a real Ollama server -- retry/fallback
timing and error paths need to be deterministic and fast in CI. The
``keep_alive``-string and ``think``-support behaviors these fakes exercise
were verified against a real Ollama 0.34.4 server + ``ollama`` 0.6.x
library first (see ``client.py``'s module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ultron.config import LlmConfig
from ultron.llm import client as client_module
from ultron.llm.client import LLMUnavailable, OllamaClient, _normalize_keep_alive


@dataclass
class _FakeFunction:
    name: str
    arguments: dict[str, Any]


@dataclass
class _FakeToolCall:
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None = None
    tool_calls: list[_FakeToolCall] = field(default_factory=list)


@dataclass
class _FakeResponse:
    message: _FakeMessage
    done: bool = False


async def _fake_stream(items: list[_FakeResponse]):
    for item in items:
        yield item


def _client_with_fake_raw(config: LlmConfig | None, fake_raw: Any) -> OllamaClient:
    client = OllamaClient(config or LlmConfig())
    client._client = fake_raw
    return client


def test_normalize_keep_alive_numeric_string_becomes_int() -> None:
    assert _normalize_keep_alive("-1") == -1
    assert _normalize_keep_alive("300") == 300
    assert _normalize_keep_alive("0") == 0


def test_normalize_keep_alive_duration_string_passes_through() -> None:
    assert _normalize_keep_alive("5m") == "5m"


async def test_health_true_when_model_present() -> None:
    class _FakeModel:
        def __init__(self, name: str) -> None:
            self.model = name

    class _FakeListResponse:
        models = [_FakeModel("qwen3:8b")]

    class _FakeRaw:
        async def list(self) -> _FakeListResponse:
            return _FakeListResponse()

    client = _client_with_fake_raw(LlmConfig(model="qwen3:8b"), _FakeRaw())

    assert await client.health() is True


async def test_health_false_when_model_missing() -> None:
    class _FakeListResponse:
        models: list[Any] = []

    class _FakeRaw:
        async def list(self) -> _FakeListResponse:
            return _FakeListResponse()

    client = _client_with_fake_raw(LlmConfig(model="qwen3:8b"), _FakeRaw())

    assert await client.health() is False


async def test_health_false_when_server_unreachable() -> None:
    class _FakeRaw:
        async def list(self) -> Any:
            raise ConnectionError("no route to host")

    client = _client_with_fake_raw(None, _FakeRaw())

    assert await client.health() is False


async def test_is_model_loaded_true_when_ps_lists_model() -> None:
    class _FakeModel:
        def __init__(self, name: str) -> None:
            self.model = name

    class _FakePsResponse:
        models = [_FakeModel("qwen3:8b")]

    class _FakeRaw:
        async def ps(self) -> _FakePsResponse:
            return _FakePsResponse()

    client = _client_with_fake_raw(LlmConfig(model="qwen3:8b"), _FakeRaw())

    assert await client.is_model_loaded() is True


async def test_is_model_loaded_false_when_not_in_ps() -> None:
    class _FakePsResponse:
        models: list[Any] = []

    class _FakeRaw:
        async def ps(self) -> _FakePsResponse:
            return _FakePsResponse()

    client = _client_with_fake_raw(LlmConfig(model="qwen3:8b"), _FakeRaw())

    assert await client.is_model_loaded() is False


async def test_is_model_loaded_false_on_error() -> None:
    class _FakeRaw:
        async def ps(self) -> Any:
            raise ConnectionError("no route to host")

    client = _client_with_fake_raw(None, _FakeRaw())

    assert await client.is_model_loaded() is False


async def test_set_keep_alive_overrides_subsequent_ensure_loaded_calls() -> None:
    calls: list[Any] = []

    class _FakeRaw:
        async def chat(self, **kwargs: Any) -> Any:
            calls.append(kwargs)

    client = _client_with_fake_raw(LlmConfig(model="qwen3:8b", keep_alive="-1"), _FakeRaw())
    client.set_keep_alive(0)
    await client.ensure_loaded()

    assert calls[0]["keep_alive"] == 0


async def test_unload_sends_keep_alive_zero() -> None:
    calls: list[Any] = []

    class _FakeRaw:
        async def chat(self, **kwargs: Any) -> Any:
            calls.append(kwargs)

    client = _client_with_fake_raw(LlmConfig(model="qwen3:8b", keep_alive="-1"), _FakeRaw())
    await client.unload()

    assert calls[0]["keep_alive"] == 0
    assert calls[0]["model"] == "qwen3:8b"


async def test_chat_stream_yields_content_and_tool_calls() -> None:
    class _FakeRaw:
        async def chat(self, **kwargs: Any) -> Any:
            return _fake_stream(
                [
                    _FakeResponse(_FakeMessage(content="Salom")),
                    _FakeResponse(
                        _FakeMessage(tool_calls=[_FakeToolCall(_FakeFunction("get_time", {}))])
                    ),
                    _FakeResponse(_FakeMessage(), done=True),
                ]
            )

    client = _client_with_fake_raw(None, _FakeRaw())

    chunks = [c async for c in client.chat_stream([{"role": "system", "content": "sp"}])]

    assert chunks[0].content == "Salom"
    assert chunks[1].tool_calls == [{"name": "get_time", "arguments": {}}]
    assert chunks[-1].done is True


async def test_think_unsupported_falls_back_to_no_think_and_retries() -> None:
    class _FakeRaw:
        def __init__(self) -> None:
            self.chat_calls: list[dict[str, Any]] = []

        async def chat(self, **kwargs: Any) -> Any:
            self.chat_calls.append(kwargs)
            if "think" in kwargs:
                raise TypeError("chat() got an unexpected keyword argument 'think'")
            return _fake_stream([_FakeResponse(_FakeMessage(content="Salom"), done=True)])

    fake = _FakeRaw()
    client = _client_with_fake_raw(None, fake)
    messages = [
        {"role": "system", "content": "Tizim ko'rsatmasi"},
        {"role": "user", "content": "salom"},
    ]

    chunks = [c async for c in client.chat_stream(messages)]

    assert chunks[0].content == "Salom"
    assert len(fake.chat_calls) == 2
    assert "think" in fake.chat_calls[0]
    assert "think" not in fake.chat_calls[1]
    assert messages[0]["content"].endswith("/no_think")


async def test_connection_error_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "_RETRY_DELAYS_S", (0.0, 0.0, 0.0, 0.0))

    class _FlakyRaw:
        def __init__(self) -> None:
            self.attempts = 0

        async def chat(self, **kwargs: Any) -> Any:
            self.attempts += 1
            if self.attempts < 2:
                raise ConnectionError("no route to host")
            return _fake_stream([_FakeResponse(_FakeMessage(content="Salom"), done=True)])

    fake = _FlakyRaw()
    client = _client_with_fake_raw(None, fake)

    chunks = [c async for c in client.chat_stream([{"role": "system", "content": "sp"}])]

    assert fake.attempts == 2
    assert chunks[0].content == "Salom"


async def test_connection_error_gives_up_after_all_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "_RETRY_DELAYS_S", (0.0, 0.0))

    class _AlwaysFailingRaw:
        async def chat(self, **kwargs: Any) -> Any:
            raise ConnectionError("no route to host")

    client = _client_with_fake_raw(None, _AlwaysFailingRaw())

    with pytest.raises(LLMUnavailable):
        _ = [c async for c in client.chat_stream([{"role": "system", "content": "sp"}])]
