"""Async Ollama wrapper: streaming chat with tool calling, retries, and a
``think``-unsupported fallback (docs/tz/06_llm.md §3.1).

Verified live against the installed ``ollama`` 0.6.x Python package and an
Ollama 0.34.4 server running ``qwen3:8b`` (2026-09-27):

- The structured ``think`` kwarg works end-to-end: passing ``think=False``
  suppresses reasoning entirely, and ``think=True`` returns it in a separate
  ``message.thinking`` field -- never inline ``<think>...</think>`` tags in
  ``message.content``. The ``/no_think`` + tag-stripping fallback this module
  implements is therefore untested against a *real* old server/library
  combination; it's exercised here only via ``test_client.py``'s fakes, for
  whenever this project's ``ollama`` dependency is downgraded or an older
  server is targeted.
- ``keep_alive`` must NOT be sent as the unitless string ``"-1"`` that
  ``config.yaml`` writes (docs/tz/00_umumiy_arxitektura.md §5) -- Ollama's Go
  duration parser 400s on it ("missing unit in duration"). A bare integer
  number of seconds (``-1`` meaning forever) works, so ``_normalize_keep_alive``
  converts the numeric string once at construction time.
- Tool calls arrive as one non-final streamed chunk with ``message.tool_calls``
  set and ``message.content`` empty, followed by an empty ``done=True`` chunk
  -- never interleaved with content deltas.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import ollama
from loguru import logger

from ultron.config import LlmConfig

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_RETRY_DELAYS_S = (0.0, 0.5, 1.0, 2.0)
_CONNECTION_ERRORS = (ConnectionError, httpx.TransportError, TimeoutError)


class LLMUnavailable(Exception):
    """Ollama couldn't be reached after retries, or the request timed out."""


@dataclass
class ChatChunk:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


def _normalize_keep_alive(keep_alive: str) -> int | str:
    """``"-1"``/``"300"`` (unitless, as written in config.yaml) -> ``int``
    seconds; anything else (``"5m"``) passes through unchanged."""
    try:
        return int(keep_alive)
    except ValueError:
        return keep_alive


async def _prepend(
    first: ollama.ChatResponse | None, rest: AsyncIterator[ollama.ChatResponse]
) -> AsyncIterator[ollama.ChatResponse]:
    if first is not None:
        yield first
    async for item in rest:
        yield item


class OllamaClient:
    """Thin async wrapper around ``ollama.AsyncClient`` (docs/tz/06_llm.md §3.1)."""

    def __init__(self, config: LlmConfig) -> None:
        self._config = config
        self._client = ollama.AsyncClient(host=config.host)
        self._keep_alive = _normalize_keep_alive(config.keep_alive)
        self._think_supported: bool | None = None  # probed lazily, cached after first success

    async def health(self) -> bool:
        """True if the server responds and the configured model is present."""
        try:
            response = await self._client.list()
        except Exception as exc:
            logger.warning("Ollama health-check muvaffaqiyatsiz: {}", exc)
            return False
        names = [m.model for m in response.models if m.model]
        return self._config.model in names or any(
            n.startswith(f"{self._config.model}:") for n in names
        )

    async def ensure_loaded(self) -> None:
        """Empty request so Ollama loads the model into VRAM up front (called at startup)."""
        try:
            await self._client.chat(
                model=self._config.model, messages=[], keep_alive=self._keep_alive
            )
        except Exception as exc:
            logger.warning("Modelni oldindan yuklab bo'lmadi: {}", exc)

    async def is_model_loaded(self) -> bool:
        """True if the configured model is currently resident in VRAM
        (``ollama ps``/``/api/ps``) -- distinct from :meth:`health`, which
        only checks that the model is *installed* (docs/tz/09_autostart.md
        §3.6: "Ollama /api/tags va model yuklanganmi (/api/ps)")."""
        try:
            response = await self._client.ps()
        except Exception as exc:
            logger.warning("Ollama 'ps' so'rovi muvaffaqiyatsiz: {}", exc)
            return False
        names = [m.model for m in response.models if m.model]
        return self._config.model in names or any(
            n.startswith(f"{self._config.model}:") for n in names
        )

    def set_keep_alive(self, keep_alive: int | str) -> None:
        """docs/tz/09_autostart.md §3.7 ("GPU'ni bo'shatish"): overrides the
        ``keep_alive`` every subsequent :meth:`chat_stream`/:meth:`ensure_loaded`
        call sends, so the model unloads itself instead of staying pinned in
        VRAM once the current response finishes."""
        self._keep_alive = keep_alive

    async def unload(self) -> None:
        """Forces the model out of VRAM immediately (``keep_alive=0``), rather
        than waiting for the current ``keep_alive`` countdown to expire."""
        try:
            await self._client.chat(model=self._config.model, messages=[], keep_alive=0)
        except Exception as exc:
            logger.warning("Modelni VRAM'dan tushirib bo'lmadi: {}", exc)

    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[ChatChunk]:
        """Stream a chat completion, retrying transient connection failures
        up to 3 times (0.5/1/2 s backoff) and falling back to ``/no_think``
        if ``think`` isn't supported by the installed library/server."""
        base_kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "tools": tools or [],
            "stream": True,
            "options": {
                "temperature": self._config.temperature,
                "num_ctx": self._config.num_ctx,
            },
            "keep_alive": self._keep_alive,
        }

        stream = await self._open_stream(base_kwargs, messages)
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream.__anext__(), timeout=self._config.request_timeout_s
                    )
                except StopAsyncIteration:
                    return
                except TimeoutError as exc:
                    raise LLMUnavailable("Ollama javob bermadi (timeout)") from exc

                message = chunk.message
                content = _THINK_TAG_RE.sub("", message.content or "")
                tool_calls = [
                    {"name": tc.function.name, "arguments": dict(tc.function.arguments)}
                    for tc in (message.tool_calls or [])
                ]
                yield ChatChunk(content=content, tool_calls=tool_calls, done=bool(chunk.done))
        except _CONNECTION_ERRORS as exc:
            raise LLMUnavailable(f"Ollama bilan aloqa uzildi: {exc}") from exc

    async def _open_stream(
        self, base_kwargs: dict[str, Any], messages: list[dict[str, Any]]
    ) -> AsyncIterator[ollama.ChatResponse]:
        last_exc: Exception | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS_S):
            if delay:
                await asyncio.sleep(delay)
            try:
                return await self._attempt_open(base_kwargs, messages)
            except _CONNECTION_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "Ollama ulanish xatosi (urinish {}/{}): {}",
                    attempt + 1,
                    len(_RETRY_DELAYS_S),
                    exc,
                )
        raise LLMUnavailable("Ollama bilan aloqa o'rnatilmadi") from last_exc

    async def _attempt_open(
        self, base_kwargs: dict[str, Any], messages: list[dict[str, Any]]
    ) -> AsyncIterator[ollama.ChatResponse]:
        kwargs = dict(base_kwargs)
        if self._think_supported is not False:
            kwargs["think"] = self._config.think

        try:
            stream = await asyncio.wait_for(
                self._client.chat(**kwargs), timeout=self._config.request_timeout_s
            )
        except TypeError:
            if "think" not in kwargs:
                raise
            self._disable_think(messages)
            return await self._attempt_open(base_kwargs, messages)

        try:
            first = await asyncio.wait_for(
                stream.__anext__(), timeout=self._config.request_timeout_s
            )
        except StopAsyncIteration:
            first = None
        except ollama.ResponseError as exc:
            if "think" in kwargs and "think" in str(exc).lower():
                self._disable_think(messages)
                return await self._attempt_open(base_kwargs, messages)
            raise LLMUnavailable(str(exc)) from exc

        if "think" in kwargs:
            self._think_supported = True
        return _prepend(first, stream)

    def _disable_think(self, messages: list[dict[str, Any]]) -> None:
        self._think_supported = False
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0]['content'].rstrip()}\n/no_think"
        logger.warning("Ollama 'think' parametrini qo'llamaydi, '/no_think' fallback ishlatildi")
