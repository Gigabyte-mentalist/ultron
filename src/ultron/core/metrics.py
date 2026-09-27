"""Per-request latency metrics (docs/tz/08_orchestrator.md §8B.5).

``RequestTimer`` accumulates monotonic timestamps as one command moves
through the pipeline; ``finalize()`` turns it into the immutable
``RequestMetrics`` record that ``MetricsWriter`` appends as one JSON line to
``data/logs/metrics.jsonl``. ``*_ms`` fields are computed from
``time.monotonic()`` deltas (immune to wall-clock adjustments); only the
record's own ``ts`` is wall-clock, for human-readable log correlation.
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class RequestMetrics:
    ts: str
    wake_source: str
    path: str = "llm"  # "llm" | "fastpath"
    tools: list[str] = field(default_factory=list)
    stt_text: str = ""
    confidence: float | None = None
    wake_ms: int | None = None
    vad_end_to_stt_ms: int | None = None
    stt_to_first_token_ms: int | None = None
    first_token_to_first_audio_ms: int | None = None
    total_to_first_audio_ms: int | None = None
    error: str | None = None


class MetricsWriter:
    """Appends one ``RequestMetrics`` per call to ``path`` as a JSON line."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def record(self, metrics: RequestMetrics) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(asdict(metrics), ensure_ascii=False)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # noqa: BLE001 -- a metrics-write failure must not break a request
            logger.warning("Metrikani yozib bo'lmadi: {}", exc)


def _ms(start: float | None, end: float | None) -> int | None:
    if start is None or end is None:
        return None
    return int((end - start) * 1000)


class RequestTimer:
    """Mutable accumulator for one request's timing anchors.

    Construct it the moment the endpointed ``Utterance``/wake trailing-text
    is available (that's the ``vad_end`` anchor), then call the ``mark_*``
    methods as the request moves through wake confirmation, STT, the first
    LLM/fastpath token, and the first audio actually reaching the speaker.
    """

    def __init__(self, *, wake_source: str, vad_end: float | None = None) -> None:
        self.wake_source = wake_source
        self.path = "llm"
        self.tools: list[str] = []
        self.stt_text = ""
        self.confidence: float | None = None
        self.error: str | None = None
        # Not part of RequestMetrics -- only consumed by the orchestrator's
        # docs/tz/11_uzbek_stt_finetune.md §3.1 "silver" auto-label decision.
        self.tool_ok = True

        self._vad_end = vad_end if vad_end is not None else time.monotonic()
        self._wake_ms: int | None = None
        self._stt_done: float | None = None
        self._first_token: float | None = None
        self._first_audio: float | None = None

    def set_wake_ms(self, value_ms: int) -> None:
        self._wake_ms = value_ms

    def mark_stt_done(self, *, text: str, confidence: float) -> None:
        self._stt_done = time.monotonic()
        self.stt_text = text
        self.confidence = confidence

    def mark_first_token(self) -> None:
        if self._first_token is None:
            self._first_token = time.monotonic()

    def mark_first_audio(self) -> None:
        if self._first_audio is None:
            self._first_audio = time.monotonic()

    def add_tool(self, name: str) -> None:
        self.tools.append(name)

    def mark_tool_result(self, ok: bool) -> None:
        self.tool_ok = self.tool_ok and ok

    def finalize(self) -> RequestMetrics:
        return RequestMetrics(
            ts=datetime.datetime.now().isoformat(timespec="milliseconds"),
            wake_source=self.wake_source,
            path=self.path,
            tools=self.tools,
            stt_text=self.stt_text,
            confidence=self.confidence,
            wake_ms=self._wake_ms,
            vad_end_to_stt_ms=_ms(self._vad_end, self._stt_done),
            stt_to_first_token_ms=_ms(self._stt_done, self._first_token),
            first_token_to_first_audio_ms=_ms(self._first_token, self._first_audio),
            total_to_first_audio_ms=_ms(self._vad_end, self._first_audio),
            error=self.error,
        )
