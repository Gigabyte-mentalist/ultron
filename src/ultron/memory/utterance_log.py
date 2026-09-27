"""Per-utterance audio+transcript capture (docs/tz/11_uzbek_stt_finetune.md §3.1).

Every LISTENING/FOLLOW_UP utterance that reaches STT is written to
``<data_dir>/recordings/utterances/YYYY-MM-DD/<uuid>.flac`` plus one JSON line
in that day's ``index.jsonl`` -- raw material for ``scripts/label_tool.py``
and, eventually, TZ-11's personal LoRA fine-tune dataset. This is opt-in via
``config.memory.save_utterances`` (checked by the caller, not here) and
best-effort: a write failure must never break the voice pipeline, so every
public method that touches disk swallows and logs its own ``OSError``.

The "silver" auto-label (§3.1: "tool muvaffaqiyatli bajarilgan va
foydalanuvchi 10 s ichida qayta aytmagan yoki bekor qilmagan bo'lsa") is
implemented as a two-step commit: ``begin_silver_window()`` opens a 10s
window for a record whose command just finished with a successful tool call,
and ``note_new_utterance()`` -- called for *every* subsequent recognized
utterance, whatever it turns out to be -- invalidates that window if the new
utterance is a cancel phrase or a close repeat of the pending command's text
(the practical proxy for "the previous attempt was probably wrong"). Only
one window is tracked at a time, matching the orchestrator's single-flight
command execution.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from rapidfuzz import fuzz

_SAMPLE_RATE = 16000
SILVER_WINDOW_S = 10.0
_REPEAT_FUZZY_THRESHOLD = 75
_RETENTION_DAYS = 60
_CONFIRMED_LABELS = frozenset({"correct", "corrected"})


@dataclass
class UtteranceRecord:
    id: str
    audio_path: Path
    day_dir: Path
    text: str
    confidence: float
    model: str
    wake_source: str
    ts: str


@dataclass
class PendingSilver:
    record: UtteranceRecord
    invalidated: bool = False


class UtteranceLog:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)
        self._pending: PendingSilver | None = None

    def capture(
        self,
        pcm: np.ndarray,
        *,
        text: str,
        confidence: float,
        model: str,
        wake_source: str,
    ) -> UtteranceRecord | None:
        """Writes the audio immediately (so it survives a crash before the
        command finishes) and returns a handle for the later ``finalize_*``
        call. Returns ``None`` -- and logs -- on any write failure."""
        rec_id = uuid.uuid4().hex
        day_dir = self._base_dir / date.today().isoformat()
        try:
            day_dir.mkdir(parents=True, exist_ok=True)
            audio_path = day_dir / f"{rec_id}.flac"
            sf.write(audio_path, pcm, _SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        except OSError as exc:
            logger.warning("Utterance audio yozib bo'lmadi: {}", exc)
            return None
        return UtteranceRecord(
            id=rec_id,
            audio_path=audio_path,
            day_dir=day_dir,
            text=text,
            confidence=confidence,
            model=model,
            wake_source=wake_source,
            ts=datetime.now().isoformat(timespec="milliseconds"),
        )

    def note_new_utterance(self, normalized_text: str, *, is_cancel: bool) -> None:
        pending = self._pending
        if pending is None or pending.invalidated:
            return
        prior = pending.record.text.lower().strip(" .!?")
        if is_cancel or fuzz.ratio(normalized_text, prior) >= _REPEAT_FUZZY_THRESHOLD:
            pending.invalidated = True

    def finalize_immediate(
        self, record: UtteranceRecord, *, tool_called: bool, tool_ok: bool | None
    ) -> None:
        self._append(record, tool_called=tool_called, tool_ok=tool_ok, label="unlabeled")

    def begin_silver_window(self, record: UtteranceRecord) -> PendingSilver:
        pending = PendingSilver(record)
        self._pending = pending
        return pending

    def resolve_silver_window(self, pending: PendingSilver) -> None:
        """Call after waiting ``SILVER_WINDOW_S`` since ``begin_silver_window``."""
        if self._pending is pending:
            self._pending = None
        label = "unlabeled" if pending.invalidated else "silver"
        self._append(pending.record, tool_called=True, tool_ok=True, label=label)

    def _append(
        self, record: UtteranceRecord, *, tool_called: bool, tool_ok: bool | None, label: str
    ) -> None:
        row = {
            "id": record.id,
            "ts": record.ts,
            "path": record.audio_path.name,
            "stt_text": record.text,
            "confidence": record.confidence,
            "model": record.model,
            "wake_source": record.wake_source,
            "tool_called": tool_called,
            "tool_ok": tool_ok,
            "label": label,
        }
        try:
            with (record.day_dir / "index.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Utterance yozuvini index.jsonl'ga yozib bo'lmadi: {}", exc)

    def cleanup_stale(self, *, retention_days: int = _RETENTION_DAYS) -> int:
        """§3.1's "60 kundan eski va tasdiqlanmagan yozuvlar o'chiriladi":
        deletes the audio + index row for any entry older than
        ``retention_days`` whose label was never confirmed by
        ``scripts/label_tool.py`` (still "unlabeled"/"silver"). Entries
        labeled "correct"/"corrected" are kept indefinitely; "discarded" ones
        are removed on sight regardless of age. Best-effort: one unreadable
        day-folder is skipped, not fatal."""
        if not self._base_dir.is_dir():
            return 0
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = 0
        for day_dir in self._base_dir.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                day = datetime.strptime(day_dir.name, "%Y-%m-%d")
            except ValueError:
                continue
            index_path = day_dir / "index.jsonl"
            if not index_path.is_file():
                continue
            try:
                lines = index_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue

            kept: list[str] = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                label = row.get("label", "unlabeled")
                stale = day < cutoff and label not in _CONFIRMED_LABELS
                if label == "discarded" or stale:
                    with contextlib.suppress(OSError):
                        (day_dir / row["path"]).unlink()
                    deleted += 1
                    continue
                kept.append(line)

            with contextlib.suppress(OSError):
                if kept:
                    index_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
                else:
                    index_path.unlink(missing_ok=True)
                    if not any(day_dir.iterdir()):
                        day_dir.rmdir()

        if deleted:
            logger.info("{} ta eski/tasdiqlanmagan utterance yozuvi tozalandi", deleted)
        return deleted
