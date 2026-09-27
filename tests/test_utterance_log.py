"""Tests for ultron.memory.utterance_log.UtteranceLog (docs/tz/11_uzbek_stt_finetune.md §3.1).

Pure file-I/O + in-memory state, no asyncio/event loop involved -- the 10s
"silver" window's actual waiting happens in the orchestrator
(``asyncio.sleep`` + ``resolve_silver_window``), so it's exercised here as two
separate synchronous calls with no delay, and end-to-end (including the real
wait) in ``tests/test_orchestrator_fsm.py``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from ultron.memory.utterance_log import UtteranceLog

_SR = 16000


def _pcm(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(seconds * _SR), dtype=np.int16)


def _read_index(day_dir: Path) -> list[dict]:
    lines = (day_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_capture_writes_flac_and_returns_record(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path)

    record = log.capture(
        _pcm(), text="telegramni och", confidence=0.9, model="ultron-uz-v1", wake_source="stt_kws"
    )

    assert record is not None
    assert record.audio_path.is_file()
    assert record.audio_path.suffix == ".flac"
    assert record.day_dir == tmp_path / datetime.date.today().isoformat()


def test_finalize_immediate_writes_unlabeled_row(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path)
    record = log.capture(
        _pcm(), text="bekor qil", confidence=0.8, model="m", wake_source="follow_up"
    )
    assert record is not None

    log.finalize_immediate(record, tool_called=False, tool_ok=None)

    rows = _read_index(record.day_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["stt_text"] == "bekor qil"
    assert row["confidence"] == 0.8
    assert row["model"] == "m"
    assert row["wake_source"] == "follow_up"
    assert row["tool_called"] is False
    assert row["tool_ok"] is None
    assert row["label"] == "unlabeled"
    assert row["path"] == record.audio_path.name


def test_silver_window_without_invalidation_becomes_silver(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path)
    record = log.capture(
        _pcm(), text="telegramni och", confidence=0.9, model="m", wake_source="stt_kws"
    )
    assert record is not None

    pending = log.begin_silver_window(record)
    log.resolve_silver_window(pending)

    rows = _read_index(record.day_dir)
    assert rows[0]["label"] == "silver"
    assert rows[0]["tool_called"] is True
    assert rows[0]["tool_ok"] is True


def test_cancel_phrase_invalidates_pending_silver_window(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path)
    record = log.capture(
        _pcm(), text="telegramni och", confidence=0.9, model="m", wake_source="stt_kws"
    )
    assert record is not None
    pending = log.begin_silver_window(record)

    log.note_new_utterance("bekor qil", is_cancel=True)
    log.resolve_silver_window(pending)

    rows = _read_index(record.day_dir)
    assert rows[0]["label"] == "unlabeled"


def test_repeated_command_invalidates_pending_silver_window(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path)
    record = log.capture(
        _pcm(), text="telegramni och", confidence=0.4, model="m", wake_source="stt_kws"
    )
    assert record is not None
    pending = log.begin_silver_window(record)

    # Same command, said again shortly after -- the clearest available sign
    # the first attempt was misheard/failed, even though the tool "succeeded".
    log.note_new_utterance("telegramni och", is_cancel=False)
    log.resolve_silver_window(pending)

    rows = _read_index(record.day_dir)
    assert rows[0]["label"] == "unlabeled"


def test_unrelated_next_command_does_not_invalidate_silver_window(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path)
    record = log.capture(
        _pcm(), text="telegramni och", confidence=0.9, model="m", wake_source="stt_kws"
    )
    assert record is not None
    pending = log.begin_silver_window(record)

    log.note_new_utterance("soat nechi", is_cancel=False)
    log.resolve_silver_window(pending)

    rows = _read_index(record.day_dir)
    assert rows[0]["label"] == "silver"


def test_note_new_utterance_without_pending_window_is_a_noop(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path)
    log.note_new_utterance("bekor qil", is_cancel=True)  # must not raise


def test_cleanup_stale_removes_old_unconfirmed_but_keeps_confirmed_and_recent(
    tmp_path: Path,
) -> None:
    log = UtteranceLog(tmp_path)
    old_day = (datetime.date.today() - datetime.timedelta(days=61)).isoformat()
    recent_day = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

    old_dir = tmp_path / old_day
    old_dir.mkdir(parents=True)
    for name in ("a", "b", "c", "d"):
        (old_dir / f"{name}.flac").write_bytes(b"fake")
    rows = [
        {"id": "a", "path": "a.flac", "label": "unlabeled"},
        {"id": "b", "path": "b.flac", "label": "silver"},
        {"id": "c", "path": "c.flac", "label": "correct"},
        {"id": "d", "path": "d.flac", "label": "corrected"},
    ]
    (old_dir / "index.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    recent_dir = tmp_path / recent_day
    recent_dir.mkdir(parents=True)
    (recent_dir / "e.flac").write_bytes(b"fake")
    (recent_dir / "index.jsonl").write_text(
        json.dumps({"id": "e", "path": "e.flac", "label": "unlabeled"}) + "\n", encoding="utf-8"
    )

    deleted = log.cleanup_stale()

    assert deleted == 2  # old "unlabeled" + old "silver"
    assert not (old_dir / "a.flac").exists()
    assert not (old_dir / "b.flac").exists()
    assert (old_dir / "c.flac").exists()
    assert (old_dir / "d.flac").exists()
    remaining = _read_index(old_dir)
    assert {r["id"] for r in remaining} == {"c", "d"}
    assert (recent_dir / "e.flac").exists()  # too recent, kept regardless of label


def test_cleanup_stale_removes_discarded_regardless_of_age(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path)
    recent_dir = tmp_path / (datetime.date.today().isoformat())
    recent_dir.mkdir(parents=True)
    (recent_dir / "x.flac").write_bytes(b"fake")
    (recent_dir / "index.jsonl").write_text(
        json.dumps({"id": "x", "path": "x.flac", "label": "discarded"}) + "\n", encoding="utf-8"
    )

    deleted = log.cleanup_stale()

    assert deleted == 1
    assert not (recent_dir / "x.flac").exists()


def test_cleanup_stale_on_missing_base_dir_returns_zero(tmp_path: Path) -> None:
    log = UtteranceLog(tmp_path / "does_not_exist")
    assert log.cleanup_stale() == 0
