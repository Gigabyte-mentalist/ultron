"""Latency report from ``data/logs/metrics.jsonl`` (docs/tz/08_orchestrator.md §8B.5).

    uv run python scripts/metrics_report.py [--days 7] [--slowest 10]

Prints p50/p95 of ``total_to_first_audio_ms`` grouped by ``path``
("llm"/"fastpath") over the last N days, plus the slowest requests -- the
same numbers docs/tz/08_orchestrator.md §8B.7's acceptance criteria are
stated against (fast-path p95 <= 1.0 s, LLM path p95 <= 2.0 s).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from ultron.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron metrics.jsonl latency report")
    parser.add_argument("--days", type=int, default=7, help="Look back this many days (default 7)")
    parser.add_argument(
        "--metrics-path", type=Path, default=None, help="Override metrics.jsonl path"
    )
    parser.add_argument(
        "--slowest", type=int, default=10, help="How many slowest requests to list (default 10)"
    )
    return parser.parse_args(argv)


def _load_records(path: Path, cutoff: datetime) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            ts = datetime.fromisoformat(record["ts"])
        except (KeyError, ValueError, TypeError):
            continue
        if ts >= cutoff:
            records.append(record)
    return records


def _percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct * (len(ordered) - 1))))
    return ordered[idx]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    metrics_path = args.metrics_path
    if metrics_path is None:
        config = load_config()
        metrics_path = Path(config.app.data_dir) / "logs" / "metrics.jsonl"

    cutoff = datetime.now() - timedelta(days=args.days)
    records = _load_records(metrics_path, cutoff)
    if not records:
        logger.warning("Metrikalar topilmadi ({}, oxirgi {} kun)", metrics_path, args.days)
        return 1

    logger.info(
        "{} so'rov tahlil qilinmoqda (oxirgi {} kun, {})", len(records), args.days, metrics_path
    )

    by_path: dict[str, list[int]] = {}
    for record in records:
        latency = record.get("total_to_first_audio_ms")
        if latency is None:
            continue
        by_path.setdefault(record.get("path", "?"), []).append(latency)

    for path_name, latencies in sorted(by_path.items()):
        p50 = _percentile(latencies, 0.50)
        p95 = _percentile(latencies, 0.95)
        logger.info(
            "  {}: n={} p50={} ms p95={} ms (o'rtacha={:.0f} ms)",
            path_name,
            len(latencies),
            p50,
            p95,
            statistics.mean(latencies),
        )

    errors = [r for r in records if r.get("error")]
    if errors:
        logger.info("Xatoliklar: {} / {}", len(errors), len(records))

    timed = [r for r in records if r.get("total_to_first_audio_ms") is not None]
    slowest = sorted(timed, key=lambda r: r["total_to_first_audio_ms"], reverse=True)
    slowest = slowest[: args.slowest]
    logger.info("Eng sekin {} so'rov:", len(slowest))
    for record in slowest:
        logger.info(
            "  {} ms | path={} | wake={} | '{}'",
            record["total_to_first_audio_ms"],
            record.get("path"),
            record.get("wake_source"),
            (record.get("stt_text") or "")[:60],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
