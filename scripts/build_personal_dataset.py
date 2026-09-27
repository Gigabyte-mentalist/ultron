"""Build the personal STT train/dev/test split (docs/tz/11_uzbek_stt_finetune.md §3.2).

    uv run python scripts/build_personal_dataset.py
    uv run python scripts/build_personal_dataset.py --test-days 10 --dev-days 5

Reads every ``data/recordings/utterances/YYYY-MM-DD/index.jsonl`` written by
``ultron.memory.utterance_log`` (§3.1) and reviewed via ``scripts/label_tool.py``
(§3.2), keeps only the rows a human confirmed (label "correct"/"corrected"),
runs each transcript through ``ultron.text.normalize.normalize_transcript``
(§3.2: "barcha transkriptlar ... bir xil ko'rinishga keltiriladi" -- the same
rule TZ-03's benchmark and TZ-11's open-dataset prep both use), and splits by
**date** rather than randomly: this is a single speaker, so a random split
would leak near-duplicate utterances (the same phrase said twice one minute
apart) across train/test and inflate the eval score. The most recent
``--test-days`` calendar days become the test set, the ``--dev-days`` before
that become dev, and everything older becomes train.

Output: ``data/datasets/personal/{train,dev,test}.jsonl``, one row each:
``{"audio_path": "recordings/utterances/<day>/<id>.flac"  (relative to
data_dir, so training/stt/'s WSL scripts can resolve it under their own
mount), "text": ..., "duration_s": ..., "date": "YYYY-MM-DD"}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf
from loguru import logger

from ultron.config import ConfigError, RootConfig, load_config
from ultron.text.normalize import normalize_transcript

_CONFIRMED_LABELS = frozenset({"correct", "corrected"})


@dataclass
class Row:
    audio_path: str
    text: str
    duration_s: float
    date: str


def _collect_confirmed_rows(base_dir: Path) -> list[Row]:
    rows: list[Row] = []
    if not base_dir.is_dir():
        return rows
    for day_dir in sorted(base_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        index_path = day_dir / "index.jsonl"
        if not index_path.is_file():
            continue
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("label") not in _CONFIRMED_LABELS:
                continue
            raw_text = record.get("final_text") or record.get("stt_text") or ""
            text = normalize_transcript(raw_text)
            if not text:
                continue
            audio_file = day_dir / record["path"]
            if not audio_file.is_file():
                logger.warning("Audio fayl topilmadi, o'tkazib yuborildi: {}", audio_file)
                continue
            duration_s = sf.info(audio_file).duration
            rows.append(
                Row(
                    audio_path=f"recordings/utterances/{day_dir.name}/{record['path']}",
                    text=text,
                    duration_s=duration_s,
                    date=day_dir.name,
                )
            )
    return rows


def split_rows(
    rows: list[Row], *, test_days: int, dev_days: int
) -> tuple[list[Row], list[Row], list[Row]]:
    unique_days = sorted({row.date for row in rows})
    test_day_set = set(unique_days[-test_days:]) if test_days > 0 else set()
    remaining_days = [d for d in unique_days if d not in test_day_set]
    dev_day_set = set(remaining_days[-dev_days:]) if dev_days > 0 else set()

    train, dev, test = [], [], []
    for row in rows:
        if row.date in test_day_set:
            test.append(row)
        elif row.date in dev_day_set:
            dev.append(row)
        else:
            train.append(row)
    return train, dev, test


def _write_split(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                json.dumps(
                    {
                        "audio_path": row.audio_path,
                        "text": row.text,
                        "duration_s": row.duration_s,
                        "date": row.date,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _summarize(name: str, rows: list[Row]) -> str:
    hours = sum(row.duration_s for row in rows) / 3600
    days = len({row.date for row in rows})
    return f"{name}: {len(rows)} ta gap, {hours:.2f} soat, {days} kun"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron shaxsiy STT dataset qurish")
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--dev-days", type=int, default=7)
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    args = parse_args(argv)
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        try:
            config: RootConfig = load_config()
        except ConfigError as exc:
            logger.info("Config yuklab bo'lmadi:\n{}", exc)
            return 1
        data_dir = Path(config.app.data_dir)

    base_dir = data_dir / "recordings" / "utterances"
    rows = _collect_confirmed_rows(base_dir)
    if not rows:
        logger.info(
            "Tasdiqlangan yozuv topilmadi. Avval ishga tushiring: "
            "uv run python scripts/label_tool.py"
        )
        return 1

    unique_days = len({row.date for row in rows})
    if unique_days <= args.test_days:
        logger.warning(
            "Faqat {} kunlik ma'lumot bor ({} kun so'ralgan test uchun) -- "
            "bo'linish juda notekis bo'lishi mumkin, iloji boricha ko'proq kun "
            "to'plagandan keyin qayta ishga tushiring.",
            unique_days,
            args.test_days,
        )

    train, dev, test = split_rows(rows, test_days=args.test_days, dev_days=args.dev_days)

    out_dir = data_dir / "datasets" / "personal"
    _write_split(out_dir / "train.jsonl", train)
    _write_split(out_dir / "dev.jsonl", dev)
    _write_split(out_dir / "test.jsonl", test)

    logger.info("Yozildi: {}", out_dir)
    logger.info(_summarize("train", train))
    logger.info(_summarize("dev", dev))
    logger.info(_summarize("test", test))
    return 0


if __name__ == "__main__":
    sys.exit(main())
