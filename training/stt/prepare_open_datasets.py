"""Open Uzbek speech corpora for the LoRA mix (docs/tz/11_uzbek_stt_finetune.md §3.3).

Run inside the training venv (``bash setup_env.sh`` first):

    python prepare_open_datasets.py

Downloads and prepares two corpora:
  - Mozilla Common Voice uz (``mozilla-foundation/common_voice_17_0``, config
    "uz") -- CC0. **Gated on HF**: before this will work you must (once) open
    that dataset's page while logged into a HF account, click "agree" on its
    terms, then run ``huggingface-cli login`` (or set ``HF_TOKEN``) in this
    venv. If that friction is unwanted, pass ``--common-voice-id`` pointing at
    an ungated mirror with the same ``audio``/``sentence`` columns (e.g.
    ``fsicoli/common_voice_17_0``) -- verify its license note matches CC0
    first.
  - Google FLEURS uz_uz (``google/fleurs``, config "uz_uz") -- CC-BY-4.0,
    ungated.

Both are >=16kHz audio with a transcript column; this script resamples to
16kHz mono, runs every transcript through ``ultron.text.normalize.
normalize_transcript`` (the same normalization TZ-03's benchmark and
``scripts/build_personal_dataset.py`` both use, per §3.3's "bir xil ko'rinishga
keltiriladi"), and writes:

  - ``open_datasets/{train,dev}.jsonl`` + ``open_datasets/audio/*.flac`` --
    CV train/validation + FLEURS train/validation, for ``build_mix_manifest.py``.
  - ``<data_dir>/datasets/fleurs_uz_test/{test.jsonl,audio/*.flac}`` -- FLEURS'
    *test* split only, written straight to the Windows-visible data_dir so
    ``scripts/finetune_eval.py`` (Windows main venv, no HF/network access
    needed) can measure general-Uzbek regression per §3.5.

FLEURS' test split is never mixed into training -- it's held out specifically
to catch catastrophic forgetting from over-fitting to one personal voice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_PROJECT_SRC = "/mnt/c/ultron/src"


def _normalize_fn(project_src: str):
    """Imports ``ultron.text.normalize`` from the main project's ``src/``
    (visible from WSL under ``/mnt/c/...``). Only needs ``loguru`` -- see
    ``ultron/_cuda_dlls.py``'s module docstring for why that's the sole
    transitive import cost of touching ``ultron/__init__.py`` at all."""
    sys.path.insert(0, project_src)
    from ultron.text.normalize import normalize_transcript

    return normalize_transcript


def _prepare_split(
    dataset, *, text_column: str, out_dir: Path, prefix: str, normalize
) -> list[dict]:
    import soundfile as sf

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, example in enumerate(dataset):
        text = normalize(example[text_column] or "")
        if not text:
            continue
        audio = example["audio"]
        filename = f"{prefix}_{i:06d}.flac"
        sf.write(audio_dir / filename, audio["array"], audio["sampling_rate"], format="FLAC")
        rows.append({"audio_path": f"audio/{filename}", "text": text})
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(
    *,
    out_dir: Path,
    data_dir: Path,
    common_voice_id: str,
    max_per_split: int | None,
    project_src: str,
    skip_common_voice: bool,
) -> None:
    from datasets import Audio, load_dataset
    from loguru import logger

    normalize = _normalize_fn(project_src)

    train_rows: list[dict] = []
    dev_rows: list[dict] = []

    if not skip_common_voice:
        logger.info("Common Voice ({}) yuklanmoqda...", common_voice_id)
        for split, bucket in (("train", train_rows), ("validation", dev_rows)):
            cv = load_dataset(common_voice_id, "uz", split=split, trust_remote_code=True)
            if max_per_split is not None:
                cv = cv.select(range(min(max_per_split, len(cv))))
            cv = cv.cast_column("audio", Audio(sampling_rate=16000))
            bucket.extend(
                _prepare_split(
                    cv,
                    text_column="sentence",
                    out_dir=out_dir,
                    prefix=f"cv_{split}",
                    normalize=normalize,
                )
            )
            logger.info("Common Voice {}: {} ta gap", split, len(bucket))
    else:
        logger.info("Common Voice o'tkazib yuborildi (--skip-common-voice).")

    logger.info("FLEURS (uz_uz) yuklanmoqda...")
    for split, bucket in (("train", train_rows), ("validation", dev_rows)):
        fleurs = load_dataset("google/fleurs", "uz_uz", split=split, trust_remote_code=True)
        if max_per_split is not None:
            fleurs = fleurs.select(range(min(max_per_split, len(fleurs))))
        bucket.extend(
            _prepare_split(
                fleurs,
                text_column="raw_transcription",
                out_dir=out_dir,
                prefix=f"fleurs_{split}",
                normalize=normalize,
            )
        )
        logger.info("FLEURS {}: {} ta gap jami", split, len(bucket))

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "dev.jsonl", dev_rows)
    logger.info("Yozildi: {} (train={}, dev={})", out_dir, len(train_rows), len(dev_rows))

    logger.info("FLEURS test (Windows uchun, {}) tayyorlanmoqda...", data_dir)
    fleurs_test_dir = data_dir / "datasets" / "fleurs_uz_test"
    fleurs_test = load_dataset("google/fleurs", "uz_uz", split="test", trust_remote_code=True)
    if max_per_split is not None:
        fleurs_test = fleurs_test.select(range(min(max_per_split, len(fleurs_test))))
    test_rows_raw = _prepare_split(
        fleurs_test,
        text_column="raw_transcription",
        out_dir=fleurs_test_dir,
        prefix="fleurs_test",
        normalize=normalize,
    )
    # finetune_eval.py's loader expects audio_path relative to data_dir, not
    # to fleurs_test_dir -- rewrite the prefix accordingly.
    test_rows = [
        {"audio_path": f"datasets/fleurs_uz_test/{r['audio_path']}", "text": r["text"]}
        for r in test_rows_raw
    ]
    _write_jsonl(fleurs_test_dir / "test.jsonl", test_rows)
    logger.info("Yozildi: {} ({} ta gap)", fleurs_test_dir, len(test_rows))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ochiq o'zbekcha nutq korpuslarini tayyorlash")
    parser.add_argument("--out-dir", type=Path, default=Path("open_datasets"))
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/d/ultron_data"))
    parser.add_argument("--common-voice-id", default="mozilla-foundation/common_voice_17_0")
    parser.add_argument("--skip-common-voice", action="store_true")
    parser.add_argument(
        "--max-per-split", type=int, default=None, help="Sinov/disk uchun cheklash (ixtiyoriy)"
    )
    parser.add_argument("--project-src", default=_DEFAULT_PROJECT_SRC)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare(
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        common_voice_id=args.common_voice_id,
        max_per_split=args.max_per_split,
        project_src=args.project_src,
        skip_common_voice=args.skip_common_voice,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
