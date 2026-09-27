"""Trains the personal openWakeWord "custom verifier" (docs/tz/04_wake_word.md
§3.3, docs/tz/10_wake_word_training.md §3.4).

    uv run python scripts/train_wake_verifier.py

Runs entirely in the main project's ``.venv`` (``openwakeword`` + scikit-learn,
both already installed) -- unlike the base ``altron.onnx`` model itself, this
does not need WSL/Piper/GPU training. It calls
``openwakeword.train_custom_verifier`` (a documented, official API: a small
logistic-regression classifier on top of the base model's own embeddings,
distinguishing *your* voice saying the wake word from everything else) and
writes ``data/models/wake/altron_verifier.pkl``.

Requires ``data/models/wake/altron.onnx`` to already exist (docs/tz/10 §4) --
the verifier is trained *on top of* that base model, not standalone.

The library's own guidance (docs/custom_verifier_models.md) is to use only a
handful of positive examples ("it's not advised to train on a large number
... the capacity of the verifier model is small") even though hundreds of
real positives exist by this stage -- ``--max-positive`` caps how many are
sampled (default 40).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from ultron.config import ConfigError, load_config

_DEFAULT_MAX_POSITIVE = 40
_DEFAULT_MAX_NEGATIVE = 60
_POSITIVE_SUBDIRS = ("wake", "wake_auto")
_NEGATIVE_SUBDIRS = ("wake_neg", "wake_hardneg")


def _collect_wavs(data_dir: Path, subdirs: tuple[str, ...], *, cap: int, seed: int) -> list[str]:
    paths = []
    for sub in subdirs:
        paths.extend(sorted((data_dir / "recordings" / sub).glob("*.wav")))
    if not paths:
        return []
    if len(paths) > cap:
        paths = random.Random(seed).sample(paths, cap)
    return [str(p) for p in paths]


def write_config_verifier_path(config_path: Path, *, verifier_rel_path: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    text, n = re.subn(
        r"(?m)^(wake:\n(?:[ \t].*\n)*?[ \t]+oww_verifier_path:\s*)\S.*$",
        lambda m: f'{m.group(1)}"{verifier_rel_path}"',
        text,
        count=1,
    )
    if n == 0:
        logger.warning(
            "config.yaml'dagi wake.oww_verifier_path qatori avtomatik topilmadi, qo'lda yozing: {}",
            verifier_rel_path,
        )
        return
    config_path.write_text(text, encoding="utf-8")
    logger.info("config.yaml yangilandi: wake.oww_verifier_path={}", verifier_rel_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron shaxsiy wake-word verifier o'qitish")
    parser.add_argument("--max-positive", type=int, default=_DEFAULT_MAX_POSITIVE)
    parser.add_argument("--max-negative", type=int, default=_DEFAULT_MAX_NEGATIVE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-write-config", action="store_true", help="config.yaml'ni o'zgartirmaslik"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    args = parse_args(argv)
    try:
        config = load_config()
    except ConfigError as exc:
        logger.info("Config yuklab bo'lmadi:\n{}", exc)
        return 1

    model_path = Path(config.wake.oww_model_path)
    if not model_path.is_file():
        raise SystemExit(
            f"Asosiy openWakeWord modeli topilmadi: {model_path}\n"
            "Avval docs/tz/10_wake_word_training.md bo'yicha o'qiting."
        )

    data_dir = Path(config.app.data_dir)
    positive_clips = _collect_wavs(
        data_dir, _POSITIVE_SUBDIRS, cap=args.max_positive, seed=args.seed
    )
    negative_clips = _collect_wavs(
        data_dir, _NEGATIVE_SUBDIRS, cap=args.max_negative, seed=args.seed
    )
    recordings_dir = data_dir / "recordings"
    if not positive_clips:
        raise SystemExit(f"Pozitiv namunalar topilmadi: {recordings_dir} / {_POSITIVE_SUBDIRS}")
    if not negative_clips:
        raise SystemExit(f"Negativ namunalar topilmadi: {recordings_dir} / {_NEGATIVE_SUBDIRS}")
    logger.info(
        "{} ta pozitiv, {} ta negativ namuna bilan verifier o'qitilmoqda...",
        len(positive_clips),
        len(negative_clips),
    )

    from openwakeword import train_custom_verifier

    output_path = model_path.parent / "altron_verifier.pkl"
    train_custom_verifier(
        positive_reference_clips=positive_clips,
        negative_reference_clips=negative_clips,
        output_path=str(output_path),
        model_name=str(model_path),
    )
    logger.info("Verifier yozildi: {}", output_path)

    if not args.no_write_config:
        rel_path = f"data/models/wake/{output_path.name}"
        write_config_verifier_path(Path("config/config.yaml"), verifier_rel_path=rel_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
