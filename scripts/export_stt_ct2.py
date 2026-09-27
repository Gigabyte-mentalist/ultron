"""Export a merged LoRA fine-tune to CTranslate2 (docs/tz/11_uzbek_stt_finetune.md §3.5).

    uv run python scripts/export_stt_ct2.py \
        --merged-model D:/ultron_data/models/stt/ultron-uz-v1-merged --version 1

``training/stt/merge_lora.py`` (WSL) merges the trained LoRA adapter into the
base Whisper weights and saves a plain HF-format checkpoint directly under
``data_dir`` (visible from Windows). This script converts that checkpoint to
CTranslate2 int8_float16 -- the exact same ``ctranslate2.converters.
TransformersConverter`` call ``scripts/download_models.py``'s
``convert_uzbek_finetune`` already uses for the community fine-tunes TZ-03
picked, just pointed at a local directory instead of a HF repo id.

Output: ``data/models/stt/ultron-uz-v<version>-ct2/``. Prints the
``finetune_eval.py`` command to run next rather than calling it automatically
-- each step here is independently re-runnable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from ultron.config import ConfigError, load_config


def convert_merged_checkpoint(merged_model_dir: Path, ct2_dir: Path) -> None:
    from ctranslate2.converters import TransformersConverter

    if not merged_model_dir.is_dir():
        raise SystemExit(f"Merged model topilmadi: {merged_model_dir}")
    if (ct2_dir / "model.bin").is_file():
        logger.info("Allaqachon konvertatsiya qilingan, ustidan yoziladi: {}", ct2_dir)

    logger.info(
        "CTranslate2 formatiga o'girilmoqda (int8_float16): {} -> {}", merged_model_dir, ct2_dir
    )
    converter = TransformersConverter(
        str(merged_model_dir),
        copy_files=["preprocessor_config.json", "tokenizer.json"],
        load_as_float16=True,
    )
    converter.convert(str(ct2_dir), quantization="int8_float16", force=True)
    logger.info("Tayyor: {}", ct2_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron STT LoRA fine-tune -> CTranslate2 eksport")
    parser.add_argument("--merged-model", type=Path, required=True)
    parser.add_argument("--version", required=True, help='masalan "1" -> ultron-uz-v1-ct2')
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
            config = load_config()
        except ConfigError as exc:
            logger.info("Config yuklab bo'lmadi:\n{}", exc)
            return 1
        data_dir = Path(config.app.data_dir)

    ct2_dir = data_dir / "models" / "stt" / f"ultron-uz-v{args.version}-ct2"
    convert_merged_checkpoint(args.merged_model, ct2_dir)

    logger.info("")
    logger.info("Keyingi qadam:")
    logger.info("  uv run python scripts/finetune_eval.py --new-model {}", ct2_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
