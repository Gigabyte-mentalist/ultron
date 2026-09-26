"""Model download skeleton (docs/tz/01_muhit_sozlash.md §4.7).

Each later stage (03 STT, 04 wake word, 05 TTS) fills in the actual download
logic behind its matching flag. For now this only creates the
data/models/{stt,tts,wake} folders so later stages have somewhere to write.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from ultron.config import load_config

_MODEL_SUBDIRS = ("stt", "tts", "wake")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron model yuklovchisi")
    parser.add_argument("--stt", action="store_true", help="STT modelini yuklash (03-bosqich)")
    parser.add_argument("--tts", action="store_true", help="TTS modelini yuklash (05-bosqich)")
    parser.add_argument("--vad", action="store_true", help="VAD/wake modelini yuklash (04-bosqich)")
    parser.add_argument(
        "--all", action="store_true", help="Barchasini yuklash (keyingi bosqichlar)"
    )
    return parser.parse_args(argv)


def ensure_model_dirs(models_root: Path) -> list[Path]:
    created = []
    for name in _MODEL_SUBDIRS:
        subdir = models_root / name
        subdir.mkdir(parents=True, exist_ok=True)
        created.append(subdir)
    return created


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    args = parse_args(argv)
    config = load_config()
    models_root = Path(config.app.data_dir) / "models"
    created = ensure_model_dirs(models_root)

    logger.info("Model papkalari tayyor:")
    for directory in created:
        logger.info("  - {}", directory)

    if args.stt or args.tts or args.vad or args.all:
        logger.info("Haqiqiy model yuklash logikasi 03/04/05-TZ bosqichlarida qo'shiladi.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
