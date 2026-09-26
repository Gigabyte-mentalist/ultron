"""Model download skeleton (docs/tz/01_muhit_sozlash.md §4.7).

``--vad`` is implemented by the 02-audio stage (copies the bundled Silero
VAD ONNX graph into ``data/models/vad/``). The remaining flags are filled in
by their matching later stage (03 STT, 04 wake word -- the openWakeWord
model also under ``data/models/wake/``, 05 TTS); until then they only
ensure their folder exists.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from ultron.audio.vad import default_model_path
from ultron.config import load_config

_MODEL_SUBDIRS = ("stt", "tts", "wake", "vad")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron model yuklovchisi")
    parser.add_argument("--stt", action="store_true", help="STT modelini yuklash (03-bosqich)")
    parser.add_argument("--tts", action="store_true", help="TTS modelini yuklash (05-bosqich)")
    parser.add_argument(
        "--vad", action="store_true", help="Silero VAD ONNX modelini nusxalash (02-bosqich)"
    )
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


def download_vad_model(models_root: Path) -> None:
    import silero_vad

    src = Path(silero_vad.__file__).resolve().parent / "data" / "silero_vad.onnx"
    dst = default_model_path(models_root.parent)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    logger.info("Silero VAD modeli nusxalandi: {}", dst)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    args = parse_args(argv)
    config = load_config()
    data_dir = Path(config.app.data_dir)
    models_root = data_dir / "models"
    created = ensure_model_dirs(models_root)

    logger.info("Model papkalari tayyor:")
    for directory in created:
        logger.info("  - {}", directory)

    if args.vad or args.all:
        download_vad_model(models_root)

    if args.stt or args.tts:
        logger.info("Haqiqiy model yuklash logikasi 03/05-TZ bosqichlarida qo'shiladi.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
