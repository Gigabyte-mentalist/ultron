"""Model download skeleton (docs/tz/01_muhit_sozlash.md §4.7).

``--vad`` is implemented by the 02-audio stage (copies the bundled Silero
VAD ONNX graph into ``data/models/vad/``). ``--stt`` is implemented by the
03-STT stage: it pre-downloads the standard faster-whisper checkpoints used
by ``scripts/benchmark_stt.py`` and converts the two Uzbek fine-tunes chosen
in docs/tz/03_stt.md to CTranslate2 (int8_float16). ``--tts`` and the
openWakeWord model (04/05 stages) are filled in by their matching later
stage; until then they only ensure their folder exists.
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


# Standard faster-whisper checkpoints compared in scripts/benchmark_stt.py
# (docs/tz/03_stt.md §3.5). faster_whisper resolves these names to their
# Systran/mobiuslabsgmbh CTranslate2 repos on the Hub.
_STANDARD_STT_MODELS = ("large-v3-turbo", "large-v3", "medium", "small")

# Uzbek Whisper fine-tunes picked from the community models surveyed on
# Hugging Face (docs/tz/03_stt.md §3.5 asks for >= 2, license open, high
# downloads). Both are apache-2.0, medium-sized, and have the best
# self-reported WER/CER among the candidates found:
#   - Kotib/uzbek_stt_v1: ~1600h / 3-stage curriculum, WER 16.7% / CER 7.0%
#     across 8 held-out test sets, ~9.2k monthly downloads.
#   - islomov/rubaistt_v2_medium: ~475h mixed human+pseudo-labeled data,
#     WER ~17% / CER ~5.5% self-reported, ~6.1k monthly downloads, 27 likes.
# Neither ships CTranslate2 weights, so both need conversion below.
_UZBEK_FINETUNES = {
    "kotib-uzbek-stt-v1": "Kotib/uzbek_stt_v1",
    "islomov-rubaistt-v2-medium": "islomov/rubaistt_v2_medium",
}


def download_standard_stt_models(models_root: Path) -> None:
    from faster_whisper.utils import download_model

    stt_root = models_root / "stt"
    for name in _STANDARD_STT_MODELS:
        logger.info("STT modeli yuklanmoqda: {}", name)
        download_model(name, cache_dir=str(stt_root))


def convert_uzbek_finetune(repo_id: str, dest_name: str, models_root: Path) -> Path:
    """Download a transformers-format Uzbek Whisper fine-tune and convert it to
    CTranslate2 int8_float16 so faster-whisper can load it directly by path."""
    from ctranslate2.converters import TransformersConverter
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

    stt_root = models_root / "stt"
    hf_dir = stt_root / f"{dest_name}-hf"
    ct2_dir = stt_root / f"{dest_name}-ct2"

    if (ct2_dir / "model.bin").is_file():
        logger.info("Allaqachon konvertatsiya qilingan, o'tkazib yuborildi: {}", ct2_dir)
        return ct2_dir

    # Fetch model/tokenizer/feature-extractor via from_pretrained(repo_id) and
    # re-save locally, rather than huggingface_hub.snapshot_download: the
    # latter first lists the whole repo tree, an endpoint that has been
    # flaky/rate-limited for some repos. from_pretrained only fetches the
    # specific files each class needs. Re-saving also guarantees
    # tokenizer.json/preprocessor_config.json exist even when the source repo
    # only ships the slow tokenizer and no feature-extractor config.
    logger.info("HF repodan yuklanmoqda: {} -> {}", repo_id, hf_dir)
    WhisperForConditionalGeneration.from_pretrained(repo_id).save_pretrained(hf_dir)
    AutoTokenizer.from_pretrained(repo_id).save_pretrained(hf_dir)
    WhisperFeatureExtractor.from_pretrained(repo_id).save_pretrained(hf_dir)

    logger.info("CTranslate2 formatiga o'girilmoqda (int8_float16): {}", ct2_dir)
    converter = TransformersConverter(
        str(hf_dir),
        copy_files=["preprocessor_config.json", "tokenizer.json"],
        load_as_float16=True,
    )
    converter.convert(str(ct2_dir), quantization="int8_float16", force=True)
    logger.info("Tayyor: {}", ct2_dir)
    return ct2_dir


def download_uzbek_finetunes(models_root: Path) -> list[Path]:
    return [
        convert_uzbek_finetune(repo_id, dest_name, models_root)
        for dest_name, repo_id in _UZBEK_FINETUNES.items()
    ]


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

    if args.stt or args.all:
        download_standard_stt_models(models_root)
        download_uzbek_finetunes(models_root)

    if args.tts:
        logger.info("Haqiqiy TTS model yuklash logikasi 05-TZ bosqichida qo'shiladi.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
