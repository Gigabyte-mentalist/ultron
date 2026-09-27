"""Model download skeleton (docs/tz/01_muhit_sozlash.md §4.7).

``--vad`` is implemented by the 02-audio stage (copies the bundled Silero
VAD ONNX graph into ``data/models/vad/``). ``--stt`` is implemented by the
03-STT stage: it pre-downloads the standard faster-whisper checkpoints used
by ``scripts/benchmark_stt.py`` and converts the two Uzbek fine-tunes chosen
in docs/tz/03_stt.md to CTranslate2 (int8_float16). ``--oww`` is implemented
by the 04-wake-word stage: it downloads openWakeWord's *shared* feature
models (melspectrogram + audio embedding, ONNX) into ``data/models/wake/``,
which is where ``wake/oww_detector.py`` expects to find them next to the
custom keyword model trained in docs/tz/10_wake_word_training.md (not yet
run, so ``--oww`` alone does not make wake.mode="oww" usable -- that needs
the trained model file too). ``--tts`` is implemented by the 05-TTS stage:
it pre-downloads the two Uzbek MMS-TTS candidates compared in
``scripts/tts_compare.py`` into ``data/models/tts`` (the ``AutoTokenizer``/
``VitsModel`` HF cache dir that ``MMSEngine`` also loads from, so this just
warms the cache -- ``MMSEngine.load()`` works without it too, just slower on
first run).
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
        "--oww",
        action="store_true",
        help="openWakeWord umumiy feature modellarini yuklash (04-bosqich)",
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


# A bogus name that cannot match any official pretrained model: passing it as
# ``model_names`` to openwakeword's downloader skips fetching those (we only
# want the two shared feature-extraction models it always fetches first).
_NO_OFFICIAL_MODELS = ["__ultron_skip_official_models__"]


def download_oww_shared_models(wake_dir: Path) -> None:
    """Fetch openWakeWord's shared melspectrogram/embedding ONNX models.

    Downloaded next to ``wake.oww_model_path`` (not into
    ``data_dir/models/wake`` and not into openwakeword's own site-packages
    resources dir) so ``oww_detector.py`` -- which looks for
    ``melspectrogram.onnx``/``embedding_model.onnx`` right beside the custom
    model file -- always finds them regardless of where ``config.local.yaml``
    points ``app.data_dir`` (``wake.oww_model_path`` is resolved against the
    project root, not against ``data_dir``, see ``config.py``'s
    ``_resolve_paths_in_place``).
    """
    from openwakeword.utils import download_models

    wake_dir.mkdir(parents=True, exist_ok=True)
    download_models(model_names=_NO_OFFICIAL_MODELS, target_directory=str(wake_dir))
    logger.info("openWakeWord feature modellari tayyor: {}", wake_dir)


# The two candidates compared in docs/tz/05_tts.md §3.1 / scripts/tts_compare.py.
_TTS_CANDIDATES = (
    "facebook/mms-tts-uzb-script_cyrillic",
    "MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic",
)


def download_tts_models(models_root: Path) -> None:
    from transformers import AutoTokenizer, VitsModel

    tts_dir = models_root / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    for repo_id in _TTS_CANDIDATES:
        logger.info("TTS modeli yuklanmoqda: {}", repo_id)
        AutoTokenizer.from_pretrained(repo_id, cache_dir=str(tts_dir))
        VitsModel.from_pretrained(repo_id, cache_dir=str(tts_dir))
    logger.info("TTS modellari tayyor: {}", tts_dir)


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

    if args.oww or args.all:
        download_oww_shared_models(config.wake.oww_model_path.parent)

    if args.tts or args.all:
        download_tts_models(models_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
