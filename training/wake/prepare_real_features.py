"""Builds ``real_positive_features.npy`` for configs/altron_v2.yml's
``feature_data_files.real_positive`` (docs/tz/10_wake_word_training.md §3.3).

Run inside WSL after ``setup_env.sh`` (needs the ``openwakeword`` editable
install, not just the pip package):

    cd ~/oww_train
    source venv/bin/activate
    python /mnt/c/ultron/training/wake/prepare_real_features.py \\
        --recordings-dir /mnt/d/ultron_data/recordings \\
        --output real_positive_features.npy

**This mechanism is not part of openWakeWord's documented workflow** -- see
training/wake/README.md §3/§6. It's composed from two public functions that
exist for a different documented purpose (building the shared ACAV100M
negative-feature file): ``openwakeword.data.load_audio_clips`` and
``openwakeword.utils.compute_features_from_generator``. Treat
``altron_v1.onnx`` (synthetic-only, fully within the documented path) as the
safe fallback if v2 (which uses this file) behaves oddly.

Real "Altron" recordings from ``record_samples.py wake``/``wake_auto`` are
VAD-endpointed clips that end shortly after speech ends but can have a
variable amount of *leading* silence (the endpointer's pre-roll). Rather
than guess where the word starts, every clip is cropped/padded to
``--clip-seconds`` from its *end* -- consistent across files without needing
per-file speech-boundary detection, at the cost of occasionally trimming
the very start of a slowly-spoken word if the file's pre-roll was unusually
long.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import soundfile as sf

_SAMPLE_RATE = 16000
_DEFAULT_CLIP_SECONDS = 1.6  # rough match to Piper's typical 1-2 word clip length


def _find_wavs(recordings_dir: Path, subdirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for sub in subdirs:
        paths.extend(sorted(Path(p) for p in glob.glob(str(recordings_dir / sub / "*.wav"))))
    return paths


def _load_and_crop(path: Path, clip_size: int) -> np.ndarray:
    pcm, sr = sf.read(path, dtype="int16")
    if sr != _SAMPLE_RATE:
        raise SystemExit(f"Kutilmagan chastota {sr} Hz: {path} (16 kHz kerak)")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if len(pcm) >= clip_size:
        return pcm[-clip_size:]
    out = np.zeros(clip_size, dtype=np.int16)
    out[-len(pcm) :] = pcm
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        required=True,
        help="e.g. /mnt/d/ultron_data/recordings (config.app.data_dir/recordings on Windows)",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["wake", "wake_auto"],
        help="Positive-sample subfolders to include (default: wake, wake_auto)",
    )
    parser.add_argument("--clip-seconds", type=float, default=_DEFAULT_CLIP_SECONDS)
    parser.add_argument("--output", type=Path, default=Path("real_positive_features.npy"))
    args = parser.parse_args(argv)

    wav_paths = _find_wavs(args.recordings_dir, args.subdirs)
    if not wav_paths:
        raise SystemExit(
            f"Hech qanday .wav topilmadi: {args.recordings_dir} / {args.subdirs}\n"
            "Avval Windows tomonida yozib oling: scripts/record_samples.py wake --count 300"
        )
    print(f"{len(wav_paths)} ta real pozitiv fayl topildi.")

    from openwakeword.utils import compute_features_from_generator

    clip_size = int(args.clip_seconds * _SAMPLE_RATE)
    clips = np.stack([_load_and_crop(p, clip_size) for p in wav_paths])

    def _batches() -> object:
        batch_size = 64
        for i in range(0, len(clips), batch_size):
            yield clips[i : i + batch_size]

    compute_features_from_generator(
        _batches(),
        n_total=len(clips),
        clip_duration=clip_size,
        output_file=str(args.output),
    )
    print(f"Yozildi: {args.output}")
    print(
        "configs/altron_v2.yml'da feature_data_files.real_positive shu faylni "
        "ko'rsatishi kerak (default nomi mos keladi)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
