#!/usr/bin/env bash
# docs/tz/10_wake_word_training.md §3.1 -- run inside WSL2 Ubuntu, once.
#
#   wsl -d Ubuntu
#   cd /mnt/c/ultron/training/wake
#   bash setup_env.sh
#
# Everything lands in ~/oww_train/ (not /mnt/c/..., so it's on WSL's own
# ext4 filesystem -- see README.md §0 for why that matters here) and never
# touches the Windows-side project's own .venv.
set -euo pipefail

ROOT="$HOME/oww_train"
VENV="$ROOT/venv"
mkdir -p "$ROOT"
cd "$ROOT"

echo "== 1/6: nvidia-smi check =="
if ! nvidia-smi >/dev/null 2>&1; then
    echo "GPU ko'rinmayapti -- CPU'da ham ishlaydi, lekin ancha sekinroq." >&2
fi

echo "== 2/6: virtualenv =="
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip

echo "== 3/6: piper-sample-generator =="
# The automatic_model_training notebook's *actual* git clone command uses
# this fork (the YAML's own doc-comment names a different one -- see
# README.md §6's "farqlar" table).
if [ ! -d piper-sample-generator ]; then
    git clone https://github.com/rhasspy/piper-sample-generator
fi
mkdir -p piper-sample-generator/models
if [ ! -f piper-sample-generator/models/en_US-libritts_r-medium.pt ]; then
    wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt \
        'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
fi
pip install piper-phonemize webrtcvad

echo "== 4/6: openWakeWord (editable, from source -- train.py isn't in the PyPI wheel) =="
if [ ! -d openwakeword ]; then
    git clone https://github.com/dscripka/openWakeWord openwakeword
fi
pip install -e ./openwakeword

echo "== 5/6: training extras (versions pinned per the official notebook) =="
pip install \
    mutagen==1.47.0 \
    torchinfo==1.8.0 \
    torchmetrics==1.2.0 \
    speechbrain==0.5.14 \
    audiomentations==0.33.0 \
    torch-audiomentations==0.11.0 \
    acoustics==0.2.6 \
    tensorflow-cpu==2.8.1 \
    tensorflow_probability==0.16.0 \
    onnx_tf==1.10.0 \
    pronouncing==0.2.0 \
    datasets==2.14.6 \
    deep-phonemizer==0.0.19
# Torch itself isn't pinned by the notebook (it relies on Colab's preinstall).
# Install a CUDA build matching this machine's driver if not already present:
python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "== 6/6: shared feature models + negative/background datasets =="
# Same utility C:\ultron\scripts\download_models.py --oww already uses
# successfully (not the notebook's version-pinned wget links) -- see
# README.md §1.5.
python -c "
from openwakeword.utils import download_models
download_models(model_names=['__ultron_skip_official_models__'], target_directory='openwakeword/openwakeword/resources/models')
"

# Negative training/validation features (~2000h + ~11h, precomputed).
[ -f openwakeword_features_ACAV100M_2000_hrs_16bit.npy ] || wget \
    https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
[ -f validation_set_features.npy ] || wget \
    https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy

# Room impulse responses + background noise/music (small starter sets --
# see README.md §8 "Kengaytirish" to grow these for production quality).
pip install soundfile librosa
python - <<'PY'
import os
from datasets import load_dataset
import soundfile as sf
import numpy as np

os.makedirs("mit_rirs", exist_ok=True)
if not os.listdir("mit_rirs"):
    ds = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
    for i, row in enumerate(ds):
        audio = row["audio"]
        sf.write(f"mit_rirs/{i:04d}.wav", audio["array"], audio["sampling_rate"], subtype="PCM_16")
        if i >= 200:
            break
PY

mkdir -p audioset_16k
if [ ! "$(ls -A audioset_16k)" ]; then
    wget -O bal_train09.tar https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar
    mkdir -p audioset_raw && tar -xf bal_train09.tar -C audioset_raw
    python - <<'PY'
import glob
import os
import librosa
import soundfile as sf

os.makedirs("audioset_16k", exist_ok=True)
for path in glob.glob("audioset_raw/**/*.flac", recursive=True) + glob.glob("audioset_raw/**/*.wav", recursive=True):
    y, _ = librosa.load(path, sr=16000, mono=True)
    sf.write(f"audioset_16k/{os.path.basename(path)}.wav", y, 16000, subtype="PCM_16")
PY
    rm -f bal_train09.tar
fi

python - <<'PY'
import os
from datasets import load_dataset
import soundfile as sf

os.makedirs("fma", exist_ok=True)
if not os.listdir("fma"):
    ds = load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    seconds = 0
    for i, row in enumerate(ds):
        audio = row["audio"]
        sf.write(f"fma/{i:04d}.wav", audio["array"], audio["sampling_rate"], subtype="PCM_16")
        seconds += len(audio["array"]) / audio["sampling_rate"]
        if seconds >= 3600:  # ~1 hour starter set, see README.md §8 to grow this
            break
PY

echo "== Tayyor. Keyingi qadam: README.md §2-3 (namunalar) va §4 (o'qitish). =="
