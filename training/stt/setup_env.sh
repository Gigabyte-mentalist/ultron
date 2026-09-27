#!/usr/bin/env bash
# docs/tz/11_uzbek_stt_finetune.md §3.4: training venv for the LoRA fine-tune,
# entirely separate from C:\ultron\.venv (uv) -- same split as
# training/wake/setup_env.sh, just a plain venv here since there's no
# from-source library to editable-install (peft/transformers/datasets are all
# on PyPI).
#
# Usage (inside WSL2, this machine already has Ubuntu-24.04.4 + GPU
# passthrough per training/wake/README.md §0 -- nothing to install there):
#   cd /mnt/c/ultron/training/stt
#   bash setup_env.sh

set -euo pipefail

ENV_DIR="${HOME}/stt_train"
mkdir -p "${ENV_DIR}"
cd "${ENV_DIR}"

if [ ! -d venv ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

pip install --upgrade pip

# CUDA-enabled torch first (matches the RTX 4060's driver; adjust the index
# URL if `nvidia-smi` inside WSL reports a different CUDA version than 12.x).
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install \
    transformers \
    peft \
    datasets \
    accelerate \
    bitsandbytes \
    evaluate \
    jiwer \
    soundfile \
    librosa \
    loguru \
    pyyaml

echo ""
echo "Tayyor. Muhitni faollashtirish: source ${ENV_DIR}/venv/bin/activate"
echo "Keyingi qadam: python prepare_open_datasets.py --help"
