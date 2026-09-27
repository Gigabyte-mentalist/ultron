#!/usr/bin/env bash
# docs/tz/11_uzbek_stt_finetune.md §3.3/§3.4 -- runs the full personal-STT
# LoRA pipeline for one config, in order:
#   1. prepare_open_datasets.py (skipped if already done -- see below)
#   2. build_mix_manifest.py
#   3. train_lora.py
#   4. merge_lora.py
#
#   bash run_training.sh configs/lora_v1.yaml
#
# Like training/wake/run_training.sh, this copies the config into the
# training venv's own directory (~/stt_train) and runs everything from there,
# so the config's relative paths (open_train, mix_train_out, output_dir, ...)
# resolve consistently across all four steps.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: bash run_training.sh configs/lora_v1.yaml" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_SRC="$SCRIPT_DIR/$1"
if [ ! -f "$CONFIG_SRC" ]; then
    echo "Config topilmadi: $CONFIG_SRC" >&2
    exit 1
fi

ROOT="$HOME/stt_train"
mkdir -p "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
cd "$ROOT"
cp "$CONFIG_SRC" ./active_config.yaml

DATA_DIR="$(python -c "import yaml; print(yaml.safe_load(open('active_config.yaml'))['paths']['data_dir'])")"

echo "== 1/4: prepare_open_datasets (idempotent -- skipped if already done) =="
if [ -f "open_datasets/train.jsonl" ]; then
    echo "open_datasets/train.jsonl allaqachon bor, o'tkazib yuborildi (o'chirib qayta ishga tushiring)."
else
    python "$SCRIPT_DIR/prepare_open_datasets.py" --out-dir open_datasets --data-dir "$DATA_DIR"
fi

echo "== 2/4: build_mix_manifest =="
python "$SCRIPT_DIR/build_mix_manifest.py" ./active_config.yaml

echo "== 3/4: train_lora =="
python "$SCRIPT_DIR/train_lora.py" ./active_config.yaml

echo "== 4/4: merge_lora =="
OUTPUT_DIR="$(python -c "import yaml; c=yaml.safe_load(open('active_config.yaml')); print(c['paths']['output_dir'])")"
BASE_MODEL="$(python -c "import yaml; print(yaml.safe_load(open('active_config.yaml'))['base_model'])")"
VERSION_NAME="$(basename "$OUTPUT_DIR")"
MERGED_OUT="$DATA_DIR/models/stt/${VERSION_NAME}-merged"
python "$SCRIPT_DIR/merge_lora.py" "$OUTPUT_DIR/adapter" --base-model "$BASE_MODEL" --out "$MERGED_OUT"

WIN_MERGED_OUT="$(echo "$MERGED_OUT" | sed -E 's#^/mnt/([a-z])/#\U\1:/#')"
echo ""
echo "Tayyor: $MERGED_OUT"
echo "Windows tomonida (C:\\ultron):"
echo "  uv run python scripts/export_stt_ct2.py --merged-model $WIN_MERGED_OUT --version 1"
