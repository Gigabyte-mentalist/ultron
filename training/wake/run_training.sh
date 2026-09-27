#!/usr/bin/env bash
# docs/tz/10_wake_word_training.md §3.3 -- runs the 3 documented train.py
# stages (see README.md §4) for one config.
#
#   bash run_training.sh configs/altron_v1.yml
#   bash run_training.sh configs/altron_v2.yml
#
# Config paths inside the YAML are relative to ~/oww_train/ (where
# setup_env.sh downloaded everything), so this copies the config there
# and runs train.py from that directory rather than from here.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: bash run_training.sh configs/altron_v1.yml" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_SRC="$SCRIPT_DIR/$1"
if [ ! -f "$CONFIG_SRC" ]; then
    echo "Config topilmadi: $CONFIG_SRC" >&2
    exit 1
fi

ROOT="$HOME/oww_train"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
cd "$ROOT"
cp "$CONFIG_SRC" ./active_config.yml

TRAIN_PY="./openwakeword/openwakeword/train.py"
echo "== generate_clips (idempotent -- safe to re-run if interrupted) =="
python "$TRAIN_PY" --training_config ./active_config.yml --generate_clips
echo "== augment_clips =="
python "$TRAIN_PY" --training_config ./active_config.yml --augment_clips
echo "== train_model =="
python "$TRAIN_PY" --training_config ./active_config.yml --train_model

python - <<'PY'
import yaml
c = yaml.safe_load(open("active_config.yml"))
print(f"Tayyor: {c['output_dir']}/{c['model_name']}.onnx")
print(f"Windows'ga ko'chirish: cp {c['output_dir']}/{c['model_name']}.onnx /mnt/c/ultron/data/models/wake/altron.onnx")
PY
