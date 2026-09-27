"""Merge a trained LoRA adapter into the base Whisper weights (docs/tz/11 §3.5).

    python merge_lora.py checkpoints/ultron-uz-v1/adapter \\
        --base-model islomov/rubaistt_v2_medium \\
        --out /mnt/d/ultron_data/models/stt/ultron-uz-v1-merged

Produces a plain (non-PEFT) HF-format checkpoint -- ``PeftModel.
merge_and_unload()`` folds the LoRA delta into the base weights, so the
result loads with plain ``WhisperForConditionalGeneration.from_pretrained()``
and, from there, ``scripts/export_stt_ct2.py`` (Windows main venv) can
convert it to CTranslate2 exactly like ``download_models.py`` already does
for the (non-fine-tuned) community checkpoints.

Writing straight to a path under ``/mnt/d/...`` (or wherever ``config.yaml``'s
``app.data_dir`` is mounted from WSL) means no separate copy step -- same
pattern training/wake/README.md §5 uses for the wake model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def merge(adapter_dir: Path, base_model: str, out_dir: Path) -> None:
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"Asosiy model yuklanmoqda: {base_model}")
    base = WhisperForConditionalGeneration.from_pretrained(base_model)
    print(f"LoRA adapter qo'shilmoqda: {adapter_dir}")
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    WhisperProcessor.from_pretrained(str(adapter_dir)).save_pretrained(out_dir)
    print(f"Birlashtirilgan model saqlandi: {out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA adapterni asosiy modelga birlashtirish")
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    merge(args.adapter_dir, args.base_model, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
