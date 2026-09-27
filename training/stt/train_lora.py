"""Whisper LoRA fine-tune (docs/tz/11_uzbek_stt_finetune.md §3.4).

    python train_lora.py configs/lora_v1.yaml

Standard PEFT-LoRA-on-Whisper recipe (the same shape as HF's own
"Fine-Tune Whisper with 🤗 Transformers" + PEFT community notebooks, which is
almost certainly close to how this project's two candidate Uzbek fine-tunes
themselves were trained): freeze the base Whisper weights, attach LoRA
adapters to the attention projections, and train with ``Seq2SeqTrainer``,
evaluating WER on the dev mix every ``eval_steps``.

**Untested** -- this needs a real GPU + the packages ``setup_env.sh``
installs, neither of which exist in the environment that wrote this script
(docs/tz/10_wake_word_training.md's WSL training scripts carry the exact same
disclosure, for the same reason). Read training/stt/README.md §4's "known
risk points" before the first run -- PEFT/transformers/bitsandbytes version
drift is the most likely source of a first-run error, not a logic bug here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads ``input_features`` (via the feature extractor) and ``labels``
    (via the tokenizer) independently -- they have unrelated lengths. Label
    padding is replaced with -100 so the loss ignores it, and a leading BOS
    token is stripped if the tokenizer already prepended one during ``.map()``
    (Whisper's ``Seq2SeqTrainer`` re-adds it at generation time)."""

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_mix_dataset(path: Path):
    from datasets import Audio, Dataset

    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    ds = Dataset.from_list(rows)
    return ds.cast_column("audio_path", Audio(sampling_rate=16000))


def build_prepare_fn(processor):
    def prepare(example: dict) -> dict:
        audio = example["audio_path"]
        example["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    return prepare


def build_compute_metrics(processor):
    import evaluate

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


def train(config: dict, config_dir: Path) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    paths = config["paths"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]

    processor = WhisperProcessor.from_pretrained(
        config["base_model"], language=config["language"], task=config["task"]
    )

    train_ds = load_mix_dataset(config_dir / paths["mix_train_out"])
    dev_ds = load_mix_dataset(config_dir / paths["mix_dev_out"])
    prepare_fn = build_prepare_fn(processor)
    train_ds = train_ds.map(prepare_fn, remove_columns=train_ds.column_names, num_proc=1)
    dev_ds = dev_ds.map(prepare_fn, remove_columns=dev_ds.column_names, num_proc=1)
    print(f"train={len(train_ds)} dev={len(dev_ds)}")

    load_in_8bit = train_cfg.get("load_in_8bit", False)
    model = WhisperForConditionalGeneration.from_pretrained(
        config["base_model"],
        load_in_8bit=load_in_8bit,
        device_map="auto" if load_in_8bit else None,
        torch_dtype=torch.float16 if train_cfg.get("fp16", True) else torch.float32,
    )
    if not load_in_8bit:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.generation_config.language = config["language"]
    model.generation_config.task = config["task"]
    model.generation_config.forced_decoder_ids = None

    if load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    elif train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir = config_dir / paths["output_dir"]
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_ratio=train_cfg["warmup_ratio"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=train_cfg.get("fp16", True) and not load_in_8bit,
        optim=train_cfg["optim"],
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=train_cfg.get("seed", 42),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    meta = {
        "base_model": config["base_model"],
        "lora": lora_cfg,
        "training": train_cfg,
        "train_examples": len(train_ds),
        "dev_examples": len(dev_ds),
        "final_train_loss": train_result.training_loss,
        "final_dev_wer": eval_result.get("eval_wer"),
    }
    (adapter_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Adapter saqlandi: {adapter_dir}")
    print(f"Dev WER: {eval_result.get('eval_wer')}")
    return adapter_dir


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("Usage: python train_lora.py <config.yaml>", file=sys.stderr)
        return 1
    config_path = Path(argv[0])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train(config, config_path.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
