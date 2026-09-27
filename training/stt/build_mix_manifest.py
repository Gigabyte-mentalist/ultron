"""Merge personal + open datasets for LoRA training (docs/tz/11 §3.3/§3.4).

    python build_mix_manifest.py configs/lora_v1.yaml

Reads ``scripts/build_personal_dataset.py``'s ``{train,dev}.jsonl`` (personal,
Windows-visible under ``paths.data_dir``) and ``prepare_open_datasets.py``'s
``{train,dev}.jsonl`` (open, relative to their own directory), then writes
``mix_train.jsonl``/``mix_dev.jsonl`` -- personal rows repeated (oversampled)
until they make up ``mix.personal_ratio`` of the mix (§3.3: "shaxsiy 50% /
ochiq 50%, aralashma nisbati ... tajriba orqali sozlanadi" -- edit the yaml
and re-run to try a different ratio). Every row in the output carries an
absolute ``audio_path`` so ``train_lora.py`` doesn't need to know which
source directory it came from.

No third-party imports -- runs with the training venv's plain Python (or even
outside it), no need to wait on ``setup_env.sh`` to try this part out.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import yaml


def _load_jsonl(path: Path, *, root: Path) -> list[dict]:
    if not path.is_file():
        print(f"Ogohlantirish: topilmadi, bo'sh deb hisoblanadi: {path}", file=sys.stderr)
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append({"audio_path": str(root / record["audio_path"]), "text": record["text"]})
    return rows


def oversample(rows: list[dict], target_count: int, *, seed: int) -> list[dict]:
    """Repeats ``rows`` (in a fixed, shuffled order) until it reaches
    ``target_count``; returns ``rows`` unchanged if it's already >= target."""
    if not rows or len(rows) >= target_count:
        return rows
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    cycle = itertools.cycle(shuffled)
    return [next(cycle) for _ in range(target_count)]


def build_mix(
    *, personal: list[dict], open_data: list[dict], personal_ratio: float, seed: int
) -> list[dict]:
    if not open_data:
        return personal
    if not personal:
        return open_data
    # personal / (personal + open) = personal_ratio  =>  personal = ratio/(1-ratio) * open
    target_personal = round(personal_ratio / (1 - personal_ratio) * len(open_data))
    personal_mixed = oversample(personal, target_personal, seed=seed)
    mix = personal_mixed + open_data
    random.Random(seed).shuffle(mix)
    return mix


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shaxsiy + ochiq datasetlarni aralashtirish")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    data_dir = Path(paths["data_dir"])
    config_dir = args.config.parent

    personal_train = _load_jsonl(Path(paths["personal_train"]), root=data_dir)
    personal_dev = _load_jsonl(Path(paths["personal_dev"]), root=data_dir)
    open_train_path = config_dir / paths["open_train"]
    open_dev_path = config_dir / paths["open_dev"]
    open_train = _load_jsonl(open_train_path, root=open_train_path.parent)
    open_dev = _load_jsonl(open_dev_path, root=open_dev_path.parent)

    seed = config.get("training", {}).get("seed", 42)
    ratio = config["mix"]["personal_ratio"]

    mix_train = build_mix(
        personal=personal_train, open_data=open_train, personal_ratio=ratio, seed=seed
    )
    mix_dev = build_mix(personal=personal_dev, open_data=open_dev, personal_ratio=ratio, seed=seed)

    _write_jsonl(config_dir / paths["mix_train_out"], mix_train)
    _write_jsonl(config_dir / paths["mix_dev_out"], mix_dev)

    print(
        f"train: shaxsiy(asl)={len(personal_train)} ochiq={len(open_train)} "
        f"-> aralashma={len(mix_train)}"
    )
    print(
        f"dev:   shaxsiy(asl)={len(personal_dev)} ochiq={len(open_dev)} -> aralashma={len(mix_dev)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
