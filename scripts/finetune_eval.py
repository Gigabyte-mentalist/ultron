"""LoRA fine-tune acceptance check (docs/tz/11_uzbek_stt_finetune.md §3.5).

    uv run python scripts/finetune_eval.py --new-model data/models/stt/ultron-uz-v1-ct2

Compares the currently configured model (``config.yaml``'s ``stt.model``)
against a fine-tuned candidate across three test sets, reusing
``scripts/benchmark_stt.py``'s ``run_one_config`` (same WER/CER/latency/VRAM
measurement TZ-03's model selection already used) rather than duplicating it:

  1. TZ-03's general 80-sentence set (``data/recordings/testset/``).
  2. The personal test split (``data/datasets/personal/test.jsonl``, built by
     ``scripts/build_personal_dataset.py``).
  3. FLEURS uz test (``data/datasets/fleurs_uz_test/``, exported by
     ``training/stt/prepare_open_datasets.py``) -- checks the fine-tune
     didn't wreck general Uzbek quality while specializing on one voice.

Acceptance rule (§3.5/§4): the candidate must beat the current model's WER on
(1) *and* (2), must stay within TZ-03's latency/VRAM budget (p95<=800ms,
VRAM<=2.2GB -- ``ConfigResult.meets_budget``) on (1), and must not regress by
more than 3 WER points on (3). If it passes, ``config.yaml``'s ``stt.model``
is updated (beam_size untouched -- unlike ``benchmark_stt.py``'s initial
model pick, this step doesn't
re-benchmark beam size) and a ``MODEL_CARD.md`` is written next to the new
model directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_stt import ConfigResult, run_one_config
from loguru import logger

from ultron.config import ConfigError, RootConfig, load_config

_FLEURS_REGRESSION_BUDGET = 3.0  # WER points


@dataclass
class Comparison:
    name: str
    old: ConfigResult | None
    new: ConfigResult | None

    @property
    def improved(self) -> bool | None:
        if self.old is None or self.new is None:
            return None
        return self.new.wer < self.old.wer

    @property
    def regression(self) -> float | None:
        if self.old is None or self.new is None:
            return None
        return self.new.wer - self.old.wer


def _load_jsonl_testset(path: Path, data_dir: Path) -> list[tuple[Path, str]]:
    """Reads the ``{"audio_path": <relative to data_dir>, "text": ...}`` rows
    ``build_personal_dataset.py``/``prepare_open_datasets.py`` both write."""
    if not path.is_file():
        return []
    rows: list[tuple[Path, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append((data_dir / record["audio_path"], record["text"]))
    return rows


def _load_general_testset(testset_dir: Path) -> list[tuple[Path, str]]:
    tsv_path = testset_dir / "transcripts.tsv"
    if not tsv_path.is_file():
        return []
    rows: list[tuple[Path, str]] = []
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        filename, reference = line.split("\t", 1)
        rows.append((testset_dir / filename, reference.strip()))
    return rows


def run_comparison(
    config: RootConfig, *, old_model: str, new_model: str, beam_size: int, data_dir: Path
) -> list[Comparison]:
    testsets = {
        "general (TZ-03, 80 gap)": _load_general_testset(data_dir / "recordings" / "testset"),
        "shaxsiy test": _load_jsonl_testset(
            data_dir / "datasets" / "personal" / "test.jsonl", data_dir
        ),
        "FLEURS uz test": _load_jsonl_testset(
            data_dir / "datasets" / "fleurs_uz_test" / "test.jsonl", data_dir
        ),
    }

    comparisons: list[Comparison] = []
    for name, testset in testsets.items():
        if not testset:
            logger.warning("{}: test to'plami topilmadi/bo'sh, o'tkazib yuborildi", name)
            comparisons.append(Comparison(name, None, None))
            continue
        logger.info("=== {} ({} ta gap) ===", name, len(testset))
        old = run_one_config(config, old_model, beam_size, testset)
        new = run_one_config(config, new_model, beam_size, testset)
        comparisons.append(Comparison(name, old, new))
    return comparisons


def decide(comparisons: list[Comparison]) -> tuple[bool, str]:
    by_name = {c.name: c for c in comparisons}
    general = by_name.get("general (TZ-03, 80 gap)")
    personal = by_name.get("shaxsiy test")
    fleurs = by_name.get("FLEURS uz test")

    if general is None or general.improved is not True:
        return False, "general (TZ-03) to'plamida yaxshilanish yo'q (yoki to'plam topilmadi)"
    if personal is None or personal.improved is not True:
        return False, "shaxsiy test to'plamida yaxshilanish yo'q (yoki to'plam topilmadi)"
    if not general.new.meets_budget:
        return False, (
            f"yangi model TZ-03 latency/VRAM byudjetiga mos kelmaydi "
            f"(p95={general.new.p95_latency_ms:.0f}ms, VRAM={general.new.vram_mb})"
        )
    if fleurs is not None and fleurs.regression is not None:
        if fleurs.regression > _FLEURS_REGRESSION_BUDGET:
            return False, f"FLEURS'da {fleurs.regression:.1f} punkt yomonlashish (limit 3.0)"
    else:
        logger.warning("FLEURS test to'plami topilmadi -- regressiya tekshiruvi o'tkazilmadi")
    return True, "barcha shartlar bajarildi"


def render_report(comparisons: list[Comparison], *, old_model: str, new_model: str) -> str:
    lines = [
        f"# STT fine-tune baholash -- {date.today().isoformat()}",
        "",
        f"Eski model: `{old_model}`",
        f"Yangi model: `{new_model}`",
        "",
        "| Test to'plami | Eski WER % | Yangi WER % | O'zgarish |",
        "|---|---|---|---|",
    ]
    for c in comparisons:
        if c.old is None or c.new is None:
            lines.append(f"| {c.name} | - | - | topilmadi |")
            continue
        delta = c.regression
        arrow = "yaxshilandi" if delta is not None and delta < 0 else "yomonlashdi"
        lines.append(f"| {c.name} | {c.old.wer:.1f} | {c.new.wer:.1f} | {delta:+.1f} ({arrow}) |")

    accepted, reason = decide(comparisons)
    lines.append("")
    lines.append(f"**Qaror:** {'QABUL QILINDI' if accepted else 'RAD ETILDI'} -- {reason}")
    return "\n".join(lines) + "\n"


def write_model_card(
    model_dir: Path,
    *,
    old_model: str,
    new_model: str,
    comparisons: list[Comparison],
    accepted: bool,
    training_meta: dict | None,
) -> None:
    lines = [f"# {model_dir.name}", "", f"Yaratilgan: {date.today().isoformat()}", ""]
    if training_meta:
        lines.append("## O'qitish parametrlari")
        for key, value in training_meta.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("## Baholash")
    lines.append(f"Eski model: `{old_model}`")
    for c in comparisons:
        if c.old is None or c.new is None:
            continue
        lines.append(f"- {c.name}: WER {c.old.wer:.1f}% -> {c.new.wer:.1f}%")
    lines.append("")
    accepted_text = "ha" if accepted else "yo'q"
    lines.append(f"**Qabul qilindi:** {accepted_text}")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "MODEL_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stt_model(config_path: Path, model: str) -> None:
    """Patches only ``stt.model`` in config.yaml -- unlike
    ``benchmark_stt.write_config_selection``, beam_size isn't re-decided here."""
    import re

    text = config_path.read_text(encoding="utf-8")
    text, n = re.subn(
        r'(?m)^(stt:\n(?:[ \t].*\n)*?[ \t]+model:\s*)"[^"]*"',
        lambda m: f'{m.group(1)}"{model}"',
        text,
        count=1,
    )
    if n == 0:
        logger.warning("config.yaml'dagi stt.model qatori topilmadi, qo'lda yangilang: {}", model)
        return
    config_path.write_text(text, encoding="utf-8")
    logger.info("config.yaml yangilandi: stt.model={!r}", model)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron STT fine-tune qabul tekshiruvi")
    parser.add_argument("--new-model", required=True, help="Yangi CT2 model papkasi")
    parser.add_argument("--old-model", default=None, help="Standart: config.yaml'dagi stt.model")
    parser.add_argument(
        "--beam", type=int, default=None, help="Standart: config.yaml'dagi beam_size"
    )
    parser.add_argument(
        "--training-meta", type=Path, default=None, help="train_lora.py'ning JSON metadata fayli"
    )
    parser.add_argument("--no-write-config", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    args = parse_args(argv)
    try:
        config = load_config()
    except ConfigError as exc:
        logger.info("Config yuklab bo'lmadi:\n{}", exc)
        return 1

    old_model = args.old_model or config.stt.model
    beam_size = args.beam or config.stt.beam_size
    data_dir = Path(config.app.data_dir)

    comparisons = run_comparison(
        config,
        old_model=old_model,
        new_model=args.new_model,
        beam_size=beam_size,
        data_dir=data_dir,
    )

    report = render_report(comparisons, old_model=old_model, new_model=args.new_model)
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"stt_finetune_eval_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Hisobot yozildi: {}", report_path)
    logger.info("\n{}", report)

    accepted, _reason = decide(comparisons)
    training_meta = None
    if args.training_meta and args.training_meta.is_file():
        training_meta = json.loads(args.training_meta.read_text(encoding="utf-8"))
    write_model_card(
        Path(args.new_model),
        old_model=old_model,
        new_model=args.new_model,
        comparisons=comparisons,
        accepted=accepted,
        training_meta=training_meta,
    )

    if accepted and not args.no_write_config:
        _write_stt_model(Path("config/config.yaml"), args.new_model)
    elif not accepted:
        logger.info("Yangi model RAD ETILDI, config.yaml o'zgartirilmadi.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
