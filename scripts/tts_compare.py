"""TTS model comparison (docs/tz/05_tts.md §3.1).

    uv run python scripts/tts_compare.py --generate
    uv run python scripts/tts_compare.py --score

``--generate`` synthesizes every sentence in ``scripts/data/tts_compare_sentences.txt``
with every candidate model, measures RTF (real-time factor = synth_time /
audio_duration), and writes:
    data/reports/tts_compare/<model_slug>/NN.wav
    data/reports/tts_compare/rtf.tsv                 (machine-readable)
    data/reports/tts_compare/tts_compare_YYYYMMDD.md  (human-readable table)

``--score`` is the human listening step: it plays each not-yet-scored wav
file back (default output device) and asks for a 1-5 rating, appends it to
``data/reports/tts_compare/scores.tsv``, then -- once at least one score
exists -- picks the winner (highest average score among models that meet
the RTF <= 0.3 budget) and patches ``tts.model``/``tts.model_script`` into
config.yaml in place.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf
from loguru import logger

from ultron.config import ConfigError, load_config
from ultron.tts.mms_engine import MMSEngine

_RTF_BUDGET = 0.3
_DEFAULT_SENTENCES_PATH = Path("scripts/data/tts_compare_sentences.txt")

# Candidates surveyed on Hugging Face (docs/tz/05_tts.md §3.1): both are
# transformers VitsModel + AutoTokenizer, cyrillic script, is_uroman=False,
# so they plug into the same MMSEngine unchanged.
#   - facebook/mms-tts-uzb-script_cyrillic: Meta's base MMS Uzbek voice
#     (cc-by-nc-4.0 -- personal use only, which is what this project is).
#   - MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic: fine-tuned
#     from the base MMS checkpoint on Beehzod/UzTTS_data (cc-by-4.0, more
#     permissive, and specifically trained for natural Uzbek speech).
_DEFAULT_MODELS: dict[str, str] = {
    "facebook/mms-tts-uzb-script_cyrillic": "cyrillic",
    "MuzaffarSharofitdinov/mms-tts-uzbek-girl-voice_cyrillic": "cyrillic",
}


@dataclass
class SynthResult:
    index: int
    text: str
    duration_s: float
    synth_ms: float
    rtf: float


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _load_sentences(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Test gaplar fayli topilmadi: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_model(
    model_name: str, model_script: str, sentences: list[str], cache_dir: Path, out_dir: Path
) -> list[SynthResult]:
    engine = MMSEngine(model_name=model_name, model_script=model_script, cache_dir=cache_dir)
    engine.load()

    model_dir = out_dir / _slug(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, text in enumerate(sentences, start=1):
        start = time.monotonic()
        pcm = engine.synthesize(text)
        elapsed_s = time.monotonic() - start
        duration_s = len(pcm) / engine.sample_rate if pcm.size else 0.0
        rtf = elapsed_s / duration_s if duration_s > 0 else float("inf")
        sf.write(model_dir / f"{i:02d}.wav", pcm, engine.sample_rate)
        results.append(
            SynthResult(
                index=i, text=text, duration_s=duration_s, synth_ms=elapsed_s * 1000, rtf=rtf
            )
        )
        logger.info(
            "[{}] {:02d}: {:.2f}s audio, {:.0f}ms sintez, RTF={:.3f}",
            model_name,
            i,
            duration_s,
            elapsed_s * 1000,
            rtf,
        )
    return results


def _write_rtf_tsv(out_dir: Path, all_results: dict[str, list[SynthResult]]) -> None:
    lines = ["model\tindex\tduration_s\tsynth_ms\trtf"]
    for model_name, results in all_results.items():
        for r in results:
            lines.append(
                f"{model_name}\t{r.index}\t{r.duration_s:.4f}\t{r.synth_ms:.1f}\t{r.rtf:.4f}"
            )
    (out_dir / "rtf.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_rtf_tsv(out_dir: Path) -> dict[str, list[float]]:
    path = out_dir / "rtf.tsv"
    if not path.is_file():
        return {}
    per_model: dict[str, list[float]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        if not line.strip():
            continue
        model_name, _index, _duration, _synth, rtf = line.split("\t")
        per_model.setdefault(model_name, []).append(float(rtf))
    return per_model


def render_report(all_results: dict[str, list[SynthResult]]) -> str:
    lines = [f"# TTS model solishtirish hisoboti — {date.today().isoformat()}", ""]
    lines.append("| Model | O'rtacha RTF | Eng yuqori RTF | RTF <= 0.3 |")
    lines.append("|---|---|---|---|")
    for model_name, results in all_results.items():
        rtfs = [r.rtf for r in results]
        avg = statistics.mean(rtfs)
        budget_ok = "✅" if avg <= _RTF_BUDGET else "❌"
        lines.append(f"| {model_name} | {avg:.3f} | {max(rtfs):.3f} | {budget_ok} |")
    lines.append("")

    for model_name, results in all_results.items():
        lines.append(f"### {model_name}\n")
        lines.append("| # | Matn | Audio (s) | Sintez (ms) | RTF |")
        lines.append("|---|---|---|---|---|")
        for r in results:
            preview = r.text if len(r.text) <= 60 else r.text[:57] + "..."
            lines.append(
                f"| {r.index} | {preview} | {r.duration_s:.2f} | {r.synth_ms:.0f} | {r.rtf:.3f} |"
            )
        lines.append("")

    lines.append(
        "## Eshitib baholash\n\n"
        "Har bir model uchun audio namunalar yuqoridagi papkalarda: "
        "`<model>/NN.wav`. Eshitib baholash uchun:\n\n"
        "    uv run python scripts/tts_compare.py --score\n\n"
        "Bu skript har bir faylni navbat bilan chalib 1-5 ball so'raydi, "
        "`scores.tsv`ga yozadi va RTF byudjetidan o'tgan modellar orasida "
        "eng yuqori o'rtacha ball olgan modelni `config.yaml`ga yozadi.\n"
    )
    return "\n".join(lines) + "\n"


def cmd_generate(
    models: dict[str, str], sentences_path: Path, out_dir: Path, cache_dir: Path
) -> None:
    sentences = _load_sentences(sentences_path)
    logger.info("Test gaplar: {} ta ({})", len(sentences), sentences_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[SynthResult]] = {}
    for model_name, model_script in models.items():
        logger.info("=== {} ===", model_name)
        all_results[model_name] = _run_model(
            model_name, model_script, sentences, cache_dir, out_dir
        )

    _write_rtf_tsv(out_dir, all_results)
    report = render_report(all_results)
    report_path = out_dir / f"tts_compare_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Hisobot yozildi: {}", report_path)
    logger.info("\n{}", report)


# -- scoring -------------------------------------------------------------


def _load_scores(scores_path: Path) -> dict[tuple[str, str], int]:
    if not scores_path.is_file():
        return {}
    scores: dict[tuple[str, str], int] = {}
    for line in scores_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        model_name, index, score = line.split("\t")
        scores[(model_name, index)] = int(score)
    return scores


def _append_scores(scores_path: Path, new_rows: list[tuple[str, str, int]]) -> None:
    is_new = not scores_path.is_file()
    with scores_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("model\tindex\tscore\n")
        for model_name, index, score in new_rows:
            f.write(f"{model_name}\t{index}\t{score}\n")


def _pick_winner(
    scores: dict[tuple[str, str], int], rtf_by_model: dict[str, list[float]]
) -> tuple[str, float] | None:
    if not scores:
        return None
    avg_score_by_model: dict[str, float] = {}
    for model_name in {m for m, _ in scores}:
        model_scores = [s for (m, _), s in scores.items() if m == model_name]
        avg_score_by_model[model_name] = statistics.mean(model_scores)

    def within_budget(model_name: str) -> bool:
        rtfs = rtf_by_model.get(model_name)
        return rtfs is not None and statistics.mean(rtfs) <= _RTF_BUDGET

    in_budget = {m: s for m, s in avg_score_by_model.items() if within_budget(m)}
    pool = in_budget if in_budget else avg_score_by_model
    if not in_budget and avg_score_by_model:
        logger.warning(
            "RTF <= {} byudjetidan o'tgan model yo'q, baribir eng yuqori ball asosida tanlanadi.",
            _RTF_BUDGET,
        )
    winner = max(pool.items(), key=lambda kv: kv[1])
    return winner


def write_config_selection(config_path: Path, *, model: str, model_script: str) -> None:
    """Patch ``tts.model``/``tts.model_script`` in config.yaml in place, leaving
    every other line (including comments) untouched."""
    text = config_path.read_text(encoding="utf-8")
    text, n_model = re.subn(
        r'(?m)^(tts:\n(?:[ \t].*\n)*?[ \t]+model:\s*)"[^"]*"',
        lambda m: f'{m.group(1)}"{model}"',
        text,
        count=1,
    )
    text, n_script = re.subn(
        r'(?m)^(tts:\n(?:[ \t].*\n)*?[ \t]+model_script:\s*)"[^"]*"',
        lambda m: f'{m.group(1)}"{model_script}"',
        text,
        count=1,
    )
    if n_model == 0 or n_script == 0:
        logger.warning(
            "config.yaml'dagi tts.model/tts.model_script qatorlari avtomatik topilmadi, "
            "qo'lda yangilang: model={}, model_script={}",
            model,
            model_script,
        )
        return
    config_path.write_text(text, encoding="utf-8")
    logger.info(
        "config.yaml yangilandi: tts.model={!r}, tts.model_script={!r}", model, model_script
    )


def cmd_score(out_dir: Path, config_path: Path, write_config: bool) -> None:
    import sounddevice as sd

    if not out_dir.is_dir():
        raise SystemExit(
            f"{out_dir} topilmadi. Avval generatsiya qiling: "
            "uv run python scripts/tts_compare.py --generate"
        )
    model_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
    if not model_dirs:
        raise SystemExit(f"{out_dir} ichida model papkalari topilmadi.")

    # rtf.tsv is always written with real HF model ids (e.g.
    # "facebook/mms-tts-uzb-script_cyrillic"), while model_dir.name is the
    # filesystem-safe slug (_slug() replaces "/" with "__"). Scores must be
    # keyed by the same real id so _pick_winner's RTF-budget lookup and
    # write_config_selection's model= actually match -- keying by the raw
    # slug here previously wrote the slug itself into config.yaml as an
    # (invalid) model id and silently broke the RTF-budget filter.
    rtf_by_model = _load_rtf_tsv(out_dir)
    slug_to_model = {_slug(name): name for name in rtf_by_model} | {
        _slug(name): name for name in _DEFAULT_MODELS
    }

    def _real_name(model_dir: Path) -> str:
        return slug_to_model.get(model_dir.name, model_dir.name.replace("__", "/", 1))

    scores_path = out_dir / "scores.tsv"
    existing = _load_scores(scores_path)
    new_rows: list[tuple[str, str, int]] = []

    for model_dir in model_dirs:
        model_name = _real_name(model_dir)
        for wav_path in sorted(model_dir.glob("*.wav")):
            key = (model_name, wav_path.stem)
            if key in existing:
                continue
            pcm, sr = sf.read(wav_path, dtype="float32")
            print(f"\n[{model_name}] {wav_path.name}")
            while True:
                sd.play(pcm, sr)
                sd.wait()
                raw = input("Ball (1-5, qayta eshitish uchun Enter): ").strip()
                if raw == "":
                    continue
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    new_rows.append((model_name, wav_path.stem, int(raw)))
                    break
                print("1 dan 5 gacha son kiriting.")

    if new_rows:
        _append_scores(scores_path, new_rows)
        logger.info("{} ta yangi ball yozildi: {}", len(new_rows), scores_path)

    all_scores = _load_scores(scores_path)
    if not all_scores:
        logger.info("Hali hech qanday ball yo'q.")
        return

    winner = _pick_winner(all_scores, rtf_by_model)
    if winner is None:
        logger.info("G'olib aniqlanmadi.")
        return
    model_name, avg_score = winner
    logger.info("G'olib: {} (o'rtacha ball={:.2f})", model_name, avg_score)
    if write_config:
        write_config_selection(
            config_path, model=model_name, model_script=_DEFAULT_MODELS.get(model_name, "cyrillic")
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron TTS model solishtirish vositasi")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate", action="store_true", help="Har bir model bilan namuna yaratish"
    )
    group.add_argument("--score", action="store_true", help="Namunalarni eshitib baholash")
    parser.add_argument(
        "--models", nargs="+", default=None, help="HF model id'lari (standart: 2 nomzod)"
    )
    parser.add_argument("--sentences", type=Path, default=_DEFAULT_SENTENCES_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--no-write-config", action="store_true", help="config.yaml'ni o'zgartirmaslik"
    )
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

    out_dir = args.out_dir or (Path(config.app.data_dir) / "reports" / "tts_compare")
    cache_dir = Path(config.app.data_dir) / "models" / "tts"

    if args.generate:
        if args.models:
            models = {name: _DEFAULT_MODELS.get(name, "cyrillic") for name in args.models}
        else:
            models = _DEFAULT_MODELS
        cmd_generate(models, args.sentences, out_dir, cache_dir)
    elif args.score:
        cmd_score(out_dir, Path("config/config.yaml"), write_config=not args.no_write_config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
