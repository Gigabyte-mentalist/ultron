"""STT model/beam-size benchmark (docs/tz/03_stt.md §3.5).

    uv run python scripts/benchmark_stt.py --models large-v3-turbo large-v3 medium small --beam 1 3
    uv run python scripts/benchmark_stt.py --models data/models/stt/kotib-uzbek-stt-v1-ct2 --beam 1

Reads the recorded test set from ``data/recordings/testset/`` (``transcripts.tsv``
+ ``NNNN.wav``, produced by ``record_samples.py testset``), transcribes every
sample with every (model, beam) combination, and writes a comparison table to
``data/reports/stt_benchmark_YYYYMMDD.md``. The config-selection rule (lowest
WER among configs meeting the latency/VRAM budget) is applied at the end and
the winner is written back into ``config/config.yaml``.
"""

from __future__ import annotations

import argparse
import gc
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jiwer
import soundfile as sf
from loguru import logger

from ultron.config import ConfigError, RootConfig, load_config
from ultron.stt.whisper_engine import WhisperEngine
from ultron.text.normalize import normalize_transcript

_SAMPLE_RATE = 16000
_LATENCY_BUDGET_MS = 800
_VRAM_BUDGET_MB = 2200
_ALTRON_MARKER = "altron"
_DEFAULT_MODELS = ("large-v3-turbo", "large-v3", "medium", "small")
_DEFAULT_BEAMS = (1, 3)


@dataclass
class SampleResult:
    reference: str
    hypothesis: str
    latency_ms: int


@dataclass
class ConfigResult:
    model: str
    beam_size: int
    wer: float
    cer: float
    avg_latency_ms: float
    p95_latency_ms: float
    vram_mb: float | None
    altron_recall: float | None
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def meets_budget(self) -> bool:
        vram_ok = self.vram_mb is None or self.vram_mb <= _VRAM_BUDGET_MB
        return self.p95_latency_ms <= _LATENCY_BUDGET_MS and vram_ok


def _load_testset(testset_dir: Path) -> list[tuple[Path, str]]:
    tsv_path = testset_dir / "transcripts.tsv"
    if not tsv_path.is_file():
        raise SystemExit(
            f"Test to'plami topilmadi: {tsv_path}\n"
            "Avval yozib oling: uv run python scripts/record_samples.py testset "
            "--file scripts/data/prompts_uz.txt"
        )
    rows: list[tuple[Path, str]] = []
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        filename, reference = line.split("\t", 1)
        rows.append((testset_dir / filename, reference.strip()))
    if not rows:
        raise SystemExit(f"Test to'plami bo'sh: {tsv_path}")
    return rows


def _current_vram_mb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def _release(engine: WhisperEngine) -> None:
    del engine
    gc.collect()
    time.sleep(1.0)  # let the CUDA driver report freed memory before the next nvidia-smi read


def run_one_config(
    config: RootConfig, model: str, beam_size: int, testset: list[tuple[Path, str]]
) -> ConfigResult:
    logger.info("=== {} (beam={}) ===", model, beam_size)
    vram_before = _current_vram_mb()

    engine = WhisperEngine(
        model=model,
        device=config.stt.device,
        compute_type=config.stt.compute_type,
        language=config.stt.language,
        beam_size=beam_size,
        no_speech_threshold=config.stt.no_speech_threshold,
        log_prob_threshold=config.stt.log_prob_threshold,
        initial_prompt=config.stt.initial_prompt,
        download_root=Path(config.app.data_dir) / "models" / "stt",
    )
    engine.load()
    vram_after = _current_vram_mb()
    vram_mb = (
        max(0.0, vram_after - vram_before)
        if vram_before is not None and vram_after is not None
        else None
    )

    samples: list[SampleResult] = []
    for wav_path, reference in testset:
        pcm, sr = sf.read(wav_path, dtype="int16")
        if sr != _SAMPLE_RATE:
            raise SystemExit(f"Kutilmagan chastota {sr} Hz: {wav_path}")
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        transcript = engine.transcribe(pcm)
        samples.append(
            SampleResult(
                reference=reference, hypothesis=transcript.text, latency_ms=transcript.latency_ms
            )
        )

    _release(engine)

    refs_norm = [normalize_transcript(s.reference).lower() for s in samples]
    hyps_norm = [normalize_transcript(s.hypothesis).lower() for s in samples]
    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm, strict=True) if r]
    refs_final = [r for r, _ in pairs]
    hyps_final = [h for _, h in pairs]

    wer = jiwer.wer(refs_final, hyps_final) * 100
    cer = jiwer.process_characters(refs_final, hyps_final).cer * 100

    latencies = [float(s.latency_ms) for s in samples]
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = _p95(latencies)

    altron_rows = [(r, h) for r, h in zip(refs_norm, hyps_norm, strict=True) if _ALTRON_MARKER in r]
    altron_recall = (
        sum(1 for r, h in altron_rows if _ALTRON_MARKER in h) / len(altron_rows)
        if altron_rows
        else None
    )

    return ConfigResult(
        model=model,
        beam_size=beam_size,
        wer=wer,
        cer=cer,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        vram_mb=vram_mb,
        altron_recall=altron_recall,
        samples=samples,
    )


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.1f}{suffix}"


def render_report(results: list[ConfigResult], testset_size: int) -> str:
    lines = [
        f"# STT benchmark — {date.today().isoformat()}",
        "",
        f"Test to'plami: {testset_size} ta gap (`scripts/data/prompts_uz.txt`).",
        f"Qabul mezoni: WER <= 25%, p95 latency <= {_LATENCY_BUDGET_MS} ms, "
        f"VRAM <= {_VRAM_BUDGET_MB} MB.",
        "",
        "| Model | Beam | WER % | CER % | O'rtacha latency (ms) | p95 latency (ms) "
        "| VRAM (MB) | Altron tanilishi | Byudjetga mos |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    row_template = "| {model} | {beam} | {wer} | {cer} | {avg} | {p95} | {vram} | {altron} | {ok} |"
    for r in sorted(results, key=lambda x: x.wer):
        lines.append(
            row_template.format(
                model=r.model,
                beam=r.beam_size,
                wer=_fmt(r.wer),
                cer=_fmt(r.cer),
                avg=_fmt(r.avg_latency_ms),
                p95=_fmt(r.p95_latency_ms),
                vram=_fmt(r.vram_mb),
                altron=_fmt(r.altron_recall * 100 if r.altron_recall is not None else None, "%"),
                ok="ha" if r.meets_budget else "yo'q",
            )
        )

    winner = pick_winner(results)
    lines.append("")
    if winner is not None:
        lines.append(
            f"**G'olib:** `{winner.model}` (beam={winner.beam_size}), "
            f"WER={winner.wer:.1f}%, p95={winner.p95_latency_ms:.0f} ms, "
            f"VRAM={_fmt(winner.vram_mb)} MB."
        )
        if winner.wer > 25.0:
            lines.append(
                "\n**Diqqat:** g'olib ham WER <= 25% mezonini bajarmadi. "
                "docs/tz/11_uzbek_stt_finetune.md (LoRA fine-tune) ni oldinroq "
                "bajarish tavsiya etiladi."
            )
    else:
        lines.append(
            "**G'olib topilmadi:** hech bir konfiguratsiya latency/VRAM byudjetiga mos kelmadi. "
            "Eng past WER'li konfiguratsiya baribir tanlanadi, quyida ko'rsatilgan."
        )
    return "\n".join(lines) + "\n"


def pick_winner(results: list[ConfigResult]) -> ConfigResult | None:
    """Lowest WER among configs meeting the latency/VRAM budget; falls back to
    the overall lowest WER (and flags it in the report) if none qualify."""
    if not results:
        return None
    in_budget = [r for r in results if r.meets_budget]
    pool = in_budget if in_budget else results
    return min(pool, key=lambda r: r.wer)


def write_config_selection(config_path: Path, *, model: str, beam_size: int) -> None:
    """Patch ``stt.model``/``stt.beam_size`` in config.yaml in place, leaving
    every other line (including comments) untouched."""
    text = config_path.read_text(encoding="utf-8")
    text, n_model = re.subn(
        r'(?m)^(stt:\n(?:[ \t].*\n)*?[ \t]+model:\s*)"[^"]*"',
        lambda m: f'{m.group(1)}"{model}"',
        text,
        count=1,
    )
    text, n_beam = re.subn(
        r"(?m)^(stt:\n(?:[ \t].*\n)*?[ \t]+beam_size:\s*)\d+",
        lambda m: f"{m.group(1)}{beam_size}",
        text,
        count=1,
    )
    if n_model == 0 or n_beam == 0:
        logger.warning(
            "config.yaml'dagi stt.model/stt.beam_size qatorlari avtomatik topilmadi, "
            "qo'lda yangilang: model={}, beam_size={}",
            model,
            beam_size,
        )
        return
    config_path.write_text(text, encoding="utf-8")
    logger.info("config.yaml yangilandi: stt.model={!r}, stt.beam_size={}", model, beam_size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron STT benchmark")
    parser.add_argument("--models", nargs="+", default=list(_DEFAULT_MODELS))
    parser.add_argument("--beam", nargs="+", type=int, default=list(_DEFAULT_BEAMS))
    parser.add_argument("--testset-dir", type=Path, default=None)
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

    testset_dir = args.testset_dir or (Path(config.app.data_dir) / "recordings" / "testset")
    testset = _load_testset(testset_dir)
    logger.info("Test to'plami: {} ta gap ({})", len(testset), testset_dir)

    results: list[ConfigResult] = []
    for model in args.models:
        for beam in args.beam:
            try:
                results.append(run_one_config(config, model, beam, testset))
            except Exception as exc:  # noqa: BLE001 - one bad config must not abort the whole run
                logger.error("{} (beam={}) bajarilmadi: {}", model, beam, exc)

    if not results:
        logger.error("Hech qanday konfiguratsiya muvaffaqiyatli bajarilmadi.")
        return 1

    report = render_report(results, len(testset))
    reports_dir = Path(config.app.data_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"stt_benchmark_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Hisobot yozildi: {}", report_path)
    logger.info("\n{}", report)

    if not args.no_write_config:
        winner = pick_winner(results)
        if winner is not None:
            write_config_selection(
                Path("config/config.yaml"), model=winner.model, beam_size=winner.beam_size
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
