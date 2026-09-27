"""Wake-word benchmark, all three docs/tz/04_wake_word.md modes
(docs/tz/04_wake_word.md §3.6, docs/tz/10_wake_word_training.md §3.5).

    uv run python scripts/wake_live.py                      # jonli sinov
    uv run python scripts/benchmark_wake.py --mode stt_kws \\
        --positives data/recordings/wake --negatives data/recordings/wake_neg
    uv run python scripts/benchmark_wake.py --mode oww       # data/models/wake/altron.onnx kerak
    uv run python scripts/benchmark_wake.py --mode hybrid    # yuqoridagi ikkalasi ham kerak
    uv run python scripts/benchmark_wake.py --mode all       # 10-TZ §3.5'dagi jadval, uchalasi

Positives: one VAD-isolated wake-word utterance per wav file (produced by
``uv run python scripts/record_samples.py wake --count 100``, which writes
into ``data/recordings/wake/`` -- the 04-TZ's own example names this
directory ``wake_pos``; the ``--positives`` default here matches what that
02-TZ tool actually writes, overridable if you rename/relocate it).

Negatives: one or more long wav files with no wake word (ordinary speech,
TV, music). ``--mode stt_kws``/``oww`` split each into VAD/frame-level
events independently; ``--mode hybrid`` replays each through a single real
``HybridDetector`` (dedup included) so its false-accept count reflects the
actual combined pipeline, not a naive union of the other two.

``--mode stt_kws`` (unchanged since 04-TZ): the *raw* fuzzy score is computed
once per sample so the ``--thresholds`` sweep is free -- no repeated STT
calls. ``--mode oww`` mirrors this: one openWakeWord inference pass per file
records the full per-frame score curve, so the ``--oww-thresholds`` sweep is
also free (no repeated model calls) -- a threshold only changes how that
curve is *read*: recall (any frame >= t), false accepts (rising-edge count),
latency (index of first frame >= t). ``--mode hybrid`` cannot be swept this
cheaply (its false-accept count depends on the *live* dedup between two
detectors), so it runs once at whatever ``wake.fuzzy_threshold``/
``wake.oww_threshold`` are currently set in config.yaml -- run stt_kws/oww
first so those are the sweep winners.

"Latency p95 (so'z oxiridan)" for oww/hybrid is approximate: positives are
already VAD-endpointed recordings (docs/tz/02_audio.md), so each file's own
tail already includes ``vad.end_silence_ms`` of trailing silence; the word's
true end is estimated as ``file_duration - end_silence_ms`` and latency is
measured from there (per docs/tz/04_wake_word.md §1, oww can legitimately
trigger *before* that point -- a negative latency is the expected win over
stt_kws, not a bug). "GPU band vaqti" sums real STT-engine ``.transcribe()``
wall-clock time actually spent during that mode's pass (0 for pure oww,
since its threshold sweep never calls STT); "CPU %" is this process's
``psutil`` CPU usage during that pass (unnormalized: 100% = one core).

The report is written to ``data/reports/wake_benchmark_YYYYMMDD.md``. With
``--mode stt_kws``/``oww`` (not ``--no-write-config``), the winning threshold
is patched into ``config/config.yaml``. With ``--mode all``, both winners are
written and, per docs/tz/10_wake_word_training.md §3.5's acceptance rule,
``wake.mode`` itself is set to ``hybrid`` only if hybrid's recall is within
1 point of stt_kws's and its p95 latency is <= 250 ms -- otherwise it's left
at ``stt_kws`` and the report explains why.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import psutil
import soundfile as sf
from loguru import logger

from ultron.audio.ring_buffer import RingBuffer
from ultron.audio.vad import Endpointer, SileroVAD, default_model_path
from ultron.config import ConfigError, RootConfig, load_config
from ultron.events import AudioFrame, Utterance, WakeEvent
from ultron.stt.whisper_engine import WhisperEngine
from ultron.wake import create_wake_detector
from ultron.wake.matcher import MatchResult, WakeMatcher

_SAMPLE_RATE = 16000
_KWS_HEAD_SECONDS = 1.5
_KWS_INITIAL_PROMPT = "Altron."
_KWS_BEAM_SIZE = 1
_RECALL_BUDGET = 0.95
_FA_PER_HOUR_BUDGET = 1.0
_LATENCY_BUDGET_MS = 400
_DEFAULT_THRESHOLDS = (70, 75, 78, 80, 82, 85, 88, 90, 92, 95)
_DEFAULT_OWW_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85)
# docs/tz/10_wake_word_training.md §3.5: hybrid's own acceptance rule.
_HYBRID_RECALL_SLACK = 0.01
_HYBRID_LATENCY_BUDGET_MS = 250.0


@dataclass
class SampleScore:
    name: str
    score: float
    latency_ms: int


@dataclass
class ThresholdResult:
    threshold: float
    recall: float
    fa_per_hour: float
    n_positive_hits: int
    n_negative_segments: int
    n_negative_hits: int
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def meets_budget(self) -> bool:
        return self.recall >= _RECALL_BUDGET and self.fa_per_hour <= _FA_PER_HOUR_BUDGET


@dataclass
class ModeReport:
    """One row of docs/tz/10_wake_word_training.md §3.5's comparison table."""

    mode: str
    recall: float
    fa_per_hour: float
    latency_p95_ms: float
    cpu_percent: float
    gpu_busy_s: float
    n_positive: int
    n_negative_hits: int
    negative_hours: float
    detail: str  # e.g. chosen threshold, for the narrative section below the table


def _load_wav(path: Path) -> np.ndarray:
    pcm, sr = sf.read(path, dtype="int16")
    if sr != _SAMPLE_RATE:
        raise SystemExit(f"Kutilmagan chastota {sr} Hz: {path}")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    return np.asarray(pcm, dtype=np.int16)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


class _ResourceSampler:
    """``with _ResourceSampler() as s: ...`` then read ``s.cpu_percent``.

    ``psutil.Process.cpu_percent(interval=None)`` reports usage *since the
    last call* -- calling it once on ``__enter__`` (thrown away) establishes
    that baseline so ``__exit__``'s call reports exactly this block's usage.
    Unnormalized: 100% == one full core, matching Task Manager's per-core view.
    """

    def __init__(self) -> None:
        self._proc = psutil.Process()
        self.cpu_percent = 0.0

    def __enter__(self) -> _ResourceSampler:
        self._proc.cpu_percent(interval=None)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cpu_percent = self._proc.cpu_percent(interval=None)


class _GpuTimer:
    """Wraps an ``STTEngine.transcribe`` to accumulate real wall-clock time
    spent inside it -- the "GPU band vaqti" column, since every STT call in
    this project runs on the configured STT device (``cuda`` by default)."""

    def __init__(self) -> None:
        self.total_s = 0.0

    def wrap(self, engine: Any) -> None:
        original = engine.transcribe

        def timed(pcm: np.ndarray, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                return original(pcm, **kwargs)
            finally:
                self.total_s += time.monotonic() - start

        engine.transcribe = timed


# -- stt_kws (unchanged since 04-TZ) -----------------------------------------


def _build_kws_engine(config: RootConfig) -> WhisperEngine:
    return WhisperEngine(
        model=config.stt.kws_model,
        device=config.stt.kws_device,
        compute_type=config.stt.compute_type,
        language=config.stt.language,
        beam_size=_KWS_BEAM_SIZE,
        no_speech_threshold=config.stt.no_speech_threshold,
        log_prob_threshold=config.stt.log_prob_threshold,
        initial_prompt=_KWS_INITIAL_PROMPT,
        download_root=Path(config.app.data_dir) / "models" / "stt",
    )


def _effective_score(match: MatchResult) -> float:
    """The score to sweep thresholds against.

    ``matcher`` here is always built with ``fuzzy_threshold=0``, so
    ``match.matched`` is ``False`` only when the denylist rejected an exact
    match (e.g. "altin", "patron") -- a threshold-independent veto. Reporting
    ``match.score`` (its raw fuzz ratio, still high) for such a sample would
    make the sweep count it as a false accept at every threshold, when in
    real operation no threshold ever lets it through.
    """
    return match.score if match.matched else -1.0


def score_positives(
    engine: WhisperEngine, matcher: WakeMatcher, positives_dir: Path, gpu_timer: _GpuTimer
) -> list[SampleScore]:
    wav_paths = sorted(positives_dir.glob("*.wav"))
    if not wav_paths:
        raise SystemExit(f"Pozitiv namunalar topilmadi: {positives_dir}")

    scores: list[SampleScore] = []
    for path in wav_paths:
        pcm = _load_wav(path)
        head = pcm[: int(_KWS_HEAD_SECONDS * _SAMPLE_RATE)]
        start = time.monotonic()
        transcript = engine.transcribe(head, initial_prompt=_KWS_INITIAL_PROMPT)
        latency_ms = int((time.monotonic() - start) * 1000)
        gpu_timer.total_s += latency_ms / 1000
        match = matcher.match(transcript.text.lower())
        scores.append(
            SampleScore(name=path.name, score=_effective_score(match), latency_ms=latency_ms)
        )
    return scores


def _make_endpointer(config: RootConfig, ring: RingBuffer) -> Endpointer:
    endpointer = Endpointer(
        ring,
        sample_rate=config.audio.sample_rate,
        threshold=config.vad.threshold,
        min_speech_ms=config.vad.min_speech_ms,
        end_silence_ms=config.vad.end_silence_ms,
        max_utterance_s=config.vad.max_utterance_s,
        listen_timeout_s=config.vad.listen_timeout_s,
    )
    endpointer.set_mode("passive")
    return endpointer


def _segment_file(pcm: np.ndarray, config: RootConfig, vad_model_path: Path) -> list[Utterance]:
    ring = RingBuffer(config.audio.preroll_seconds, config.audio.sample_rate)
    vad = SileroVAD(vad_model_path)
    endpointer = _make_endpointer(config, ring)

    block = config.audio.block_size
    segments: list[Utterance] = []
    for i in range(0, len(pcm) - block + 1, block):
        chunk = pcm[i : i + block]
        ring.write(chunk)
        prob = vad.process(chunk)
        result = endpointer.push(chunk, prob)
        if isinstance(result, Utterance):
            segments.append(result)
    return segments


def score_negatives(
    engine: WhisperEngine,
    matcher: WakeMatcher,
    negatives_dir: Path,
    config: RootConfig,
    vad_model_path: Path,
    gpu_timer: _GpuTimer,
) -> tuple[list[SampleScore], float]:
    wav_paths = sorted(negatives_dir.glob("*.wav"))
    if not wav_paths:
        raise SystemExit(f"Negativ namunalar topilmadi: {negatives_dir}")

    scores: list[SampleScore] = []
    total_seconds = 0.0
    for path in wav_paths:
        pcm = _load_wav(path)
        total_seconds += len(pcm) / _SAMPLE_RATE
        segments = _segment_file(pcm, config, vad_model_path)
        logger.info(
            "{}: {:.1f} min, {} segment", path.name, len(pcm) / _SAMPLE_RATE / 60, len(segments)
        )
        for idx, seg in enumerate(segments):
            head = seg.pcm[: int(_KWS_HEAD_SECONDS * _SAMPLE_RATE)]
            start = time.monotonic()
            transcript = engine.transcribe(head, initial_prompt=_KWS_INITIAL_PROMPT)
            gpu_timer.total_s += time.monotonic() - start
            match = matcher.match(transcript.text.lower())
            scores.append(
                SampleScore(name=f"{path.name}#{idx}", score=_effective_score(match), latency_ms=0)
            )
    return scores, total_seconds / 3600.0


def sweep_thresholds(
    positive_scores: list[SampleScore],
    negative_scores: list[SampleScore],
    negative_hours: float,
    thresholds: list[float],
) -> list[ThresholdResult]:
    results = []
    n_pos = len(positive_scores)
    for t in thresholds:
        hit_positives = [s for s in positive_scores if s.score >= t]
        n_neg_hit = sum(1 for s in negative_scores if s.score >= t)
        recall = len(hit_positives) / n_pos if n_pos else 0.0
        fa_per_hour = n_neg_hit / negative_hours if negative_hours > 0 else float("inf")
        results.append(
            ThresholdResult(
                threshold=t,
                recall=recall,
                fa_per_hour=fa_per_hour,
                n_positive_hits=len(hit_positives),
                n_negative_segments=len(negative_scores),
                n_negative_hits=n_neg_hit,
                latencies_ms=[float(s.latency_ms) for s in hit_positives],
            )
        )
    return results


def pick_winner(results: list[ThresholdResult]) -> ThresholdResult | None:
    """Lowest threshold meeting both budgets (max recall within the FA budget);
    falls back to the highest-recall threshold that still meets the FA budget,
    then to ``None`` if nothing keeps false-accepts within budget at all."""
    meets_both = [r for r in results if r.meets_budget]
    if meets_both:
        return min(meets_both, key=lambda r: r.threshold)
    meets_fa = [r for r in results if r.fa_per_hour <= _FA_PER_HOUR_BUDGET]
    if meets_fa:
        return max(meets_fa, key=lambda r: r.recall)
    return None


def render_report(
    results: list[ThresholdResult],
    positive_scores: list[SampleScore],
    negative_hours: float,
    n_negative_segments: int,
) -> str:
    latencies = [float(s.latency_ms) for s in positive_scores]
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = _p95(latencies)
    latency_ok = "ha" if p95_latency <= _LATENCY_BUDGET_MS else "yo'q"

    lines = [
        f"# Wake benchmark (stt_kws) — {date.today().isoformat()}",
        "",
        f"Pozitiv: {len(positive_scores)} ta fayl. Negativ: {n_negative_segments} ta segment "
        f"({negative_hours:.2f} soat).",
        f"Qabul mezoni: recall >= {_RECALL_BUDGET * 100:.0f}%, "
        f"false accept <= {_FA_PER_HOUR_BUDGET:.0f}/soat, p95 latency <= {_LATENCY_BUDGET_MS} ms.",
        f"O'lchangan latency (qisqa \"Altron\" uchun, tez KWS o'tishi): "
        f"o'rtacha={avg_latency:.0f} ms, p95={p95_latency:.0f} ms ({latency_ok}).",
        "",
        "| fuzzy_threshold | Recall % | False accept/soat | Byudjetga mos |",
        "|---|---|---|---|",
    ]
    n_positive = len(positive_scores)
    for r in sorted(results, key=lambda x: x.threshold):
        budget_ok = "ha" if r.meets_budget else "yo'q"
        recall_pct = f"{r.recall * 100:.1f} ({r.n_positive_hits}/{n_positive})"
        fa = f"{r.fa_per_hour:.2f} ({r.n_negative_hits})"
        lines.append(f"| {r.threshold:.0f} | {recall_pct} | {fa} | {budget_ok} |")

    winner = pick_winner(results)
    lines.append("")
    if winner is not None:
        lines.append(
            f"**Tanlangan fuzzy_threshold:** {winner.threshold:.0f} "
            f"(recall={winner.recall * 100:.1f}%, false accept={winner.fa_per_hour:.2f}/soat)."
        )
        if not winner.meets_budget:
            lines.append(
                "\n**Diqqat:** g'olib ham recall/false-accept mezonini to'liq bajarmadi "
                "(false-accept byudjetiga mos keladigan eng yuqori recall'li qiymat tanlandi)."
            )
    else:
        lines.append(
            "**G'olib topilmadi:** hech bir threshold false-accept byudjetiga (<=1/soat) mos "
            "kelmadi. Negativ to'plamni tekshiring yoki `wake.phrases`/denylist'ni qayta ko'rib "
            "chiqing."
        )
    return "\n".join(lines) + "\n"


def write_config_selection(
    config_path: Path, *, key: str, value: float, fmt: str = "{:.0f}"
) -> None:
    text = config_path.read_text(encoding="utf-8")
    text, n = re.subn(
        rf"(?m)^(wake:\n(?:[ \t].*\n)*?[ \t]+{re.escape(key)}:\s*)[0-9.]+",
        lambda m: f"{m.group(1)}{fmt.format(value)}",
        text,
        count=1,
    )
    if n == 0:
        logger.warning(
            "config.yaml'dagi wake.{} qatori avtomatik topilmadi, qo'lda yangilang: {}", key, value
        )
        return
    config_path.write_text(text, encoding="utf-8")
    logger.info("config.yaml yangilandi: wake.{}={}", key, fmt.format(value))


def set_config_mode(config_path: Path, *, mode: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    text, n = re.subn(
        r'(?m)^(wake:\n(?:[ \t].*\n)*?[ \t]+mode:\s*)"[a-z_]+"',
        lambda m: f'{m.group(1)}"{mode}"',
        text,
        count=1,
    )
    if n == 0:
        logger.warning(
            "config.yaml'dagi wake.mode qatori avtomatik topilmadi, qo'lda yangilang: {}", mode
        )
        return
    config_path.write_text(text, encoding="utf-8")
    logger.info("config.yaml yangilandi: wake.mode={}", mode)


def run_stt_kws_benchmark(
    config: RootConfig,
    positives_dir: Path,
    negatives_dir: Path,
    vad_model_path: Path,
    thresholds: list[float],
) -> tuple[list[ThresholdResult], ModeReport]:
    matcher = WakeMatcher(
        phrases=config.wake.phrases,
        fuzzy_threshold=0,
        max_prefix_words=config.wake.max_prefix_words,
    )
    engine = _build_kws_engine(config)
    logger.info("stt_kws: KWS modeli yuklanmoqda ({})", config.stt.kws_model)
    engine.load()

    gpu_timer = _GpuTimer()
    with _ResourceSampler() as sampler:
        logger.info("stt_kws: pozitivlar baholanmoqda ({})", positives_dir)
        positive_scores = score_positives(engine, matcher, positives_dir, gpu_timer)
        logger.info("stt_kws: negativlar baholanmoqda ({})", negatives_dir)
        negative_scores, negative_hours = score_negatives(
            engine, matcher, negatives_dir, config, vad_model_path, gpu_timer
        )

    results = sweep_thresholds(positive_scores, negative_scores, negative_hours, thresholds)
    winner = pick_winner(results)
    latencies = [float(s.latency_ms) for s in positive_scores]
    n_hits = winner.n_negative_hits if winner else 0
    report = ModeReport(
        mode="stt_kws",
        recall=winner.recall if winner else 0.0,
        fa_per_hour=winner.fa_per_hour if winner else float("inf"),
        latency_p95_ms=_p95(latencies),
        cpu_percent=sampler.cpu_percent,
        gpu_busy_s=gpu_timer.total_s,
        n_positive=len(positive_scores),
        n_negative_hits=n_hits,
        negative_hours=negative_hours,
        detail=f"fuzzy_threshold={winner.threshold:.0f}" if winner else "g'olib topilmadi",
    )
    return results, report


# -- oww (docs/tz/10_wake_word_training.md §3.5) -----------------------------


def _require_oww_model(config: RootConfig) -> tuple[Path, Path, Path]:
    model_path = Path(config.wake.oww_model_path)
    if not model_path.is_file():
        raise SystemExit(
            f"openWakeWord modeli topilmadi: {model_path}\n"
            "Bu model docs/tz/10_wake_word_training.md bosqichida o'qitiladi."
        )
    feature_dir = model_path.parent
    melspec_path = feature_dir / "melspectrogram.onnx"
    embedding_path = feature_dir / "embedding_model.onnx"
    if not melspec_path.is_file() or not embedding_path.is_file():
        raise SystemExit(
            f"openWakeWord umumiy feature modellari topilmadi ({feature_dir}). "
            "Ishga tushirish: uv run python scripts/download_models.py --oww"
        )
    return model_path, melspec_path, embedding_path


def _new_oww_model(model_path: Path, melspec_path: Path, embedding_path: Path) -> Any:
    from openwakeword.model import Model

    return Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        melspec_model_path=str(melspec_path),
        embedding_model_path=str(embedding_path),
        vad_threshold=0,
    )


def _oww_frame_scores(model: Any, model_key: str, pcm: np.ndarray, block_size: int) -> np.ndarray:
    n_frames = len(pcm) // block_size
    scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = pcm[i * block_size : (i + 1) * block_size]
        prediction = model.predict(chunk)
        scores[i] = prediction.get(model_key, 0.0)
    return scores


@dataclass
class OwwFileScore:
    name: str
    frame_scores: np.ndarray
    word_end_s: float  # docstring above: file_duration - vad.end_silence_ms


def score_positives_oww(
    model_path: Path,
    melspec_path: Path,
    embedding_path: Path,
    positives_dir: Path,
    config: RootConfig,
) -> tuple[list[OwwFileScore], float]:
    wav_paths = sorted(positives_dir.glob("*.wav"))
    if not wav_paths:
        raise SystemExit(f"Pozitiv namunalar topilmadi: {positives_dir}")

    model_key = model_path.stem
    frame_dur = config.audio.block_size / _SAMPLE_RATE
    end_silence_s = config.vad.end_silence_ms / 1000.0
    results = []
    for path in wav_paths:
        pcm = _load_wav(path)
        model = _new_oww_model(
            model_path, melspec_path, embedding_path
        )  # fresh: no cross-file state
        scores = _oww_frame_scores(model, model_key, pcm, config.audio.block_size)
        word_end_s = max(0.0, len(pcm) / _SAMPLE_RATE - end_silence_s)
        results.append(OwwFileScore(name=path.name, frame_scores=scores, word_end_s=word_end_s))
    return results, frame_dur


def score_negatives_oww(
    model_path: Path,
    melspec_path: Path,
    embedding_path: Path,
    negatives_dir: Path,
    config: RootConfig,
) -> tuple[list[np.ndarray], float]:
    wav_paths = sorted(negatives_dir.glob("*.wav"))
    if not wav_paths:
        raise SystemExit(f"Negativ namunalar topilmadi: {negatives_dir}")

    model_key = model_path.stem
    total_seconds = 0.0
    results = []
    for path in wav_paths:
        pcm = _load_wav(path)
        total_seconds += len(pcm) / _SAMPLE_RATE
        model = _new_oww_model(model_path, melspec_path, embedding_path)
        results.append(_oww_frame_scores(model, model_key, pcm, config.audio.block_size))
    return results, total_seconds / 3600.0


def sweep_oww_thresholds(
    positive_results: list[OwwFileScore],
    negative_results: list[np.ndarray],
    negative_hours: float,
    frame_dur_s: float,
    thresholds: list[float],
) -> list[ThresholdResult]:
    results = []
    n_pos = len(positive_results)
    for t in thresholds:
        n_pos_hit = 0
        latencies_ms = []
        for pos in positive_results:
            hits = np.nonzero(pos.frame_scores >= t)[0]
            if hits.size == 0:
                continue
            n_pos_hit += 1
            trigger_s = float(hits[0]) * frame_dur_s
            latencies_ms.append((trigger_s - pos.word_end_s) * 1000)

        n_neg_hit = 0
        for scores in negative_results:
            above = scores >= t
            if above.size == 0:
                continue
            # Rising-edge count: a continuous run above threshold is one event,
            # matching OWWDetector's cooldown collapsing repeats in real use.
            n_neg_hit += int(np.count_nonzero(above[1:] & ~above[:-1])) + int(above[0])

        recall = n_pos_hit / n_pos if n_pos else 0.0
        fa_per_hour = n_neg_hit / negative_hours if negative_hours > 0 else float("inf")
        results.append(
            ThresholdResult(
                threshold=t,
                recall=recall,
                fa_per_hour=fa_per_hour,
                n_positive_hits=n_pos_hit,
                n_negative_segments=len(negative_results),
                n_negative_hits=n_neg_hit,
                latencies_ms=latencies_ms,
            )
        )
    return results


def render_oww_report(
    results: list[ThresholdResult], n_positive: int, negative_hours: float
) -> str:
    lines = [
        f"# Wake benchmark (oww) — {date.today().isoformat()}",
        "",
        f"Pozitiv: {n_positive} ta fayl. Negativ: {negative_hours:.2f} soat.",
        f"Qabul mezoni: recall >= {_RECALL_BUDGET * 100:.0f}%, "
        f"false accept <= {_FA_PER_HOUR_BUDGET:.0f}/soat.",
        "",
        "| oww_threshold | Recall % | False accept/soat | Latency p95 (ms) | Byudjetga mos |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: x.threshold):
        budget_ok = "ha" if r.meets_budget else "yo'q"
        recall_pct = f"{r.recall * 100:.1f} ({r.n_positive_hits}/{n_positive})"
        fa = f"{r.fa_per_hour:.2f} ({r.n_negative_hits})"
        p95 = f"{_p95(r.latencies_ms):.0f}"
        lines.append(f"| {r.threshold:.2f} | {recall_pct} | {fa} | {p95} | {budget_ok} |")
    winner = pick_winner(results)
    lines.append("")
    if winner is not None:
        lines.append(
            f"**Tanlangan oww_threshold:** {winner.threshold:.2f} "
            f"(recall={winner.recall * 100:.1f}%, false accept={winner.fa_per_hour:.2f}/soat)."
        )
    else:
        lines.append(
            "**G'olib topilmadi:** hech bir threshold false-accept byudjetiga mos kelmadi."
        )
    return "\n".join(lines) + "\n"


def run_oww_benchmark(
    config: RootConfig, positives_dir: Path, negatives_dir: Path, thresholds: list[float]
) -> tuple[list[ThresholdResult], ModeReport]:
    model_path, melspec_path, embedding_path = _require_oww_model(config)

    with _ResourceSampler() as sampler:
        logger.info("oww: pozitivlar baholanmoqda ({})", positives_dir)
        positive_results, frame_dur = score_positives_oww(
            model_path, melspec_path, embedding_path, positives_dir, config
        )
        logger.info("oww: negativlar baholanmoqda ({})", negatives_dir)
        negative_results, negative_hours = score_negatives_oww(
            model_path, melspec_path, embedding_path, negatives_dir, config
        )

    results = sweep_oww_thresholds(
        positive_results, negative_results, negative_hours, frame_dur, thresholds
    )
    winner = pick_winner(results)
    report = ModeReport(
        mode="oww",
        recall=winner.recall if winner else 0.0,
        fa_per_hour=winner.fa_per_hour if winner else float("inf"),
        latency_p95_ms=_p95(winner.latencies_ms) if winner else 0.0,
        cpu_percent=sampler.cpu_percent,
        gpu_busy_s=0.0,  # raw-score sweep never calls STT (see module docstring)
        n_positive=len(positive_results),
        n_negative_hits=winner.n_negative_hits if winner else 0,
        negative_hours=negative_hours,
        detail=f"oww_threshold={winner.threshold:.2f}" if winner else "g'olib topilmadi",
    )
    return results, report


# -- hybrid: one real, live simulation (docs/tz/04_wake_word.md §3.4) -------


async def _simulate_wake_detector(
    wake_detector: Any, pcm: np.ndarray, config: RootConfig, vad_model_path: Path
) -> list[tuple[float, WakeEvent]]:
    """Replays ``pcm`` through ``wake_detector`` exactly like
    ``Orchestrator._audio_loop`` does live: every frame goes to
    ``process_frame`` *and* through VAD/Endpointer, whose completed
    ``Utterance``s go to ``process_utterance`` -- so a ``HybridDetector``'s
    real dedup between the two paths is exercised, not approximated."""
    ring = RingBuffer(config.audio.preroll_seconds + 1.0, config.audio.sample_rate)
    vad = SileroVAD(vad_model_path)
    endpointer = _make_endpointer(config, ring)
    block = config.audio.block_size
    sample_rate = config.audio.sample_rate

    events: list[tuple[float, WakeEvent]] = []
    for i in range(0, len(pcm) - block + 1, block):
        chunk = pcm[i : i + block]
        t = i / sample_rate
        ring.write(chunk)
        prob = vad.process(chunk)

        frame_event = wake_detector.process_frame(AudioFrame(pcm=chunk))
        if frame_event is not None:
            events.append((t, frame_event))

        result = endpointer.push(chunk, prob)
        if isinstance(result, Utterance):
            utt_event = await wake_detector.process_utterance(result)
            if utt_event is not None:
                events.append((t, utt_event))
    return events


async def _run_hybrid_pass(
    config: RootConfig, positives_dir: Path, negatives_dir: Path, vad_model_path: Path
) -> tuple[list[float], list[bool], int, float, _GpuTimer]:
    """Returns (positive_latencies_ms, positive_hits, n_negative_hits,
    negative_hours, gpu_timer)."""
    hybrid_config = config.model_copy(deep=True)
    hybrid_config.wake.mode = "hybrid"  # force hybrid regardless of config.yaml's current wake.mode
    # One detector + ring buffer for the whole run (``.reset()`` between files
    # below clears cooldown/queue state, not the ring buffer itself) -- a few
    # seconds of the previous file's tail can still be in it early in the next
    # file, same tradeoff as reusing one STT engine across files elsewhere in
    # this script. Acceptable for benchmarking; a real run is one continuous
    # stream anyway, so this is arguably closer to production than resetting.
    ring_buffer = RingBuffer(config.audio.preroll_seconds + 1.0, config.audio.sample_rate)
    wake = create_wake_detector(hybrid_config, ring_buffer=ring_buffer)
    if not hasattr(wake, "_oww"):  # model missing -> factory already fell back to stt_kws
        raise SystemExit(
            f"openWakeWord modeli topilmadi: {config.wake.oww_model_path}\n"
            "Bu model docs/tz/10_wake_word_training.md bosqichida o'qitiladi."
        )
    wake.load()

    gpu_timer = _GpuTimer()
    gpu_timer.wrap(wake._stt_kws.kws_engine)
    if wake._stt_kws.main_engine is not wake._stt_kws.kws_engine:
        gpu_timer.wrap(wake._stt_kws.main_engine)

    end_silence_s = config.vad.end_silence_ms / 1000.0
    latencies_ms: list[float] = []
    hits: list[bool] = []
    for path in sorted(positives_dir.glob("*.wav")):
        pcm = _load_wav(path)
        wake.reset()
        events = await _simulate_wake_detector(wake, pcm, config, vad_model_path)
        hits.append(bool(events))
        if events:
            word_end_s = max(0.0, len(pcm) / _SAMPLE_RATE - end_silence_s)
            latencies_ms.append((events[0][0] - word_end_s) * 1000)

    n_negative_hits = 0
    negative_seconds = 0.0
    for path in sorted(negatives_dir.glob("*.wav")):
        pcm = _load_wav(path)
        negative_seconds += len(pcm) / _SAMPLE_RATE
        wake.reset()
        events = await _simulate_wake_detector(wake, pcm, config, vad_model_path)
        n_negative_hits += len(events)

    return latencies_ms, hits, n_negative_hits, negative_seconds / 3600.0, gpu_timer


def run_hybrid_benchmark(
    config: RootConfig, positives_dir: Path, negatives_dir: Path, vad_model_path: Path
) -> ModeReport:
    with _ResourceSampler() as sampler:
        latencies_ms, hits, n_negative_hits, negative_hours, gpu_timer = asyncio.run(
            _run_hybrid_pass(config, positives_dir, negatives_dir, vad_model_path)
        )

    n_positive = len(hits)
    recall = sum(hits) / n_positive if n_positive else 0.0
    fa_per_hour = n_negative_hits / negative_hours if negative_hours > 0 else float("inf")
    return ModeReport(
        mode="hybrid",
        recall=recall,
        fa_per_hour=fa_per_hour,
        latency_p95_ms=_p95(latencies_ms),
        cpu_percent=sampler.cpu_percent,
        gpu_busy_s=gpu_timer.total_s,
        n_positive=n_positive,
        n_negative_hits=n_negative_hits,
        negative_hours=negative_hours,
        detail=(
            f"wake.fuzzy_threshold={config.wake.fuzzy_threshold}, "
            f"wake.oww_threshold={config.wake.oww_threshold}"
        ),
    )


# -- combined report (docs/tz/10_wake_word_training.md §3.5's table) --------


def hybrid_meets_acceptance(hybrid: ModeReport, stt_kws: ModeReport) -> bool:
    return (
        hybrid.recall >= stt_kws.recall - _HYBRID_RECALL_SLACK
        and hybrid.latency_p95_ms <= _HYBRID_LATENCY_BUDGET_MS
    )


def render_combined_report(reports: list[ModeReport]) -> str:
    by_mode = {r.mode: r for r in reports}
    lines = [
        f"# Wake benchmark — barcha rejimlar — {date.today().isoformat()}",
        "",
        "docs/tz/10_wake_word_training.md §3.5.",
        "",
        "| Rejim | Recall | FA/soat | Latency p95 | CPU % | GPU band vaqti |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.mode} ({r.detail}) | {r.recall * 100:.1f}% ({r.n_positive}) "
            f"| {r.fa_per_hour:.2f} ({r.n_negative_hits}/{r.negative_hours:.2f}h) "
            f"| {r.latency_p95_ms:.0f} ms | {r.cpu_percent:.0f}% | {r.gpu_busy_s:.1f}s |"
        )

    lines.append("")
    stt_kws = by_mode.get("stt_kws")
    hybrid = by_mode.get("hybrid")
    if stt_kws is not None and hybrid is not None:
        if hybrid_meets_acceptance(hybrid, stt_kws):
            lines.append(
                f"**Qabul qilindi:** hybrid (recall={hybrid.recall * 100:.1f}% >= "
                f"stt_kws {stt_kws.recall * 100:.1f}% - {_HYBRID_RECALL_SLACK * 100:.0f}%, "
                f"latency p95={hybrid.latency_p95_ms:.0f}ms <= {_HYBRID_LATENCY_BUDGET_MS:.0f}ms). "
                "`wake.mode: hybrid` config.yaml'ga yozildi."
            )
        else:
            lines.append(
                f"**Rad etildi:** hybrid mezonni bajarmadi "
                f"(recall={hybrid.recall * 100:.1f}% vs stt_kws {stt_kws.recall * 100:.1f}%, "
                f"latency p95={hybrid.latency_p95_ms:.0f}ms vs "
                f"{_HYBRID_LATENCY_BUDGET_MS:.0f}ms byudjet). `wake.mode` `stt_kws`da qoladi."
            )
    return "\n".join(lines) + "\n"


# -- CLI ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron wake-word benchmark")
    parser.add_argument("--positives", type=Path, default=None)
    parser.add_argument("--negatives", type=Path, default=None)
    parser.add_argument("--mode", choices=["stt_kws", "oww", "hybrid", "all"], default="stt_kws")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(_DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--oww-thresholds", nargs="+", type=float, default=list(_DEFAULT_OWW_THRESHOLDS)
    )
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

    data_dir = Path(config.app.data_dir)
    positives_dir = args.positives or (data_dir / "recordings" / "wake")
    negatives_dir = args.negatives or (data_dir / "recordings" / "wake_neg")
    vad_model_path = default_model_path(data_dir)
    if not vad_model_path.is_file():
        raise SystemExit(
            f"Silero VAD modeli topilmadi: {vad_model_path}\n"
            "Avval ishga tushiring: uv run python scripts/download_models.py --vad"
        )

    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path("config/config.yaml")

    # --mode stt_kws keeps the exact 04-TZ single-mode report shape/behavior.
    if args.mode == "stt_kws":
        return _run_stt_kws_single(
            config, positives_dir, negatives_dir, vad_model_path, args, reports_dir, config_path
        )

    if args.mode == "oww":
        oww_results, oww_report = run_oww_benchmark(
            config, positives_dir, negatives_dir, args.oww_thresholds
        )
        report = render_oww_report(oww_results, oww_report.n_positive, oww_report.negative_hours)
        report_path = reports_dir / f"wake_benchmark_oww_{date.today().strftime('%Y%m%d')}.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Hisobot yozildi: {}", report_path)
        logger.info("\n{}", report)
        if not args.no_write_config:
            winner = pick_winner(oww_results)
            if winner is not None:
                write_config_selection(
                    config_path, key="oww_threshold", value=winner.threshold, fmt="{:.2f}"
                )
        return 0

    if args.mode == "hybrid":
        hybrid_report = run_hybrid_benchmark(config, positives_dir, negatives_dir, vad_model_path)
        report = render_combined_report([hybrid_report])
        report_path = reports_dir / f"wake_benchmark_hybrid_{date.today().strftime('%Y%m%d')}.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("Hisobot yozildi: {}", report_path)
        logger.info("\n{}", report)
        return 0

    # --mode all: each benchmark runs exactly once; the sweep results it
    # already computed are reused below for config-writing (no repeat runs).
    stt_results, stt_report = run_stt_kws_benchmark(
        config, positives_dir, negatives_dir, vad_model_path, args.thresholds
    )
    oww_results, oww_report = run_oww_benchmark(
        config, positives_dir, negatives_dir, args.oww_thresholds
    )
    hybrid_report = run_hybrid_benchmark(config, positives_dir, negatives_dir, vad_model_path)

    report = render_combined_report([stt_report, oww_report, hybrid_report])
    report_path = reports_dir / f"wake_benchmark_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Hisobot yozildi: {}", report_path)
    logger.info("\n{}", report)

    if not args.no_write_config:
        stt_winner = pick_winner(stt_results)
        if stt_winner is not None:
            write_config_selection(config_path, key="fuzzy_threshold", value=stt_winner.threshold)
        oww_winner = pick_winner(oww_results)
        if oww_winner is not None:
            write_config_selection(
                config_path, key="oww_threshold", value=oww_winner.threshold, fmt="{:.2f}"
            )
        if hybrid_meets_acceptance(hybrid_report, stt_report):
            set_config_mode(config_path, mode="hybrid")

    return 0


def _run_stt_kws_single(
    config: RootConfig,
    positives_dir: Path,
    negatives_dir: Path,
    vad_model_path: Path,
    args: argparse.Namespace,
    reports_dir: Path,
    config_path: Path,
) -> int:
    """The original 04-TZ single-mode stt_kws report (kept byte-for-byte
    compatible with pre-10-TZ output) -- run standalone here rather than
    threading a duplicate code path through ``main()`` above."""
    matcher = WakeMatcher(
        phrases=config.wake.phrases,
        fuzzy_threshold=0,
        max_prefix_words=config.wake.max_prefix_words,
    )
    engine = _build_kws_engine(config)
    logger.info("KWS modeli yuklanmoqda: {} ({})", config.stt.kws_model, config.stt.kws_device)
    engine.load()

    gpu_timer = _GpuTimer()
    logger.info("Pozitivlar baholanmoqda: {}", positives_dir)
    positive_scores = score_positives(engine, matcher, positives_dir, gpu_timer)
    logger.info("Negativlar baholanmoqda: {}", negatives_dir)
    negative_scores, negative_hours = score_negatives(
        engine, matcher, negatives_dir, config, vad_model_path, gpu_timer
    )

    results = sweep_thresholds(positive_scores, negative_scores, negative_hours, args.thresholds)
    report = render_report(results, positive_scores, negative_hours, len(negative_scores))

    report_path = reports_dir / f"wake_benchmark_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Hisobot yozildi: {}", report_path)
    logger.info("\n{}", report)

    if not args.no_write_config:
        winner = pick_winner(results)
        if winner is not None:
            write_config_selection(config_path, key="fuzzy_threshold", value=winner.threshold)

    return 0


if __name__ == "__main__":
    sys.exit(main())
