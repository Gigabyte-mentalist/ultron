"""Sample-recording console tool (docs/tz/02_audio.md §3.6, docs/tz/04_wake_word.md §3.6).

    uv run python scripts/record_samples.py mic-test
    uv run python scripts/record_samples.py vad-live
    uv run python scripts/record_samples.py wake --count 100
    uv run python scripts/record_samples.py testset --file prompts_uz.txt
    uv run python scripts/record_samples.py wake-neg --minutes 30
    uv run python scripts/record_samples.py wake-hardneg --reps 20

``wake`` and ``testset`` save to data/recordings/<kind>/NNNN.wav; ``testset``
additionally appends to transcripts.tsv. ``wake-neg`` records one continuous
(not VAD-segmented) clip for scripts/benchmark_wake.py's negative set --
ordinary talking, TV, music, anything *without* the wake word -- and saves
it to data/recordings/wake_neg/NNNN.wav. ``wake-hardneg`` prompts each of
docs/tz/10_wake_word_training.md §3.2's "qiyin negativ" words/phrases
("elektron", "patron", "Aldar", "alt", "atrof", "Ultra", "kontrol", "altin")
``--reps`` times each (one VAD-isolated utterance per rep, like ``wake``) and
saves to data/recordings/wake_hardneg/NNNN.wav + transcripts.tsv. Needs a
real microphone -- run this yourself, Claude Code does not have one.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import soundfile as sf
from loguru import logger

from ultron.audio.capture import AudioCapture
from ultron.audio.ring_buffer import RingBuffer
from ultron.audio.vad import Endpointer, SileroVAD, default_model_path
from ultron.config import ConfigError, RootConfig, load_config
from ultron.events import Utterance

_GRAPH_BUCKET_S = 0.25
_GRAPH_WIDTH = 40

# docs/tz/10_wake_word_training.md §3.2: "qiyin negativ"lar -- fonetik jihatdan
# "altron"/"ultron"ga yaqin so'zlar, har biri kamida 20 marta yoziladi.
_HARD_NEGATIVE_WORDS = [
    "elektron",
    "patron",
    "Aldar",
    "alt",
    "atrof",
    "Ultra",
    "kontrol",
    "altin",
]


def _vad_model_path(config: RootConfig) -> Path:
    path = default_model_path(config.app.data_dir)
    if not path.is_file():
        raise SystemExit(
            f"Silero VAD modeli topilmadi: {path}\n"
            "Avval ishga tushiring: uv run python scripts/download_models.py --vad"
        )
    return path


def _rms_dbfs(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    return -math.inf if rms <= 0 else 20 * math.log10(rms / 32768.0)


def _make_endpointer(config: RootConfig, ring: RingBuffer) -> Endpointer:
    return Endpointer(
        ring,
        sample_rate=config.audio.sample_rate,
        threshold=config.vad.threshold,
        min_speech_ms=config.vad.min_speech_ms,
        end_silence_ms=config.vad.end_silence_ms,
        max_utterance_s=config.vad.max_utterance_s,
        listen_timeout_s=config.vad.listen_timeout_s,
    )


def _make_capture(config: RootConfig) -> AudioCapture:
    return AudioCapture(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        block_size=config.audio.block_size,
        input_device=config.audio.input_device,
    )


async def cmd_mic_test(config: RootConfig) -> None:
    capture = _make_capture(config)
    vad = SileroVAD(_vad_model_path(config))
    duration_s = 5.0
    frames_needed = int(duration_s * config.audio.sample_rate / config.audio.block_size)

    logger.info("{:.0f} soniya yozilmoqda...", duration_s)
    rows: list[tuple[float, float, float]] = []
    n = 0
    frame_dur_s = config.audio.block_size / config.audio.sample_rate
    async for frame in capture.frames():
        prob = vad.process(frame.pcm)
        rows.append((n * frame_dur_s, _rms_dbfs(frame.pcm), prob))
        n += 1
        if n >= frames_needed:
            break
    capture.stop()

    logger.info("Vaqt(s)  RMS(dBFS)  VAD")
    bucket_frames = max(1, int(_GRAPH_BUCKET_S / frame_dur_s))
    for i in range(0, len(rows), bucket_frames):
        chunk = rows[i : i + bucket_frames]
        t0 = chunk[0][0]
        avg_rms = sum(r[1] for r in chunk if r[1] != float("-inf")) / max(1, len(chunk))
        avg_prob = sum(r[2] for r in chunk) / len(chunk)
        bar = "#" * round(avg_prob * _GRAPH_WIDTH)
        logger.info("{:6.2f}  {:8.1f}  {:<40} {:.2f}", t0, avg_rms, bar, avg_prob)


async def cmd_vad_live(config: RootConfig) -> None:
    capture = _make_capture(config)
    ring = RingBuffer(config.audio.preroll_seconds, config.audio.sample_rate)
    vad = SileroVAD(_vad_model_path(config))
    endpointer = _make_endpointer(config, ring)
    endpointer.set_mode("passive")

    logger.info("Jonli VAD boshlandi. To'xtatish uchun Ctrl+C.")
    segment_no = 0
    async for frame in capture.frames():
        ring.write(frame.pcm)
        prob = vad.process(frame.pcm)
        result = endpointer.push(frame.pcm, prob)
        if isinstance(result, Utterance):
            segment_no += 1
            duration = result.ended_at - result.started_at
            logger.info(
                "Segment {}: {:.2f}s (sabab={}, namunalar={})",
                segment_no,
                duration,
                result.reason,
                len(result.pcm),
            )
    capture.stop()
    logger.info("Jami {} segment aniqlandi.", segment_no)


def _next_sample_index(out_dir: Path) -> int:
    existing = [int(p.stem) for p in out_dir.glob("*.wav") if p.stem.isdigit()]
    return max(existing, default=-1) + 1


async def _record_one_utterance(
    capture: AudioCapture, ring: RingBuffer, vad: SileroVAD, endpointer: Endpointer
) -> Utterance:
    capture.clear_buffered_frames()
    endpointer.reset()
    endpointer.set_mode("passive")
    async for frame in capture.frames():
        ring.write(frame.pcm)
        prob = vad.process(frame.pcm)
        result = endpointer.push(frame.pcm, prob)
        if isinstance(result, Utterance):
            return result
    raise RuntimeError("capture stopped before an utterance was captured")


async def _record_batch(
    config: RootConfig, *, kind: str, total: int, prompts: list[str] | None
) -> None:
    out_dir = Path(config.app.data_dir) / "recordings" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_path = out_dir / "transcripts.tsv" if prompts is not None else None

    capture = _make_capture(config)
    ring = RingBuffer(config.audio.preroll_seconds, config.audio.sample_rate)
    vad = SileroVAD(_vad_model_path(config))
    endpointer = _make_endpointer(config, ring)

    next_index = _next_sample_index(out_dir)
    recorded = 0
    try:
        while recorded < total:
            prompt_text = prompts[recorded] if prompts is not None else None
            if prompt_text:
                logger.info('[{}/{}] O\'qing: "{}"', recorded + 1, total, prompt_text)
            else:
                logger.info(
                    '[{}/{}] Tayyor bo\'lganda Enter bosing va "Altron" deb gapiring',
                    recorded + 1,
                    total,
                )
            await asyncio.to_thread(input, "Enter -> yozib boshlash: ")

            utt = await _record_one_utterance(capture, ring, vad, endpointer)

            target = out_dir / f"{next_index:04d}.wav"
            sf.write(target, utt.pcm, config.audio.sample_rate, subtype="PCM_16")
            logger.info("Saqlandi: {} ({:.2f}s)", target, utt.ended_at - utt.started_at)

            choice = await asyncio.to_thread(
                input, "Qayta yozish uchun 'r', keyingisi uchun Enter: "
            )
            if choice.strip().lower() == "r":
                target.unlink(missing_ok=True)
                continue

            if transcripts_path is not None:
                with transcripts_path.open("a", encoding="utf-8") as f:
                    f.write(f"{target.name}\t{prompt_text}\n")
            next_index += 1
            recorded += 1
    finally:
        capture.stop()

    logger.info("Tugadi: {} namuna {} papkasiga saqlandi.", total, out_dir)


async def cmd_wake(config: RootConfig, count: int) -> None:
    await _record_batch(config, kind="wake", total=count, prompts=None)


async def cmd_wake_neg(config: RootConfig, minutes: float) -> None:
    out_dir = Path(config.app.data_dir) / "recordings" / "wake_neg"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{_next_sample_index(out_dir):04d}.wav"

    capture = _make_capture(config)
    frames_needed = int(minutes * 60 * config.audio.sample_rate / config.audio.block_size)

    logger.info(
        "{:.0f} daqiqa yozib olinmoqda (oddiy gap/TV/musiqa, wake so'zsiz).",
        minutes,
    )
    logger.info("To'xtatish uchun Ctrl+C.")
    frames_per_minute = 60 * config.audio.sample_rate // config.audio.block_size
    chunks: list[np.ndarray] = []
    n = 0
    try:
        async for frame in capture.frames():
            chunks.append(frame.pcm)
            n += 1
            if n % (5 * frames_per_minute) == 0:
                logger.info("{:.1f} daqiqa yozildi...", n / frames_per_minute)
            if n >= frames_needed:
                break
    finally:
        capture.stop()

    pcm = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    sf.write(target, pcm, config.audio.sample_rate, subtype="PCM_16")
    logger.info("Saqlandi: {} ({:.1f} daqiqa)", target, len(pcm) / config.audio.sample_rate / 60)


async def cmd_wake_hardneg(config: RootConfig, reps: int) -> None:
    prompts = [word for _ in range(reps) for word in _HARD_NEGATIVE_WORDS]
    await _record_batch(config, kind="wake_hardneg", total=len(prompts), prompts=prompts)


async def cmd_testset(config: RootConfig, prompts_file: Path) -> None:
    if not prompts_file.is_file():
        raise SystemExit(f"Prompts fayli topilmadi: {prompts_file}")
    lines = prompts_file.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip()]
    if not prompts:
        raise SystemExit(f"Prompts fayli bo'sh: {prompts_file}")
    await _record_batch(config, kind="testset", total=len(prompts), prompts=prompts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron audio namunalarini yozish vositasi")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("mic-test", help="5 s yozib RMS/VAD grafigini matnda ko'rsatadi")
    sub.add_parser("vad-live", help="Jonli VAD: har bir gap segmentini ko'rsatadi (Ctrl+C)")

    p_wake = sub.add_parser("wake", help="Wake word namunalarini yozish (04/10-bosqich uchun)")
    p_wake.add_argument("--count", type=int, default=100)

    p_wake_neg = sub.add_parser(
        "wake-neg", help="Wake benchmark uchun uzluksiz negativ namuna yozish (04-bosqich)"
    )
    p_wake_neg.add_argument("--minutes", type=float, default=30.0)

    p_wake_hardneg = sub.add_parser(
        "wake-hardneg", help="Qiyin negativ so'zlarni yozish (10-bosqich)"
    )
    p_wake_hardneg.add_argument("--reps", type=int, default=20)

    p_testset = sub.add_parser("testset", help="STT test to'plamini yozish (03-bosqich uchun)")
    p_testset.add_argument("--file", type=Path, required=True)

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

    try:
        if args.command == "mic-test":
            asyncio.run(cmd_mic_test(config))
        elif args.command == "vad-live":
            asyncio.run(cmd_vad_live(config))
        elif args.command == "wake":
            asyncio.run(cmd_wake(config, args.count))
        elif args.command == "wake-neg":
            asyncio.run(cmd_wake_neg(config, args.minutes))
        elif args.command == "wake-hardneg":
            asyncio.run(cmd_wake_hardneg(config, args.reps))
        elif args.command == "testset":
            asyncio.run(cmd_testset(config, args.file))
    except KeyboardInterrupt:
        logger.info("To'xtatildi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
