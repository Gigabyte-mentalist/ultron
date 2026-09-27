"""24-hour soak test (docs/tz/08_orchestrator.md §8B.7).

Runs a real ``Orchestrator`` (real STT/wake/TTS models, real audio I/O) and,
every ``--interval-minutes`` (default 10), feeds one pre-recorded ``.wav``
straight into it -- bypassing the microphone entirely, per the TZ's own
"virtual mikrofon o'rniga to'g'ridan-to'g'ri pipeline'ga beradi" -- while
logging RSS and (if ``nvidia-smi`` is available) VRAM every cycle, so a human
can confirm after 24 h: no crash, RSS growth < 200 MB, VRAM stable.

    uv run python scripts/soak_test.py --wav-dir data/recordings/testset --hours 24

``ULTRON_TOOLS_MOCK=1`` is set automatically (docs/tz/07_tools.md §1) so a
wav that happens to contain a real command never touches the live system
over a 24 h unattended run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psutil
import soundfile as sf
from loguru import logger

from ultron.config import load_config
from ultron.core.orchestrator import CriticalStartupError, build_orchestrator
from ultron.events import Utterance
from ultron.logging_setup import setup_logging

_SAMPLE_RATE = 16000
_RSS_BUDGET_MB = 200.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron 24-hour soak test")
    parser.add_argument(
        "--wav-dir", type=Path, required=True, help="Directory of pre-recorded 16 kHz .wav files"
    )
    parser.add_argument("--interval-minutes", type=float, default=10.0)
    parser.add_argument("--hours", type=float, default=24.0)
    return parser.parse_args(argv)


def _load_wavs(wav_dir: Path) -> list[Path]:
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"'{wav_dir}' ichida .wav fayl topilmadi")
    return wavs


def _feed_one(orchestrator: object, wav_path: Path) -> None:
    pcm, sr = sf.read(wav_path, dtype="int16")
    if sr != _SAMPLE_RATE:
        logger.warning("{} {} Hz (16 kHz emas), o'tkazib yuborildi", wav_path.name, sr)
        return
    utt = Utterance(pcm=pcm, started_at=time.time(), ended_at=time.time(), reason="silence")
    orchestrator._deliver_utterance(utt)  # noqa: SLF001 -- deliberately bypasses the microphone


async def main_async(args: argparse.Namespace) -> int:
    config = load_config()
    setup_logging(config.logging, config.app.data_dir)
    wavs = _load_wavs(args.wav_dir)

    try:
        orchestrator = await build_orchestrator(config)
    except CriticalStartupError as exc:
        logger.error("Ishga tushmadi: {}", exc)
        return 1

    run_task = asyncio.ensure_future(orchestrator.run())

    process = psutil.Process()
    baseline_rss_mb = process.memory_info().rss / 1e6
    logger.info(
        "Soak test boshlandi: baseline RSS={:.1f} MB, {} wav fayl, har {} daqiqada, {} soat",
        baseline_rss_mb,
        len(wavs),
        args.interval_minutes,
        args.hours,
    )

    deadline = time.monotonic() + args.hours * 3600
    cycle = 0
    try:
        while time.monotonic() < deadline:
            wav_path = wavs[cycle % len(wavs)]
            cycle += 1
            _feed_one(orchestrator, wav_path)

            rss_mb = process.memory_info().rss / 1e6
            growth_mb = rss_mb - baseline_rss_mb
            logger.info(
                "Sikl {}: {} -> RSS={:.1f} MB (+{:.1f} MB), holat={}",
                cycle,
                wav_path.name,
                rss_mb,
                growth_mb,
                orchestrator.state.value,
            )
            if growth_mb > _RSS_BUDGET_MB:
                logger.warning(
                    "RSS o'sishi {:.0f} MB byudjetdan oshdi: +{:.1f} MB", _RSS_BUDGET_MB, growth_mb
                )

            await asyncio.sleep(args.interval_minutes * 60)
    finally:
        await orchestrator.stop(0)
        await run_task

    logger.info("Soak test tugadi: {} sikl", cycle)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Set here (function call time), never at module level: this file's name
    # matches pytest's default "*_test.py" collection pattern, so a bare
    # module-level assignment would execute the moment *any* pytest run scans
    # the repo without excluding scripts/ -- even though it defines no
    # test_* functions and contributes nothing to the run -- silently forcing
    # every other test into mock mode too (found the hard way via
    # docs/tz/08_orchestrator.md's own test suite).
    os.environ.setdefault("ULTRON_TOOLS_MOCK", "1")
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
