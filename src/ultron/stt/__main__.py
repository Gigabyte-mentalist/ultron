"""STT CLI debug tool (docs/tz/03_stt.md §3.6).

uv run python -m ultron.stt --file some.wav     # transcribe one file
uv run python -m ultron.stt --live              # VAD + STT live, prints each utterance
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import soundfile as sf
from loguru import logger

from ultron.audio.capture import AudioCapture
from ultron.audio.ring_buffer import RingBuffer
from ultron.audio.vad import Endpointer, SileroVAD, default_model_path
from ultron.config import ConfigError, RootConfig, load_config
from ultron.events import Utterance
from ultron.stt.whisper_engine import WhisperEngine

_SAMPLE_RATE = 16000


def _build_engine(config: RootConfig) -> WhisperEngine:
    return WhisperEngine(
        model=config.stt.model,
        device=config.stt.device,
        compute_type=config.stt.compute_type,
        language=config.stt.language,
        beam_size=config.stt.beam_size,
        no_speech_threshold=config.stt.no_speech_threshold,
        log_prob_threshold=config.stt.log_prob_threshold,
        initial_prompt=config.stt.initial_prompt,
        download_root=Path(config.app.data_dir) / "models" / "stt",
    )


def cmd_file(config: RootConfig, path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Fayl topilmadi: {path}")
    pcm, sr = sf.read(path, dtype="int16")
    if sr != _SAMPLE_RATE:
        raise SystemExit(f"Kutilgan {_SAMPLE_RATE} Hz, faylda {sr} Hz: {path}")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]

    engine = _build_engine(config)
    engine.load()
    transcript = engine.transcribe(pcm)
    logger.info("Matn: {}", transcript.text)
    logger.info(
        "raw={!r} til={} ishonch={:.2f} davomiylik={:.2f}s kechikish={}ms",
        transcript.raw_text,
        transcript.language,
        transcript.confidence,
        transcript.duration_s,
        transcript.latency_ms,
    )


async def cmd_live(config: RootConfig) -> None:
    vad_model_path = default_model_path(config.app.data_dir)
    if not vad_model_path.is_file():
        raise SystemExit(
            f"Silero VAD modeli topilmadi: {vad_model_path}\n"
            "Avval ishga tushiring: uv run python scripts/download_models.py --vad"
        )

    capture = AudioCapture(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        block_size=config.audio.block_size,
        input_device=config.audio.input_device,
    )
    ring = RingBuffer(config.audio.preroll_seconds, config.audio.sample_rate)
    vad = SileroVAD(vad_model_path)
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

    engine = _build_engine(config)
    engine.load()

    logger.info("Jonli STT boshlandi. To'xtatish uchun Ctrl+C.")
    async for frame in capture.frames():
        ring.write(frame.pcm)
        prob = vad.process(frame.pcm)
        result = endpointer.push(frame.pcm, prob)
        if isinstance(result, Utterance):
            transcript = await asyncio.to_thread(engine.transcribe, result.pcm)
            if transcript.text:
                logger.info("> {}", transcript.text)
            else:
                logger.debug("(bo'sh/filtrlangan, raw={!r})", transcript.raw_text)
    capture.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron STT debug vositasi")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Bitta wav faylni transkripsiya qilish")
    group.add_argument("--live", action="store_true", help="VAD + STT jonli rejim")
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
        if args.file:
            cmd_file(config, args.file)
        elif args.live:
            asyncio.run(cmd_live(config))
    except KeyboardInterrupt:
        logger.info("To'xtatildi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
