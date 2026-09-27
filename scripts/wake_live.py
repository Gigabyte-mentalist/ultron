"""Live wake-word demo (docs/tz/04_wake_word.md §3.6).

    uv run python scripts/wake_live.py

Prints time/score/source/trailing_text on every trigger and plays the
"wake" earcon. Uses ``wake.mode`` from config; falls back to ``stt_kws`` if
``oww``/``hybrid`` was requested but no trained model is available yet.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from ultron.audio.capture import AudioCapture
from ultron.audio.earcons import generate as generate_earcon
from ultron.audio.player import AudioPlayer
from ultron.audio.ring_buffer import RingBuffer
from ultron.audio.vad import Endpointer, SileroVAD, default_model_path
from ultron.config import ConfigError, RootConfig, load_config
from ultron.events import Utterance, WakeEvent
from ultron.wake import create_wake_detector


def _vad_model_path(config: RootConfig) -> Path:
    path = default_model_path(config.app.data_dir)
    if not path.is_file():
        raise SystemExit(
            f"Silero VAD modeli topilmadi: {path}\n"
            "Avval ishga tushiring: uv run python scripts/download_models.py --vad"
        )
    return path


def _announce(event: WakeEvent, player: AudioPlayer, sample_rate: int) -> None:
    logger.info(
        "WAKE! score={:.2f} manba={} trailing_text={!r}",
        event.score,
        event.source,
        event.trailing_text,
    )
    player.enqueue(generate_earcon("wake", sample_rate), sample_rate)


async def run(config: RootConfig) -> None:
    ring = RingBuffer(config.audio.preroll_seconds, config.audio.sample_rate)
    capture = AudioCapture(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        block_size=config.audio.block_size,
        input_device=config.audio.input_device,
    )
    vad = SileroVAD(_vad_model_path(config))
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

    detector = create_wake_detector(config, ring_buffer=ring)
    logger.info("Wake detektor yuklanmoqda (mode={})...", config.wake.mode)
    detector.load()

    player = AudioPlayer(
        sample_rate=config.audio.sample_rate,
        output_device=config.audio.output_device,
        output_volume=config.audio.output_volume,
    )
    player.start()

    logger.info(
        'Jonli wake-word rejimi boshlandi ("Altron" deb gapiring). To\'xtatish uchun Ctrl+C.'
    )
    try:
        async for frame in capture.frames():
            ring.write(frame.pcm)
            prob = vad.process(frame.pcm)

            frame_event = detector.process_frame(frame)
            if frame_event is not None:
                _announce(frame_event, player, config.audio.sample_rate)

            result = endpointer.push(frame.pcm, prob)
            if isinstance(result, Utterance):
                utt_event = await detector.process_utterance(result)
                if utt_event is not None:
                    _announce(utt_event, player, config.audio.sample_rate)
    finally:
        capture.stop()
        player.stop_stream()


def main(argv: list[str] | None = None) -> int:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")

    try:
        config = load_config()
    except ConfigError as exc:
        logger.info("Config yuklab bo'lmadi:\n{}", exc)
        return 1

    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        logger.info("To'xtatildi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
