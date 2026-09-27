"""TTS CLI debug tool (docs/tz/05_tts.md §3.5).

uv run python scripts/tts_say.py "Salom! Bugun yigirma yettinchi sentyabr, soat 09:30."
uv run python scripts/tts_say.py --file long_text.txt --stream
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sounddevice as sd
from loguru import logger

from ultron.audio.player import AudioPlayer
from ultron.config import ConfigError, RootConfig, load_config
from ultron.tts.mms_engine import MMSEngine
from ultron.tts.preprocess import split_sentences


def _build_engine(config: RootConfig) -> MMSEngine:
    return MMSEngine(
        model_name=config.tts.model,
        model_script=config.tts.model_script,
        cache_dir=Path(config.app.data_dir) / "models" / "tts",
        speed=config.tts.speed,
        sentence_pause_ms=config.tts.sentence_pause_ms,
    )


def cmd_speak_once(config: RootConfig, text: str) -> None:
    """Synthesize the whole text in one call and play it back (no streaming)."""
    engine = _build_engine(config)
    engine.load()
    pcm = engine.synthesize(text)
    if pcm.size == 0:
        logger.info("(bo'sh chiqish -- matn to'liq filtrlandi)")
        return
    sd.play(pcm, engine.sample_rate)
    sd.wait()


async def cmd_speak_stream(config: RootConfig, text: str) -> None:
    """Split into sentences and speak them through the pipelined streaming interface."""
    engine = _build_engine(config)
    engine.load()

    player = AudioPlayer(
        sample_rate=engine.sample_rate,
        output_device=config.audio.output_device,
        output_volume=config.audio.output_volume,
    )
    player.start()
    try:
        sentences = split_sentences(text)
        logger.info("{} ta gapga bo'lindi", len(sentences))

        async def _iter():
            for sentence in sentences:
                yield sentence

        await engine.speak_stream(_iter(), player)

        # speak_stream returns once every sentence has been queued, not once
        # playback has finished -- wait for the queue to actually drain.
        await asyncio.sleep(0.2)
        while player.is_playing:
            await asyncio.sleep(0.05)
    finally:
        player.stop_stream()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ultron TTS debug vositasi")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("text", nargs="?", help="Aytiladigan matn")
    source.add_argument("--file", type=Path, help="Matn fayldan o'qiladi")
    parser.add_argument(
        "--stream", action="store_true", help="Gap-gap streaming interfeysi orqali aytish"
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

    if args.file:
        if not args.file.is_file():
            raise SystemExit(f"Fayl topilmadi: {args.file}")
        text = args.file.read_text(encoding="utf-8")
    else:
        text = args.text

    try:
        if args.stream:
            asyncio.run(cmd_speak_stream(config, text))
        else:
            cmd_speak_once(config, text)
    except KeyboardInterrupt:
        logger.info("To'xtatildi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
