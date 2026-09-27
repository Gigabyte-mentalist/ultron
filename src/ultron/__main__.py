"""Ultron CLI entry point.

python -m ultron            # full voice pipeline (docs/tz/08_orchestrator.md)
python -m ultron --text     # text debug mode (docs/tz/06_llm.md §3.7)
python -m ultron --text --speak   # also speaks replies through TTS
python -m ultron --debug    # debug-level logging + live status line (§8B.6)
python -m ultron --check    # runs scripts/check_env.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from ultron.config import ConfigError, RootConfig, load_config
from ultron.logging_setup import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ultron", description="Ultron — lokal ovozli AI yordamchi"
    )
    parser.add_argument("--text", action="store_true", help="Matnli debug rejim (mikrofonsiz)")
    parser.add_argument(
        "--speak", action="store_true", help="--text bilan birga: javoblarni TTS orqali aytadi"
    )
    parser.add_argument("--debug", action="store_true", help="Debug darajasidagi log")
    parser.add_argument("--check", action="store_true", help="Muhitni tekshirish (check_env)")
    return parser.parse_args(argv)


async def _run_text_mode(config: RootConfig, *, speak: bool) -> int:
    # Imported lazily so `--check`/skeleton runs don't pay for loading ollama/pydantic
    # tool schemas that plain `python -m ultron` doesn't need.
    from ultron.llm.agent import (
        Agent,
        DoneEvent,
        ErrorEvent,
        SentenceEvent,
        ToolEndEvent,
        ToolStartEvent,
    )
    from ultron.llm.client import OllamaClient
    from ultron.llm.fastpath import FastPath
    from ultron.tools import async_init_all, async_shutdown_all, configure_all, default_registry
    from ultron.tools.memory_tools import get_facts_for_prompt

    configure_all(config)
    await async_init_all()

    client = OllamaClient(config.llm)
    if not await client.health():
        logger.warning(
            "Ollama serverga ulanib bo'lmadi yoki '{}' modeli topilmadi -- "
            "javoblar 'Hozir o'ylay olmayapman' bilan tugaydi, toki server tiklanmaguncha",
            config.llm.model,
        )
    else:
        await client.ensure_loaded()

    fastpath = FastPath(default_registry)
    agent = Agent(
        client,
        default_registry,
        fastpath,
        config.llm,
        city=config.tools.weather.city,
        facts_provider=get_facts_for_prompt,
        fastpath_enabled=config.fastpath.enabled,
    )

    tts_engine = None
    if speak:
        try:
            from ultron.tts.mms_engine import MMSEngine

            tts_engine = MMSEngine(
                model_name=config.tts.model,
                model_script=config.tts.model_script,
                cache_dir=Path(config.app.data_dir) / "models" / "tts",
                speed=config.tts.speed,
                sentence_pause_ms=config.tts.sentence_pause_ms,
            )
            tts_engine.load()
        except Exception as exc:
            logger.warning("--speak so'ralgan, lekin TTS ishga tushmadi: {}", exc)
            tts_engine = None

    logger.info("Ultron matnli rejim. Buyruqlar: /reset /tools /fast on|off /exit")

    try:
        while True:
            try:
                line = await asyncio.to_thread(input, "Siz: ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue

            if line == "/exit":
                break
            if line == "/reset":
                agent.reset_history()
                logger.info("Tarix tozalandi.")
                continue
            if line == "/tools":
                for schema in default_registry.schemas(enabled_only=False):
                    fn = schema["function"]
                    logger.info("  {}: {}", fn["name"], fn["description"])
                continue
            if line in ("/fast on", "/fast off"):
                agent.fastpath_enabled = line.endswith("on")
                logger.info("Fast-path: {}", "yoqildi" if agent.fastpath_enabled else "o'chirildi")
                continue

            pending_tools: list[tuple[str, dict, float]] = []
            try:
                async for event in agent.respond(line):
                    if isinstance(event, SentenceEvent):
                        logger.info("Ultron: {}", event.text)
                        if tts_engine is not None:
                            _speak(tts_engine, event.text)
                    elif isinstance(event, ToolStartEvent):
                        pending_tools.append((event.name, event.args, time.monotonic()))
                    elif isinstance(event, ToolEndEvent):
                        name, args, started = pending_tools.pop(0)
                        duration_ms = int((time.monotonic() - started) * 1000)
                        status = "ok" if event.result.ok else f"xato: {event.result.error}"
                        logger.info(
                            "[tool] {} {} -> {} ({} ms)",
                            name,
                            json.dumps(args, ensure_ascii=False),
                            status,
                            duration_ms,
                        )
                    elif isinstance(event, ErrorEvent):
                        logger.info("Ultron: {}", event.message)
                    elif isinstance(event, DoneEvent):
                        pass
            except Exception:
                logger.exception("Matnli rejimda kutilmagan xato")
                logger.info("Ultron: Kechirasiz, kutilmagan xato yuz berdi.")
    finally:
        # docs/tz/07_tools.md's reminder scheduler + shared SQLite connection
        # (aiosqlite's per-connection background thread) otherwise keep the
        # process alive after this function returns.
        await async_shutdown_all()

    return 0


def _speak(tts_engine: Any, text: str) -> None:
    try:
        import sounddevice as sd

        pcm = tts_engine.synthesize(text)
        if pcm.size:
            sd.play(pcm, tts_engine.sample_rate)
            sd.wait()
    except Exception as exc:
        logger.warning("TTS bilan aytishda xato: {}", exc)


_STARTUP_RETRY_DELAY_S = 30


async def _run_voice_mode(config: RootConfig, *, debug: bool) -> int:
    """docs/tz/08_orchestrator.md §8A.1: a critical component (audio/STT/wake)
    failing to load retries every 30 s rather than exiting; TTS failing is
    non-critical and handled inside ``build_orchestrator`` itself.

    docs/tz/09_autostart.md §3.2 step 1: this is "the main process" the
    supervisor spawns as ``pythonw.exe -m ultron`` -- it holds its own
    ``Local\\UltronMain`` mutex (separate from the supervisor's own
    ``Local\\UltronSupervisor``) so a second copy (e.g. someone running
    ``python -m ultron`` by hand while the supervised instance is already up)
    exits immediately instead of fighting over the microphone/tray icon.
    """
    from ultron.core.orchestrator import CriticalStartupError, build_orchestrator
    from ultron.core.single_instance import SingleInstanceLock

    lock = SingleInstanceLock("UltronMain")
    if not lock.acquire():
        logger.warning("Ultron allaqachon ishlamoqda (asosiy jarayon), chiqilmoqda")
        return 0

    try:
        while True:
            try:
                orchestrator = await build_orchestrator(config, debug=debug)
                break
            except CriticalStartupError as exc:
                logger.error(
                    "Kritik komponent yuklanmadi: {}. {} soniyadan keyin qayta urinamiz.",
                    exc,
                    _STARTUP_RETRY_DELAY_S,
                )
                await asyncio.sleep(_STARTUP_RETRY_DELAY_S)

        return await orchestrator.run()
    finally:
        lock.release()


async def run(args: argparse.Namespace) -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("Config yuklashda xato:\n{}", exc)
        return 1

    if args.debug:
        config.logging.level = "DEBUG"
    setup_logging(config.logging, config.app.data_dir)

    if args.check:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
        from check_env import main as check_env_main

        # ``sys.argv`` still has this process's own "--check" in it, which
        # check_env's unrelated parser (--no-sound only) doesn't recognize --
        # pass an explicit empty argv rather than letting it default to
        # sys.argv[1:].
        return check_env_main([])

    if args.text:
        return await _run_text_mode(config, speak=args.speak)

    return await _run_voice_mode(config, debug=args.debug)


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
