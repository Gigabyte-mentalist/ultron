"""Ultron CLI entry point.

python -m ultron            # skeleton run: loads config + logging
python -m ultron --text     # text debug mode (placeholder, filled in later stages)
python -m ultron --debug    # debug-level logging
python -m ultron --check    # runs scripts/check_env.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger

from ultron.config import ConfigError, load_config
from ultron.logging_setup import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ultron", description="Ultron — lokal ovozli AI yordamchi"
    )
    parser.add_argument("--text", action="store_true", help="Matnli debug rejim (mikrofonsiz)")
    parser.add_argument("--debug", action="store_true", help="Debug darajasidagi log")
    parser.add_argument("--check", action="store_true", help="Muhitni tekshirish (check_env)")
    return parser.parse_args(argv)


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

        return check_env_main()

    if args.text:
        logger.info("Matnli rejim uchun joy tayyor — LLM sikli keyingi bosqichlarda qo'shiladi")
        return 0

    logger.info("Ultron skeleti ishlayapti")
    return 0


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
