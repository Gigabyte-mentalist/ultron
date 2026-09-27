"""Checks/launches ``ollama serve`` hidden (docs/tz/09_autostart.md §3.2.3).

Shared by two callers: ``ultron.supervisor`` (once, at its own startup,
before launching the main process) and ``core.orchestrator``'s health
monitor (mid-session, when Ollama stops responding -- §3.6/§5's "Ollama'ni
o'chirish -> Ultron tiklanadi ≤ 90 s" acceptance criterion needs Ultron
itself to relaunch it, since a manually-killed ``ollama.exe`` has no
built-in self-restart of its own once its Startup-folder entry has already
run once this session).
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time

import httpx
from loguru import logger

# subprocess.CREATE_NO_WINDOW -- named here too since this module must stay
# importable on non-Windows (tests run in CI without a Windows sandbox).
_CREATE_NO_WINDOW = 0x08000000


async def ollama_reachable(host: str, *, timeout_s: float = 3.0) -> bool:
    """True if ``GET {host}/api/tags`` responds successfully."""
    url = f"{host.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
        return response.status_code == 200
    except httpx.HTTPError as exc:
        logger.debug("Ollama'ga ulanib bo'lmadi ({}): {}", url, exc)
        return False


def launch_ollama_hidden() -> bool:
    """Best-effort: starts ``ollama serve`` with no visible console window.

    Returns ``False`` (without raising) if the ``ollama`` executable isn't on
    PATH or the process couldn't be spawned -- the caller treats this as "we
    tried," not a fatal error.
    """
    exe = shutil.which("ollama") or shutil.which("ollama.exe")
    if exe is None:
        logger.warning("'ollama' PATH'da topilmadi, avtomatik ishga tushirib bo'lmadi")
        return False

    creationflags = _CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        subprocess.Popen(  # noqa: S603 -- fixed argv, no user input reaches it
            [exe, "serve"],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        logger.warning("Ollama'ni ishga tushirib bo'lmadi: {}", exc)
        return False

    logger.info("Ollama fon rejimida ishga tushirildi (yashirin)")
    return True


async def ensure_ollama_running(host: str, *, start_if_missing: bool, wait_s: float = 60.0) -> bool:
    """docs/tz/09_autostart.md §3.2 step 3: check, launch-if-missing, and
    poll up to ``wait_s`` seconds. Used by the supervisor's own bootstrap;
    the orchestrator's health monitor calls the two lower-level functions
    directly instead, since it must never block its 30 s poll loop for up
    to a minute."""
    if await ollama_reachable(host):
        return True

    if not start_if_missing:
        return False

    if not launch_ollama_hidden():
        return False

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if await ollama_reachable(host):
            return True
        await asyncio.sleep(1.0)
    return False
