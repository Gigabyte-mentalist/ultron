"""Auto-imports every ``*_tools.py`` module so its ``@tool``-decorated
functions register themselves in ``default_registry`` (docs/tz/06_llm.md §3.2).
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from ultron.tools.base import (
    ConfirmationProvider,
    ConsoleConfirmationProvider,
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    default_registry,
    tool,
)

if TYPE_CHECKING:
    from ultron.config import RootConfig

__all__ = [
    "ConfirmationProvider",
    "ConsoleConfirmationProvider",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
    "tool",
    "configure_all",
    "async_init_all",
    "async_shutdown_all",
]

for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name.endswith("_tools"):
        importlib.import_module(f"{__name__}.{_module_info.name}")


def configure_all(config: RootConfig) -> None:
    """Thread ``config`` into every tool module whose tool bodies need values
    beyond their LLM-facing signature (allowed_roots, weather city/coords,
    the memory DB path, ...). Call once at startup, before any tool runs
    (docs/tz/07_tools.md)."""
    from ultron.memory import db as memory_db
    from ultron.tools import browser_tools, file_tools, memory_tools, reminder_tools, system_tools

    system_tools.configure(config)
    file_tools.configure(config)
    browser_tools.configure(config)
    memory_db.configure(config.memory.db_path)
    memory_tools.configure(config)
    reminder_tools.configure(config)


async def async_init_all() -> None:
    """The async half of startup init, run once after :func:`configure_all`:
    loads the facts cache and reschedules pending reminders/timers from
    SQLite (docs/tz/07_tools.md §7D)."""
    from ultron.tools import memory_tools, reminder_tools

    await memory_tools.load_cache()
    await reminder_tools.reschedule_pending()


async def async_shutdown_all() -> None:
    """Counterpart to :func:`async_init_all`: release resources opened during
    startup (the reminder scheduler, the shared SQLite connection) so the
    process can exit cleanly. Call once when shutting down -- without this,
    ``aiosqlite``'s per-connection background thread keeps the process alive
    (caught via manual ``--text`` smoke testing: sessions that ever called
    ``set_reminder``/``set_timer`` never exited on their own)."""
    from ultron.memory import db as memory_db
    from ultron.tools import reminder_tools

    reminder_tools.shutdown()
    await memory_db.close_db()
