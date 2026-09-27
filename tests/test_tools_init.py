"""Tests for ultron.tools.configure_all/async_init_all/async_shutdown_all
(docs/tz/07_tools.md §7D, wired into ``__main__.py``).
"""

from __future__ import annotations

import pytest

from ultron.config import RootConfig
from ultron.memory import db as memory_db
from ultron.tools import (
    async_init_all,
    async_shutdown_all,
    browser_tools,
    configure_all,
    file_tools,
    memory_tools,
    reminder_tools,
    system_tools,
)


@pytest.fixture(autouse=True)
async def _reset_state(tmp_path_factory: pytest.TempPathFactory):
    yield
    await memory_db.reset_for_tests()
    for module in (system_tools, file_tools, browser_tools, memory_tools, reminder_tools):
        module._config = None
    memory_tools._facts_cache.clear()
    reminder_tools._scheduler = None


def test_configure_all_threads_config_into_every_tool_module(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    db_path = tmp_path_factory.mktemp("cfg_db") / "test.sqlite"
    config = RootConfig.model_validate({"memory": {"db_path": str(db_path)}})

    configure_all(config)

    assert system_tools._config is config
    assert file_tools._config is config
    assert browser_tools._config is config
    assert memory_tools._config is config
    assert reminder_tools._config is config
    assert memory_db._db is not None
    assert memory_db._db.db_path == db_path


async def test_async_init_all_loads_facts_and_reschedules_reminders(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    db_path = tmp_path_factory.mktemp("init_db") / "test.sqlite"
    configure_all(RootConfig.model_validate({"memory": {"db_path": str(db_path)}}))
    await memory_tools.remember_fact(key="ism", value="Ali")
    memory_tools._facts_cache.clear()

    await async_init_all()

    assert memory_tools.get_facts_for_prompt() == ["ism: Ali"]
    assert reminder_tools._scheduler is not None


async def test_async_shutdown_all_closes_db_and_stops_scheduler(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    db_path = tmp_path_factory.mktemp("shutdown_db") / "test.sqlite"
    configure_all(RootConfig.model_validate({"memory": {"db_path": str(db_path)}}))
    await async_init_all()
    assert reminder_tools._scheduler is not None

    await async_shutdown_all()

    assert reminder_tools._scheduler is None
    assert memory_db._db is None


async def test_async_shutdown_all_is_safe_when_nothing_was_started() -> None:
    await memory_db.reset_for_tests()
    reminder_tools._scheduler = None

    await async_shutdown_all()  # must not raise
