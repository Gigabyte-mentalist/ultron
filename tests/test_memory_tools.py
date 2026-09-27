"""Tests for ultron.tools.memory_tools (docs/tz/07_tools.md §7D.2)."""

from __future__ import annotations

import pytest

from ultron.config import RootConfig
from ultron.memory import db as memory_db
from ultron.tools import memory_tools


@pytest.fixture(autouse=True)
async def _reset_state(tmp_path_factory: pytest.TempPathFactory) -> None:
    db_path = tmp_path_factory.mktemp("memtools_db") / "test.sqlite"
    memory_db.configure(db_path)
    memory_tools._facts_cache.clear()
    yield
    await memory_db.reset_for_tests()
    memory_tools._config = None
    memory_tools._facts_cache.clear()


async def test_remember_then_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)

    result = await memory_tools.remember_fact(key="tug'ilgan kun", value="15-mart")
    assert result.ok is True

    recall = await memory_tools.recall_facts()
    assert recall.ok is True
    assert recall.data["facts"][0] == {
        "key": "tug'ilgan kun",
        "value": "15-mart",
        "category": "general",
    }


async def test_remember_fact_updates_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)

    await memory_tools.remember_fact(key="ism", value="Ali")

    assert memory_tools.get_facts_for_prompt() == ["ism: Ali"]


async def test_remember_fact_mock_mode_does_not_touch_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await memory_tools.remember_fact(key="ism", value="Ali")

    assert result.ok is True
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    recall = await memory_tools.recall_facts()
    assert recall.data["facts"] == []


async def test_remember_fact_rejects_empty_key_or_value() -> None:
    assert (await memory_tools.remember_fact(key="  ", value="x")).ok is False
    assert (await memory_tools.remember_fact(key="x", value="  ")).ok is False


async def test_remember_fact_overwrites_existing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    await memory_tools.remember_fact(key="ism", value="Ali")
    await memory_tools.remember_fact(key="ism", value="Vali")

    recall = await memory_tools.recall_facts()

    assert len(recall.data["facts"]) == 1
    assert recall.data["facts"][0]["value"] == "Vali"
    assert memory_tools.get_facts_for_prompt() == ["ism: Vali"]


async def test_recall_facts_filters_by_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    await memory_tools.remember_fact(key="ism", value="Ali")
    await memory_tools.remember_fact(key="shahar", value="Toshkent")

    result = await memory_tools.recall_facts(query="tosh")

    assert len(result.data["facts"]) == 1
    assert result.data["facts"][0]["key"] == "shahar"


async def test_recall_facts_empty() -> None:
    result = await memory_tools.recall_facts()

    assert result.ok is True
    assert result.data["facts"] == []


async def test_forget_fact_removes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    await memory_tools.remember_fact(key="ism", value="Ali")

    result = await memory_tools.forget_fact(key="ism")

    assert result.ok is True
    assert memory_tools.get_facts_for_prompt() == []
    recall = await memory_tools.recall_facts()
    assert recall.data["facts"] == []


async def test_forget_fact_not_found() -> None:
    result = await memory_tools.forget_fact(key="mavjud emas")

    assert result.ok is False


async def test_forget_fact_mock_mode_does_not_touch_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    await memory_tools.remember_fact(key="ism", value="Ali")
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await memory_tools.forget_fact(key="ism")

    assert result.ok is True
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    recall = await memory_tools.recall_facts()
    assert len(recall.data["facts"]) == 1  # untouched in DB


async def test_load_cache_populates_from_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    await memory_tools.remember_fact(key="ism", value="Ali")
    memory_tools._facts_cache.clear()

    await memory_tools.load_cache()

    assert memory_tools.get_facts_for_prompt() == ["ism: Ali"]


async def test_get_facts_for_prompt_respects_max_facts_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    memory_tools.configure(RootConfig.model_validate({"memory": {"max_facts_in_prompt": 2}}))
    await memory_tools.remember_fact(key="a", value="1")
    await memory_tools.remember_fact(key="b", value="2")
    await memory_tools.remember_fact(key="c", value="3")

    assert memory_tools.get_facts_for_prompt() == ["b: 2", "c: 3"]


def test_get_facts_for_prompt_zero_disables() -> None:
    memory_tools.configure(RootConfig.model_validate({"memory": {"max_facts_in_prompt": 0}}))
    memory_tools._facts_cache["x"] = "y"

    assert memory_tools.get_facts_for_prompt() == []


def test_get_facts_for_prompt_default_without_configure() -> None:
    memory_tools._facts_cache["ism"] = "Ali"

    assert memory_tools.get_facts_for_prompt() == ["ism: Ali"]
