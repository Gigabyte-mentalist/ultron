"""Tests for the ``@tool`` decorator's JSON-schema generation (docs/tz/06_llm.md §3.2, §4)."""

from __future__ import annotations

from typing import Annotated, Literal

import pytest

from ultron.tools.base import ToolResult, _build_arg_model, _build_schema


async def _open_app(name: Annotated[str, "Application name as the user said it"]) -> ToolResult:
    return ToolResult(ok=True)


async def _set_volume(
    level: Annotated[int, "Target volume percent"] | None = None,
    delta: int | None = None,
) -> ToolResult:
    return ToolResult(ok=True)


async def _set_mode(mode: Literal["chrome", "edge", "default"] = "chrome") -> ToolResult:
    return ToolResult(ok=True)


async def _no_args() -> ToolResult:
    return ToolResult(ok=True)


def test_required_string_arg_with_description() -> None:
    schema = _build_schema(_open_app)

    assert schema == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Application name as the user said it"}
        },
        "required": ["name"],
    }


def test_optional_args_are_not_required() -> None:
    schema = _build_schema(_set_volume)

    assert schema["properties"]["level"]["type"] == "integer"
    assert schema["properties"]["level"]["description"] == "Target volume percent"
    assert schema["properties"]["delta"]["type"] == "integer"
    assert "required" not in schema


def test_literal_becomes_enum() -> None:
    schema = _build_schema(_set_mode)

    assert schema["properties"]["mode"]["type"] == "string"
    assert schema["properties"]["mode"]["enum"] == ["chrome", "edge", "default"]
    assert schema["properties"]["mode"]["default"] == "chrome"
    assert "required" not in schema


def test_no_args_schema_has_empty_properties() -> None:
    schema = _build_schema(_no_args)

    assert schema == {"type": "object", "properties": {}}


def test_arg_model_validates_required_field() -> None:
    model = _build_arg_model(_open_app)

    validated = model.model_validate({"name": "telegram"})
    assert validated.model_dump() == {"name": "telegram"}

    with pytest.raises(Exception, match="name"):
        model.model_validate({})


def test_arg_model_applies_defaults_and_literal_enum() -> None:
    model = _build_arg_model(_set_mode)

    assert model.model_validate({}).model_dump() == {"mode": "chrome"}
    assert model.model_validate({"mode": "edge"}).model_dump() == {"mode": "edge"}
    with pytest.raises(Exception, match="mode"):
        model.model_validate({"mode": "firefox"})


def test_arg_model_optional_field_defaults_to_none() -> None:
    model = _build_arg_model(_set_volume)

    validated = model.model_validate({})
    assert validated.model_dump() == {"level": None, "delta": None}
