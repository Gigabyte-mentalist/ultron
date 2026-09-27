"""Tests for ultron.config (docs/tz/01_muhit_sozlash.md §4.9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.config import ConfigError, load_config


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_default_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write(
        config_dir / "config.yaml",
        "app:\n  name: Ultron\n  language: uz\nllm:\n  model: qwen3:8b\n",
    )

    config = load_config(config_dir / "config.yaml")

    assert config.app.name == "Ultron"
    assert config.app.language == "uz"
    assert config.llm.model == "qwen3:8b"
    assert config.wake.mode == "stt_kws"  # untouched section falls back to its default


def test_local_override_merges_on_top(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write(config_dir / "config.yaml", "llm:\n  model: qwen3:8b\n  temperature: 0.3\n")
    _write(config_dir / "config.local.yaml", "llm:\n  temperature: 0.7\n")

    config = load_config(config_dir / "config.yaml")

    assert config.llm.model == "qwen3:8b"
    assert config.llm.temperature == 0.7


def test_invalid_value_raises_config_error_naming_the_key(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write(config_dir / "config.yaml", "wake:\n  fuzzy_threshold: 500\n")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_dir / "config.yaml")

    assert "wake.fuzzy_threshold" in str(exc_info.value)


def test_missing_config_file_raises_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "config" / "config.yaml"

    with pytest.raises(ConfigError):
        load_config(missing)


def test_relative_paths_resolved_against_project_root(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write(config_dir / "config.yaml", "memory:\n  db_path: data/db/ultron.sqlite\n")

    config = load_config(config_dir / "config.yaml")

    assert config.memory.db_path == tmp_path / "data" / "db" / "ultron.sqlite"


def test_oww_verifier_path_resolved_against_project_root_when_set(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write(
        config_dir / "config.yaml",
        "wake:\n  oww_verifier_path: data/models/wake/altron_verifier.pkl\n",
    )

    config = load_config(config_dir / "config.yaml")

    assert config.wake.oww_verifier_path == tmp_path / "data" / "models" / "wake" / (
        "altron_verifier.pkl"
    )


def test_oww_verifier_path_stays_none_by_default(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write(config_dir / "config.yaml", "app:\n  name: Ultron\n")

    config = load_config(config_dir / "config.yaml")

    assert config.wake.oww_verifier_path is None


def test_allowed_roots_expand_user_home(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write(
        config_dir / "config.yaml",
        "tools:\n  files:\n    allowed_roots: ['~/Desktop']\n",
    )

    config = load_config(config_dir / "config.yaml")

    assert config.tools.files.allowed_roots == [Path("~/Desktop").expanduser()]
