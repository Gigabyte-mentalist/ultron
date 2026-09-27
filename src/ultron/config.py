"""Pydantic v2 configuration schema and loader for Ultron.

Schema keys mirror docs/tz/00_umumiy_arxitektura.md §5 exactly — key names
must not change without updating that reference first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

DEFAULT_CONFIG_PATH = "config/config.yaml"


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or fails validation."""


class AppConfig(BaseModel):
    name: str = "Ultron"
    language: str = "uz"
    data_dir: Path = Path("C:/ultron/data")
    debug: bool = False


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 512
    input_device: str | None = None
    output_device: str | None = None
    preroll_seconds: float = 2.0
    output_volume: float = Field(default=0.9, ge=0.0, le=1.0)


class VadConfig(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_ms: int = Field(default=250, ge=0)
    end_silence_ms: int = Field(default=700, ge=0)
    max_utterance_s: int = Field(default=15, ge=1)
    listen_timeout_s: int = Field(default=6, ge=1)


class WakeConfig(BaseModel):
    mode: Literal["stt_kws", "oww", "hybrid"] = "stt_kws"
    phrases: list[str] = Field(
        default_factory=lambda: [
            "altron",
            "ultron",
            "oltron",
            "altiron",
            "ultiron",
            "алтрон",
            "ультрон",
            "олтрон",
        ]
    )
    fuzzy_threshold: int = Field(default=82, ge=0, le=100)
    max_prefix_words: int = Field(default=2, ge=1)
    oww_model_path: Path = Path("data/models/wake/altron.onnx")
    oww_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    oww_verify_with_stt: bool = True
    # docs/tz/10_wake_word_training.md §3.4: shaxsiy ovoz verifier (ixtiyoriy --
    # oiladagi boshqa odamlar ham foydalanishi kerak bo'lsa null qoldiriladi).
    oww_verifier_path: Path | None = None
    oww_verifier_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    cooldown_ms: int = Field(default=1200, ge=0)
    barge_in_enabled: bool = True
    barge_in_threshold_boost: float = Field(default=0.15, ge=0.0, le=1.0)


class SttConfig(BaseModel):
    engine: str = "faster_whisper"
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = "uz"
    beam_size: int = Field(default=1, ge=1)
    kws_model: str = "small"
    kws_device: str = "cuda"
    initial_prompt: str = "Altron, brauzerni och. Ovozni balandlat. Telegramni och. Eslatma qo'y."
    no_speech_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    log_prob_threshold: float = -1.0
    output_script: Literal["latin", "cyrillic"] = "latin"


class LlmConfig(BaseModel):
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    think: bool = False
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    num_ctx: int = Field(default=8192, ge=512)
    keep_alive: str = "-1"
    max_tool_iterations: int = Field(default=5, ge=1)
    history_turns: int = Field(default=8, ge=0)
    request_timeout_s: int = Field(default=60, ge=1)


class FastpathConfig(BaseModel):
    enabled: bool = True


class TtsConfig(BaseModel):
    engine: str = "mms"
    model: str = "facebook/mms-tts-uzb-script_cyrillic"
    model_script: Literal["cyrillic", "latin"] = "cyrillic"
    device: str = "cpu"
    speed: float = Field(default=1.0, gt=0.0)
    sentence_pause_ms: int = Field(default=150, ge=0)


class FilesToolConfig(BaseModel):
    allowed_roots: list[Path] = Field(
        default_factory=lambda: [
            Path("~/Desktop"),
            Path("~/Documents"),
            Path("~/Downloads"),
            Path("~/Pictures"),
            Path("~/Music"),
            Path("~/Videos"),
            Path("D:/"),
        ]
    )
    max_results: int = Field(default=10, ge=1)


class BrowserToolConfig(BaseModel):
    default: Literal["chrome", "edge", "default"] = "chrome"
    search_engine: str = "google"


class WeatherConfig(BaseModel):
    city: str = "Tashkent"
    lat: float = 41.2995
    lon: float = 69.2401


class ToolsConfig(BaseModel):
    confirm_dangerous: bool = True
    confirm_timeout_s: int = Field(default=8, ge=1)
    app_aliases_path: Path = Path("config/app_aliases.yaml")
    files: FilesToolConfig = Field(default_factory=FilesToolConfig)
    browser: BrowserToolConfig = Field(default_factory=BrowserToolConfig)
    internet_tools_enabled: bool = True
    weather: WeatherConfig = Field(default_factory=WeatherConfig)


class MemoryConfig(BaseModel):
    db_path: Path = Path("data/db/ultron.sqlite")
    max_facts_in_prompt: int = Field(default=40, ge=0)
    save_utterances: bool = True


class ConversationConfig(BaseModel):
    follow_up_window_s: int = Field(default=8, ge=0)
    end_phrases: list[str] = Field(
        default_factory=lambda: ["rahmat", "bo'ldi", "kerak emas", "yaxshi bo'ldi"]
    )
    cancel_phrases: list[str] = Field(
        default_factory=lambda: ["to'xta", "bekor qil", "jim", "stop"]
    )


class AutostartConfig(BaseModel):
    startup_delay_s: int = Field(default=20, ge=0)
    announce_ready: bool = True
    ollama_start_if_missing: bool = True
    # docs/tz/09_autostart.md §3.4: added by TZ-09, not part of 00-TZ's
    # original schema table -- global wake fallback via pynput, off by
    # default risk (a background key listener) is opt-out, not opt-in,
    # since the TZ describes it as the "ixtiyoriy" wake-word-failed backstop.
    hotkey_enabled: bool = True
    hotkey: str = "<ctrl>+<alt>+u"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "14 days"


class CompanionConfig(BaseModel):
    """Visual companion overlay -- not part of any TZ, added after MVP+TZ-10/11
    at the user's own request: a small always-on-top face widget whose eyes
    glow while Ultron is speaking (``ultron.companion``)."""

    enabled: bool = True
    position: Literal["bottom_right", "bottom_left", "top_right", "top_left"] = "bottom_right"
    size: int = Field(default=220, ge=64, le=600)
    margin: int = Field(default=24, ge=0)


class RootConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    fastpath: FastpathConfig = Field(default_factory=FastpathConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    autostart: AutostartConfig = Field(default_factory=AutostartConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    companion: CompanionConfig = Field(default_factory=CompanionConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` on top of ``base`` (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(value: str, project_root: Path) -> str:
    """Expand ``~`` and make relative paths absolute against ``project_root``."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path)


def _resolve_paths_in_place(data: dict, project_root: Path) -> None:
    """Resolve the handful of path-shaped keys defined in the 00-TZ schema.

    Values like ``wake.oww_model_path`` and ``memory.db_path`` are written in
    the TZ as project-relative (e.g. ``data/models/...``), which already sits
    inside ``app.data_dir`` (``<project_root>/data``). Resolving them against
    ``project_root`` (rather than against ``data_dir`` literally) avoids a
    doubled ``data/data/...`` path while still landing inside data_dir.
    """
    app = data.get("app")
    if isinstance(app, dict) and "data_dir" in app:
        app["data_dir"] = _resolve_path(app["data_dir"], project_root)

    wake = data.get("wake")
    if isinstance(wake, dict) and "oww_model_path" in wake:
        wake["oww_model_path"] = _resolve_path(wake["oww_model_path"], project_root)
    if isinstance(wake, dict) and wake.get("oww_verifier_path") is not None:
        wake["oww_verifier_path"] = _resolve_path(wake["oww_verifier_path"], project_root)

    memory = data.get("memory")
    if isinstance(memory, dict) and "db_path" in memory:
        memory["db_path"] = _resolve_path(memory["db_path"], project_root)

    tools = data.get("tools")
    if isinstance(tools, dict):
        if "app_aliases_path" in tools:
            tools["app_aliases_path"] = _resolve_path(tools["app_aliases_path"], project_root)
        files = tools.get("files")
        if isinstance(files, dict) and "allowed_roots" in files:
            files["allowed_roots"] = [
                _resolve_path(root, project_root) for root in files["allowed_roots"]
            ]


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Config validation failed ({path}):"]
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {loc}: {error['msg']} (got {error.get('input')!r})")
    return "\n".join(lines)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RootConfig:
    """Load, deep-merge, resolve paths, and validate the Ultron config.

    Resolution order:
      1. ``ULTRON_CONFIG`` env var overrides ``path`` if set.
      2. ``config.yaml`` is read.
      3. ``config.local.yaml`` next to it, if present, is deep-merged on top.
      4. Path-shaped keys are resolved to absolute paths.
      5. The result is validated against :class:`RootConfig`.
    """
    config_path = Path(os.environ.get("ULTRON_CONFIG", str(path)))

    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML ({config_path}): {exc}") from exc

    local_path = config_path.parent / "config.local.yaml"
    if local_path.is_file():
        try:
            local_raw = yaml.safe_load(local_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Config file is not valid YAML ({local_path}): {exc}") from exc
        raw = _deep_merge(raw, local_raw)

    project_root = config_path.resolve().parent.parent
    _resolve_paths_in_place(raw, project_root)

    try:
        return RootConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(config_path, exc)) from exc
