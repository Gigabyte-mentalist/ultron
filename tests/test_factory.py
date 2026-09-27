"""Tests for ultron.wake.create_wake_detector (docs/tz/04_wake_word.md §4, §5).

No GPU/model loading happens here: the ``oww`` fallback path fails at the
missing-file check inside ``OWWDetector.load()`` before ever touching
onnxruntime or openwakeword, and the ``stt_kws`` path is never ``.load()``ed
by the factory itself -- so this stays a fast, non-hardware unit test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ultron.config import RootConfig
from ultron.wake import HybridDetector, create_wake_detector
from ultron.wake.stt_kws import STTKeywordSpotter


def _config(*, mode: str, oww_model_path: Path | None = None) -> RootConfig:
    config = RootConfig()
    config.wake.mode = mode
    if oww_model_path is not None:
        config.wake.oww_model_path = oww_model_path
    return config


def test_stt_kws_mode_returns_stt_keyword_spotter() -> None:
    detector = create_wake_detector(_config(mode="stt_kws"))
    assert isinstance(detector, STTKeywordSpotter)


def test_oww_mode_falls_back_to_stt_kws_when_model_missing(tmp_path: Path) -> None:
    detector = create_wake_detector(
        _config(mode="oww", oww_model_path=tmp_path / "no_such_model.onnx")
    )
    assert isinstance(detector, STTKeywordSpotter)


def test_hybrid_mode_falls_back_to_stt_kws_when_model_missing(tmp_path: Path) -> None:
    detector = create_wake_detector(
        _config(mode="hybrid", oww_model_path=tmp_path / "no_such_model.onnx")
    )
    assert isinstance(detector, STTKeywordSpotter)


def test_hybrid_mode_falls_back_even_without_a_ring_buffer(tmp_path: Path) -> None:
    # ring_buffer is optional; the missing-model check must still short-circuit
    # before it would ever be needed.
    detector = create_wake_detector(
        _config(mode="hybrid", oww_model_path=tmp_path / "missing.onnx"), ring_buffer=None
    )
    assert isinstance(detector, STTKeywordSpotter)
    assert not isinstance(detector, HybridDetector)


def test_unknown_mode_raises_value_error() -> None:
    config = _config(mode="stt_kws")
    config.wake.mode = "bogus"  # bypasses the Literal validator (plain attribute assignment)

    with pytest.raises(ValueError, match="bogus"):
        create_wake_detector(config)
