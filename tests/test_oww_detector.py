"""Tests for ultron.wake.oww_detector.OWWDetector.load() (docs/tz/04_wake_word.md
§3.3, docs/tz/10_wake_word_training.md §3.4 custom verifier wiring).

``openwakeword.model.Model`` itself is monkeypatched -- these tests only check
*which kwargs* ``OWWDetector.load()`` passes to it, not real inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ultron.wake.oww_detector import OWWDetector


def _make_feature_files(tmp_path: Path) -> Path:
    model_path = tmp_path / "altron.onnx"
    model_path.write_bytes(b"")
    (tmp_path / "melspectrogram.onnx").write_bytes(b"")
    (tmp_path / "embedding_model.onnx").write_bytes(b"")
    return model_path


class _RecordingModel:
    """Stands in for ``openwakeword.model.Model``, recording its kwargs."""

    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        _RecordingModel.last_kwargs = kwargs
        self.prediction_buffer: dict[str, Any] = {}


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import openwakeword.model as model_module

    monkeypatch.setattr(model_module, "Model", _RecordingModel)
    _RecordingModel.last_kwargs = None


def test_load_without_verifier_path_omits_verifier_kwargs(tmp_path: Path) -> None:
    model_path = _make_feature_files(tmp_path)
    detector = OWWDetector(
        model_path=model_path, threshold=0.5, cooldown_ms=1000, verify_with_stt=True
    )

    detector.load()

    assert "custom_verifier_models" not in _RecordingModel.last_kwargs
    assert "custom_verifier_threshold" not in _RecordingModel.last_kwargs


def test_load_with_missing_verifier_file_omits_verifier_kwargs(tmp_path: Path) -> None:
    model_path = _make_feature_files(tmp_path)
    detector = OWWDetector(
        model_path=model_path,
        threshold=0.5,
        cooldown_ms=1000,
        verify_with_stt=True,
        verifier_path=tmp_path / "no_such_verifier.pkl",
    )

    detector.load()

    assert "custom_verifier_models" not in _RecordingModel.last_kwargs


def test_load_with_existing_verifier_file_passes_it_through(tmp_path: Path) -> None:
    model_path = _make_feature_files(tmp_path)
    verifier_path = tmp_path / "altron_verifier.pkl"
    verifier_path.write_bytes(b"")
    detector = OWWDetector(
        model_path=model_path,
        threshold=0.5,
        cooldown_ms=1000,
        verify_with_stt=True,
        verifier_path=verifier_path,
        verifier_threshold=0.42,
    )

    detector.load()

    assert _RecordingModel.last_kwargs["custom_verifier_models"] == {"altron": str(verifier_path)}
    assert _RecordingModel.last_kwargs["custom_verifier_threshold"] == 0.42
