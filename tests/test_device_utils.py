"""Tests for ultron.audio._device_utils.refresh_device_list's rate limit.

``AudioCapture`` and ``AudioPlayer`` each call this independently from their
own supervisor thread on every (re)connect attempt. Without a rate limit,
two concurrent streams calling it back-to-back tear down and rebuild
PortAudio's whole process-wide session on top of each other, invalidating
whichever stream happens to be open at the time -- a real, reproduced bug
(see the module docstring in ``_device_utils.py``), not a hypothetical one.
"""

from __future__ import annotations

import ultron.audio._device_utils as device_utils


def test_refresh_is_rate_limited(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(device_utils.sd, "_terminate", lambda: calls.append("terminate"))
    monkeypatch.setattr(device_utils.sd, "_initialize", lambda: calls.append("initialize"))
    monkeypatch.setattr(device_utils, "_last_refresh_at", 0.0)

    device_utils.refresh_device_list()
    device_utils.refresh_device_list()
    device_utils.refresh_device_list()

    assert calls == ["terminate", "initialize"]


def test_refresh_runs_again_after_the_interval_elapses(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(device_utils.sd, "_terminate", lambda: calls.append("terminate"))
    monkeypatch.setattr(device_utils.sd, "_initialize", lambda: calls.append("initialize"))
    monkeypatch.setattr(device_utils, "_last_refresh_at", 0.0)

    fake_now = [1000.0]
    monkeypatch.setattr(device_utils.time, "monotonic", lambda: fake_now[0])

    device_utils.refresh_device_list()
    assert calls == ["terminate", "initialize"]

    fake_now[0] += device_utils._MIN_REFRESH_INTERVAL_S + 0.1
    device_utils.refresh_device_list()
    assert calls == ["terminate", "initialize", "terminate", "initialize"]


def test_refresh_swallows_underlying_errors(monkeypatch) -> None:
    def _boom() -> None:
        raise RuntimeError("PortAudio exploded")

    monkeypatch.setattr(device_utils.sd, "_terminate", _boom)
    monkeypatch.setattr(device_utils, "_last_refresh_at", 0.0)

    device_utils.refresh_device_list()  # must not raise
