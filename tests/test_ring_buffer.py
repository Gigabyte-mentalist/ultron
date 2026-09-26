"""Tests for ultron.audio.ring_buffer (docs/tz/02_audio.md §4)."""

from __future__ import annotations

import numpy as np

from ultron.audio.ring_buffer import RingBuffer


def test_snapshot_before_any_data_is_empty() -> None:
    rb = RingBuffer(capacity_seconds=1.0, sample_rate=16000)

    assert len(rb.snapshot(1.0)) == 0


def test_snapshot_partial_fill_returns_only_available_samples() -> None:
    rb = RingBuffer(capacity_seconds=2.0, sample_rate=16000)
    rb.write(np.arange(8000, dtype=np.int16))  # 0.5 s written

    snap = rb.snapshot(2.0)

    assert len(snap) == 8000


def test_snapshot_length_matches_requested_seconds() -> None:
    rb = RingBuffer(capacity_seconds=2.0, sample_rate=16000)
    rb.write(np.arange(32000, dtype=np.int16))

    snap = rb.snapshot(1.0)

    assert len(snap) == 16000
    np.testing.assert_array_equal(snap, np.arange(16000, 32000, dtype=np.int16))


def test_snapshot_returns_most_recent_samples_in_order_across_wraparound() -> None:
    rb = RingBuffer(capacity_seconds=1.0, sample_rate=100)  # 100-sample capacity
    rb.write(np.arange(0, 60, dtype=np.int16))
    rb.write(np.arange(60, 150, dtype=np.int16))  # wraps past capacity

    snap = rb.snapshot(1.0)

    assert len(snap) == 100
    np.testing.assert_array_equal(snap, np.arange(50, 150, dtype=np.int16))


def test_overflow_in_a_single_write_keeps_only_latest_capacity_worth() -> None:
    rb = RingBuffer(capacity_seconds=0.5, sample_rate=100)  # 50-sample capacity
    rb.write(np.arange(200, dtype=np.int16))

    snap = rb.snapshot(0.5)

    np.testing.assert_array_equal(snap, np.arange(150, 200, dtype=np.int16))


def test_clear_empties_the_buffer() -> None:
    rb = RingBuffer(capacity_seconds=1.0, sample_rate=100)
    rb.write(np.arange(50, dtype=np.int16))

    rb.clear()

    assert len(rb.snapshot(1.0)) == 0
