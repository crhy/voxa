"""Pure tests for the app's reference capture: alignment and idle-monitor handling."""

from __future__ import annotations

import numpy as np

from voxa.audio import AudioCapture, ReferenceRing, mix_chunk


class FakeCanceller:
    def __init__(self) -> None:
        self.seen_ref = None

    def process(self, mic, ref):
        self.seen_ref = np.asarray(ref, dtype=np.float32)
        return mic


def _mic_bytes(n: int) -> bytes:
    return (np.arange(n, dtype="<i2")).tobytes()


def test_mix_chunk_ref_offset_shifts_window():
    ring = ReferenceRing(100)
    ring.append(np.arange(100, dtype=np.float32))
    fake = FakeCanceller()
    mix_chunk(_mic_bytes(5), ring, 20, fake, 10)
    assert np.array_equal(fake.seen_ref, np.arange(10, 15, dtype=np.float32))


def test_mix_chunk_bypasses_idle_monitor():
    ring = ReferenceRing(100)
    fake = FakeCanceller()
    raw = _mic_bytes(6)
    out = mix_chunk(raw, ring, 0, fake)
    assert out == raw
    assert fake.seen_ref is None


def test_mix_chunk_bypasses_zero_padded_reference():
    ring = ReferenceRing(100)
    ring.append(np.zeros(50, dtype=np.float32))
    fake = FakeCanceller()
    raw = _mic_bytes(6)
    out = mix_chunk(raw, ring, 10, fake)
    assert out == raw
    assert fake.seen_ref is None


def test_first_reference_sample_anchors_to_mic_counter():
    ring = ReferenceRing(100)
    ring.append(np.arange(100, dtype=np.float32))
    anchor = 400 - ring.total
    fake = FakeCanceller()
    mix_chunk(_mic_bytes(5), ring, 395, fake, anchor)
    assert np.array_equal(fake.seen_ref, np.arange(95, 100, dtype=np.float32))


def test_mix_or_passthrough_uses_anchor_offset():
    cap = AudioCapture()
    fake = FakeCanceller()
    cap.canceller = fake
    cap._ref_ring = ReferenceRing(100)
    cap._ref_ring.append(np.arange(100, dtype=np.float32))
    cap._mic_counter = 30
    cap._ref_anchor = 20
    cap._mix_or_passthrough(_mic_bytes(5))
    assert np.array_equal(fake.seen_ref, np.arange(10, 15, dtype=np.float32))


def test_idle_monitor_passthrough_keeps_counter_moving():
    cap = AudioCapture()
    cap.canceller = FakeCanceller()
    cap._ref_ring = ReferenceRing(100)
    raw = _mic_bytes(8)
    out = cap._mix_or_passthrough(raw)
    assert out == raw
    assert cap._mic_counter == 8
