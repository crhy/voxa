from __future__ import annotations

import types

import numpy as np

from voxa.audio import AudioCapture, ReferenceRing, mix_chunk
from voxa.config import Settings
from voxa.echo import default_monitor_source


class FakeCanceller:
    def __init__(self, raise_it: bool = False) -> None:
        self.raise_it = raise_it
        self.seen_ref = None

    def process(self, mic, ref):
        if self.raise_it:
            raise RuntimeError("boom")
        self.seen_ref = np.asarray(ref, dtype=np.float32)
        return mic


def _mic_bytes(n: int) -> bytes:
    return (np.arange(n, dtype="<i2")).tobytes()


def test_window_reference_ahead():
    ring = ReferenceRing(100)
    ring.append(np.arange(100, dtype=np.float32))
    got = ring.window(10, 5)
    assert np.array_equal(got, np.arange(10, 15, dtype=np.float32))


def test_window_reference_behind_zero_padded():
    ring = ReferenceRing(100)
    ring.append(np.arange(5, dtype=np.float32))
    got = ring.window(0, 10)
    assert np.array_equal(got, np.array([0, 1, 2, 3, 4, 0, 0, 0, 0, 0], dtype=np.float32))


def test_window_reference_missing_head_zero_padded():
    ring = ReferenceRing(10)
    ring.append(np.arange(100, dtype=np.float32))  # only last 10 kept, oldest global pos 90
    got = ring.window(85, 10)  # positions 85..94; 85..89 gone, 90..94 present
    assert np.array_equal(got, np.array([0, 0, 0, 0, 0, 90, 91, 92, 93, 94], dtype=np.float32))


def test_mix_chunk_alignment_by_counter():
    ring = ReferenceRing(100)
    ring.append(np.arange(100, dtype=np.float32))
    fake = FakeCanceller()
    out = mix_chunk(_mic_bytes(5), ring, 20, fake)
    assert np.array_equal(fake.seen_ref, np.arange(20, 25, dtype=np.float32))
    assert len(out) == 10


def test_mix_chunk_pads_when_reference_missing():
    ring = ReferenceRing(100)
    ring.append(np.arange(3, dtype=np.float32))
    fake = FakeCanceller()
    mix_chunk(_mic_bytes(6), ring, 0, fake)
    assert np.array_equal(fake.seen_ref, np.array([0, 1, 2, 0, 0, 0], dtype=np.float32))


def test_exception_disables_and_returns_raw():
    cap = AudioCapture()
    cap.canceller = FakeCanceller(raise_it=True)
    cap._ref_ring = ReferenceRing(100)
    cap._ref_ring.append(np.arange(50, dtype=np.float32))
    cap._canceller_enabled = True
    raw = _mic_bytes(8)
    out = cap._mix_or_passthrough(raw)
    assert out == raw
    assert cap._canceller_enabled is False


def test_settings_echo_cancel_false_maps_to_off():
    assert Settings(echo_cancel=False).normalized().echo_mode == "off"


def test_settings_invalid_mode_defaults_to_system():
    assert Settings(echo_mode="banana").normalized().echo_mode == "system"


def test_default_monitor_source_with_fake_pactl():
    def runner_ok(cmd, **_kwargs):
        return types.SimpleNamespace(returncode=0, stdout="alsa_output.pci-0_0.HDMI\n")

    def runner_fail(cmd, **_kwargs):
        return types.SimpleNamespace(returncode=1, stdout="")

    assert default_monitor_source(runner=runner_ok) == "alsa_output.pci-0_0.HDMI.monitor"
    assert default_monitor_source(runner=runner_fail) is None
