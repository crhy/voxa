"""The render tap's audible anchor is the device clock, not the 0.05 s fallback (issue #72, v11d).

The whole v11c blocker was that ``GstSinkClock.latency()`` always returned the fallback (the
sandbox's gi has no ``query_latency``), so every reference window was stamped ``wall-clock submit
+ 0.05 s`` - ahead of real playback - and the windows went blank while she spoke. The v11d fix
stamps each buffer with the instant it actually leaves the speakers: ``base_time + running_time``
on the very clock the microphone stamps with. These tests drive ``_tap_callback`` directly with a
faithful fake so the anchor is proven without a sound device.
"""

from __future__ import annotations

import numpy as np

from voxa.timeline import RenderTimeline, FixedSinkClock, _tap_callback

RATE = 16000


class _Structure:
    def has_field(self, name: str) -> bool:
        return name == "channels"

    def get_value(self, name: str) -> int:
        return 1


class _Caps:
    def get_structure(self, _index: int) -> _Structure:
        return _Structure()


class _Buffer:
    def __init__(self, data: bytes, running_time: int) -> None:
        self.data = data
        self._running = running_time

    def map(self, _flags):
        return True, type("Mapped", (), {"data": self.data})()

    def unmap(self, _mapped) -> None:
        return None

    def get_running_time(self) -> int:
        return self._running


class _Sample:
    def __init__(self, data: bytes, running_time: int) -> None:
        self._buffer = _Buffer(data, running_time)

    def get_buffer(self) -> _Buffer:
        return self._buffer

    def get_caps(self) -> _Caps:
        return _Caps()


class _Parent:
    """A stand-in for the pipeline the appsink sits in front of: it answers base_time."""

    def __init__(self, base_time: int) -> None:
        self._base = base_time

    def get_parent(self):
        return None  # the top of the parent chain

    def get_base_time(self) -> int:
        return self._base


class _TapSink:
    def __init__(self, data: bytes, running_time: int, base_time: int) -> None:
        self._sample = _Sample(data, running_time)
        self._parent = _Parent(base_time)

    def emit(self, _signal: str) -> _Sample:
        return self._sample

    def get_parent(self) -> _Parent:
        return self._parent


class _Gst:
    MapFlags = type("MapFlags", (), {"READ": "read"})
    FlowReturn = type("FlowReturn", (), {"OK": "ok"})
    CLOCK_TIME_NONE = -1


def _payload() -> np.ndarray:
    return (np.arange(320, dtype=np.float32) / 1000.0)


def _bytes_of(arr: np.ndarray) -> bytes:
    return arr.astype("<f4").tobytes()


def test_anchor_is_the_buffer_running_time_on_the_device_clock() -> None:
    """A buffer whose running time is valid is stamped base_time + running_time, NOT submit + latency."""
    # clock() returns a wall time far from the device clock, so any fallback would differ.
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 900.0)
    timeline.open("g1", "tts-stream")
    base_ns, running_ns = 1_000_000_000_000, 250_000_000  # base=1000 s, running=0.25 s
    expected = (base_ns + running_ns) / 1_000_000_000.0  # 1000.25 s
    callback = _tap_callback(timeline, "g1", "tts-stream", _Gst, FixedSinkClock(0.05))
    sink = _TapSink(_bytes_of(_payload()), running_ns, base_ns)
    assert callback(sink) == _Gst.FlowReturn.OK
    chunk = timeline._anchors[-1]
    assert chunk.audible_at == expected
    assert abs(chunk.audible_at - 900.05) > 0.1  # it is NOT the wall-clock + fallback anchor


def test_anchor_falls_back_to_the_sink_clock_when_running_time_is_unset() -> None:
    """A pre-roll buffer (running time NONE) keeps the old sink-clock path, so nothing breaks."""
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 500.0)
    timeline.open("g1", "tts-stream")
    callback = _tap_callback(timeline, "g1", "tts-stream", _Gst, FixedSinkClock(0.05))
    sink = _TapSink(_bytes_of(_payload()), _Gst.CLOCK_TIME_NONE, 1_000_000_000_000)
    assert callback(sink) == _Gst.FlowReturn.OK
    chunk = timeline._anchors[-1]
    # submit + sink latency (500.0 + 0.05), chained from _next_playable on the first chunk.
    assert abs(chunk.audible_at - 500.05) < 1e-9


def test_anchor_is_monotonic_across_buffers() -> None:
    """The first chunk takes the device-clock anchor; the rest chain behind it (audible never moves back)."""
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 900.0)
    timeline.open("g1", "tts-stream")
    callback = _tap_callback(timeline, "g1", "tts-stream", _Gst, FixedSinkClock(0.05))
    base = 2_000_000_000_000
    for i in range(4):
        running = 1_000_000_000 + i * (320 * 1_000_000_000 // RATE)
        sink = _TapSink(_bytes_of(_payload()), running, base)
        callback(sink)
    audible = [c.audible_at for c in timeline._anchors]
    assert audible[0] == (base + 1_000_000_000) / 1_000_000_000.0  # first chunk: the device clock
    assert all(a < b for a, b in zip(audible, audible[1:]))         # the sink plays them in order
    assert not any(abs(a - 900.05) < 0.1 for a in audible)          # none anchored to wall + fallback
