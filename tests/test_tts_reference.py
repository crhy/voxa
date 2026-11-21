"""Roadmap task 1: the render timeline that is Voxa's exact reference for her own voice.

Everything here runs without a sound card and without GStreamer: the sink's latency and the source's
latency are fakes that can be changed mid-stream, because the whole point of the timeline is to keep
the map right when those numbers move.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxa.audio import AudioCapture, clean_chunk, mix_chunk
from voxa.timeline import (
    CaptureClock,
    FixedSinkClock,
    GstSinkClock,
    RenderTimeline,
    attach_render_tap,
)

RATE = 16000


class FakeSink:
    """A sink whose reported latency the test controls, second by second."""

    def __init__(self, latency: float = 0.04) -> None:
        self.value = latency
        self.calls = 0

    def latency(self) -> float:
        self.calls += 1
        return self.value


def ramp(samples: int, *, start: float = 0.0) -> np.ndarray:
    """A ramp that identifies each sample by its value, so a window can be checked exactly."""
    return (np.arange(samples, dtype=np.float32) + start) / 1000.0


# --------------------------------------------------------------------------- the map

def test_a_submitted_chunk_is_mapped_to_when_the_sink_will_play_it() -> None:
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    chunk = timeline.submit(ramp(320), generation="g1", path="tts-file",
                            sink=FakeSink(0.04), now=100.0)
    assert chunk is not None
    assert chunk.first_index == 0
    assert chunk.end_index == 320
    # Written at 100.0, the sink says 40 ms: audible from 100.04.
    assert timeline.audible_time(0) == pytest.approx(100.04)
    assert timeline.audible_time(160) == pytest.approx(100.04 + 160 / RATE)
    # And the map runs backwards too: what is audible at a moment.
    assert timeline.index_at_audible_time(100.04) == 0
    assert timeline.index_at_audible_time(100.04 + 0.02) == 320


def test_a_sink_that_never_answers_does_not_claim_the_sound_is_instant() -> None:
    """A failed latency query must not become a 0 latency: that is the bug the tap exists to fix."""
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    timeline.submit(ramp(320), generation="g1", path="tts-file", sink=None, now=100.0)
    timeline.submit(ramp(320, start=320), generation="g1", path="tts-file", sink=None, now=100.02)
    # With no answer at all the chunks chain from one another instead of collapsing onto "now".
    assert timeline.audible_time(0) == pytest.approx(100.0)
    assert timeline.audible_time(320) == pytest.approx(100.02)


def test_a_latency_that_grows_mid_stream_moves_the_map_with_it() -> None:
    """A Bluetooth profile change lengthens the sink's queue; the reference has to follow."""
    sink = FakeSink(0.04)
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 500.0)
    timeline.open("g1", "tts-file")
    timeline.submit(ramp(320), generation="g1", path="tts-file", sink=sink, now=500.0)
    sink.value = 0.18  # the profile changed: everything from here is heard 180 ms late
    timeline.submit(ramp(320, start=320), generation="g1", path="tts-file", sink=sink, now=500.02)
    assert timeline.audible_time(0) == pytest.approx(500.04)
    assert timeline.audible_time(320) == pytest.approx(500.20)
    # The map is piecewise: 10 ms into the old latency is still the old chunk, 10 ms into the new
    # latency is the new one, and the two never overlap.
    assert timeline.index_at_audible_time(500.05) == 160
    assert timeline.index_at_audible_time(500.21) == 480


def test_a_sample_is_never_mapped_before_the_one_submitted_before_it() -> None:
    """A sink that suddenly reports a tiny latency cannot make later audio overtake earlier audio."""
    sink = FakeSink(0.2)
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 10.0)
    timeline.open("g1", "tts-file")
    timeline.submit(ramp(320), generation="g1", path="tts-file", sink=sink, now=10.0)
    sink.value = 0.001
    timeline.submit(ramp(320, start=320), generation="g1", path="tts-file", sink=sink, now=10.02)
    assert timeline.audible_time(320) >= timeline.audible_time(0) + 320 / RATE


def test_the_gstreamer_sink_clock_reads_nanoseconds_and_survives_a_pipeline_that_will_not_answer() -> None:
    class Pipeline:
        def __init__(self, answer) -> None:
            self.answer = answer

        def query_latency(self):
            if self.answer == "fail":
                raise RuntimeError("pipeline is going to PAUSED")
            if self.answer == "no":
                return (False, 0, 0, 0)
            return (True, 45_000_000, 60_000_000, 0)

    assert GstSinkClock(Pipeline("yes"), RATE).latency() == pytest.approx(0.045)
    assert GstSinkClock(Pipeline("no"), RATE, fallback=0.07).latency() == pytest.approx(0.07)
    assert GstSinkClock(Pipeline("fail"), RATE, fallback=0.07).latency() == pytest.approx(0.07)
    # Samples still queued in front of the sink are part of the answer.
    clock = GstSinkClock(Pipeline("yes"), RATE, queued=lambda: 1600)
    assert clock.latency() == pytest.approx(0.045 + 0.1)


# --------------------------------------------------------------------------- generations

def test_each_generation_keeps_its_own_counts() -> None:
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("reply-1", "tts-file")
    timeline.open("reply-2", "tts-stream")
    timeline.submit(ramp(320), generation="reply-1", path="tts-file", sink=FakeSink(0.04))
    timeline.note_generated("reply-1", 1000)
    timeline.submit(ramp(320), generation="reply-2", path="tts-stream", sink=FakeSink(0.04))
    first, second = timeline.generation("reply-1"), timeline.generation("reply-2")
    assert (first.submitted, first.generated) == (320, 1000)
    assert second.submitted == 320
    assert first.path == "tts-file" and second.path == "tts-stream"
    # The output stream is one continuous map, not one per generation.
    assert timeline.generation("reply-2").first_index == 320


def test_cancelling_a_generation_keeps_what_the_sink_had_already_queued() -> None:
    """Stopping a reply does not un-ring the bell: the queued tail is still going to be heard."""
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    for index in range(4):
        timeline.submit(ramp(320, start=index * 320), generation="g1", path="tts-file",
                        sink=FakeSink(0.04), now=100.0 + index * 0.02)
    # The play position is inside the second chunk: the first 640 samples are heard, the rest are not.
    item = timeline.cancel("g1", now=100.08)
    assert item.state == "cancelled"
    assert item.audible == 640
    assert item.cancelled == 1280 - 640
    assert timeline.finish("g1") is None or timeline.generation("g1").state == "cancelled"


def test_a_foreign_generation_never_enters_the_reference() -> None:
    """The test voice speaks through the same service; it is a human, not Voxa's echo (handoff 8.2)."""
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.mark_foreign("test-voice", "VoxaTest speaks the user")
    assert timeline.submit(ramp(320), generation="test-voice", path="tts-file",
                           sink=FakeSink(0.04)) is None
    assert timeline.total == 0
    window = timeline.reference_window(mic_start=100.04, length=320)
    assert not window.any()


# --------------------------------------------------------------------------- the window the canceller gets

def test_the_window_is_the_audio_that_was_audible_when_the_microphone_sampled() -> None:
    """The capture latency is subtracted, or every window is served 20 ms off."""
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    for index in range(20):  # 0.4 s of render: the window asked for below is inside it
        timeline.submit(ramp(320, start=index * 320), generation="g1", path="tts-file",
                        sink=FakeSink(0.04), now=100.0 + index * 0.02)
    capture = CaptureClock(rate=RATE, latency=0.02)
    # A 20 ms buffer delivered at 100.30 was sampled between 100.26 and 100.28: the driver sat on it
    # for 20 ms, and the 320 samples span another 20 ms.
    mic_start, mic_end = capture.stamp(100.30, 320)
    assert mic_start == pytest.approx(100.26)
    assert mic_end == pytest.approx(100.28)
    # The sink plays sample j at 100.04 + j/16000, so at 100.26 it was playing sample 3520.
    window = timeline.reference_window(mic_start=mic_start, length=320)
    assert np.allclose(window, ramp(320, start=3520))


def test_the_window_follows_a_source_latency_that_changes() -> None:
    """A driver whose latency moves (a headset switching profile) must not freeze the alignment."""
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    for index in range(20):
        timeline.submit(ramp(320, start=index * 320), generation="g1", path="tts-file",
                        sink=FakeSink(0.04), now=100.0 + index * 0.02)
    latency = {"value": 0.02}
    capture = CaptureClock(rate=RATE, latency=lambda: latency["value"])
    first = timeline.reference_window(mic_start=capture.stamp(100.30, 320)[0], length=320)
    latency["value"] = 0.06  # the driver is now four tens of a millisecond later
    second = timeline.reference_window(mic_start=capture.stamp(100.30, 320)[0], length=320)
    # The same delivery moment now means an earlier sampling moment, so an earlier render window.
    assert np.allclose(first, ramp(320, start=3520))
    assert np.allclose(second, ramp(320, start=3520 - 640))


def test_the_echo_travel_shifts_the_window_to_what_the_room_is_answering() -> None:
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    for index in range(20):
        timeline.submit(ramp(320, start=index * 320), generation="g1", path="tts-file",
                        sink=FakeSink(0.04), now=100.0 + index * 0.02)
    plain = timeline.reference_window(mic_start=100.28, length=320)
    travelled = timeline.reference_window(mic_start=100.28, length=320, echo_travel=0.05)
    assert np.allclose(plain, ramp(320, start=3840))
    assert np.allclose(travelled, ramp(320, start=3840 - 800))


def test_a_window_before_the_first_sample_is_silence_not_the_first_sample() -> None:
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    timeline.submit(ramp(320), generation="g1", path="tts-file", sink=FakeSink(0.04), now=100.0)
    assert not timeline.reference_window(mic_start=99.0, length=320).any()
    assert not timeline.reference_window(mic_start=100.01, length=320).any()


def test_the_ring_forgets_only_what_is_far_enough_behind_the_playhead() -> None:
    timeline = RenderTimeline(sample_rate=RATE, seconds=0.1, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    for index in range(20):
        timeline.submit(ramp(320, start=index * 320), generation="g1", path="tts-file",
                        sink=FixedSinkClock(0.0), now=100.0 + index * 0.02)
    # 0.1 s of history: the sample audible at 100.38 (index 6080) is held, the one audible at
    # 100.20 (index 3200) has fallen out of the ring.
    assert np.allclose(timeline.reference_window(mic_start=100.38, length=160), ramp(160, start=6080))
    assert not timeline.reference_window(mic_start=100.20, length=160).any()


def test_a_stereo_tap_starts_a_new_epoch_instead_of_losing_the_reference() -> None:
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    timeline.submit(ramp(320), generation="g1", path="tts-file", sink=FakeSink(0.04))
    stereo = np.stack((ramp(320, start=320), -ramp(320, start=320)), axis=1)
    timeline.submit(stereo, generation="g1", path="tts-file", sink=FakeSink(0.04), now=100.02)
    assert timeline.channels == 2
    assert timeline.epoch == 1
    # The map restarted at index 0 with the layout the sink really got; the mono chunks before it are
    # gone rather than mixed into a reference with the wrong shape.
    assert timeline.generation("g1").first_index == 0
    window = timeline.reference_window(mic_start=100.06, length=320, mono=False)
    assert window.shape == (320, 2)
    assert np.allclose(window[:, 0], ramp(320, start=320))
    assert np.allclose(window[:, 1], -ramp(320, start=320))


# --------------------------------------------------------------------------- the tap in front of the sink

class FakePad:
    def __init__(self, name: str) -> None:
        self.name = name
        self.peer: FakePad | None = None
        self.linkable = True

    def link(self, other: FakePad) -> bool:
        if not self.linkable or not other.linkable or other.peer is not None:
            return False
        self.peer, other.peer = other, self
        return True

    def unlink(self, other: FakePad) -> None:
        if self.peer is other:
            self.peer, other.peer = None, None

    def get_peer(self) -> FakePad | None:
        return self.peer


class FakeElement:
    def __init__(self, factory: str) -> None:
        self.factory = factory
        self.properties: dict[str, object] = {}
        self.signals: dict[str, object] = {}
        self.pads = {"sink": FakePad("sink"), "src": FakePad("src")}
        self.extra_pads: list[FakePad] = []

    def get_static_pad(self, name: str) -> FakePad | None:
        return self.pads.get(name)

    def request_pad_simple(self, template: str) -> FakePad:
        pad = FakePad(template)
        self.extra_pads.append(pad)
        return pad

    def set_property(self, name: str, value: object) -> None:
        self.properties[name] = value

    def find_property(self, name: str) -> object:
        return object() if name in ("leaky", "drop") else None

    def connect(self, signal: str, callback) -> None:
        self.signals[signal] = callback

    def link(self, other: FakeElement) -> bool:
        return self.pads["src"].link(other.pads["sink"])


class FakePipeline:
    def __init__(self, sink_element: FakeElement, upstream: FakeElement) -> None:
        self.elements = [sink_element, upstream]
        self.by_name = {"sink": sink_element}

    def add(self, element: FakeElement) -> None:
        self.elements.append(element)

    def get_by_name(self, name: str) -> FakeElement | None:
        return self.by_name.get(name)

    def get_property(self, name: str) -> object:
        return None


class FakeGst:
    class ElementFactory:
        @staticmethod
        def make(name: str) -> FakeElement:
            return FakeElement(name)

    class Caps:
        @staticmethod
        def from_string(text: str) -> str:
            return text

    class MapFlags:
        READ = "read"

    class FlowReturn:
        OK = "ok"


def test_the_tap_is_inserted_between_the_playback_path_and_the_sink() -> None:
    sink = FakeElement("autoaudiosink")
    upstream = FakeElement("audioresample")
    upstream.pads["src"].link(sink.pads["sink"])
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")

    made = attach_render_tap(FakePipeline(sink, upstream), timeline, "g1", "tts-file", FakeGst)
    assert made["attached"], made
    # The sink is still fed, and what feeds it is the tee.
    assert sink.pads["sink"].peer is not None
    assert sink.pads["sink"].peer.name.startswith("src_")
    assert upstream.pads["src"].peer is not None
    assert upstream.pads["src"].peer.name == "sink"
    # The tap branch ends in an appsink at the canceller's rate, never leaky.
    branch = [element for element in made["elements"].values()]
    assert any(element.factory == "appsink" for element in branch)
    tap = next(element for element in branch if element.factory == "appsink")
    assert tap.properties["sync"] is False and tap.properties["drop"] is False
    queue = next(element for element in branch if element.factory == "queue")
    assert queue.properties["leaky"] == 0


class TeeRefusingPipeline(FakePipeline):
    """A pipeline whose tee refuses the link the playback path would have to make."""

    def add(self, element: FakeElement) -> None:
        if element.factory == "tee":
            element.pads["sink"].linkable = False
        super().add(element)


def test_a_tap_that_cannot_be_connected_reports_it_and_leaves_the_sound_linked() -> None:
    sink = FakeElement("autoaudiosink")
    upstream = FakeElement("audioresample")
    upstream.pads["src"].link(sink.pads["sink"])
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")

    made = attach_render_tap(TeeRefusingPipeline(sink, upstream), timeline, "g1", "tts-file", FakeGst)
    assert not made["attached"]
    assert made["error"]
    # The original link was put back: a failed tap must never be a reason for silence.
    assert upstream.pads["src"].peer is sink.pads["sink"]


def test_a_pipeline_with_no_sink_to_tap_says_so_instead_of_raising() -> None:
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    made = attach_render_tap(FakePipeline(FakeElement("fakesink"), FakeElement("decodebin")),
                             timeline, "g1", "tts-file", FakeGst)
    assert not made["attached"] and made["error"]
    assert attach_render_tap(None, timeline, "g1", "tts-file", FakeGst)["error"]


def test_the_tap_callback_hands_the_timeline_the_pcm_it_was_given() -> None:
    class Structure:
        def has_field(self, name: str) -> bool:
            return name == "channels"

        def get_value(self, name: str) -> int:
            return 1

    class Caps:
        def get_structure(self, _index: int) -> Structure:
            return Structure()

    class Buffer:
        def __init__(self, data: bytes) -> None:
            self.data = data

        def map(self, _flags):
            return True, type("Mapped", (), {"data": self.data})()

        def unmap(self, _mapped) -> None:
            return None

    class Sample:
        def __init__(self, data: bytes) -> None:
            self.data = data

        def get_buffer(self) -> Buffer:
            return Buffer(self.data)

        def get_caps(self) -> Caps:
            return Caps()

    class TapSink:
        def __init__(self, data: bytes) -> None:
            self.data = data

        def emit(self, _signal: str) -> Sample:
            return Sample(self.data)

    sink = FakeElement("autoaudiosink")
    upstream = FakeElement("audioresample")
    upstream.pads["src"].link(sink.pads["sink"])
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    made = attach_render_tap(FakePipeline(sink, upstream), timeline, "g1", "tts-file", FakeGst)
    callback = made["elements"]["appsink"].signals["new-sample"]

    payload = ramp(320).astype("<f4").tobytes()
    assert callback(TapSink(payload)) == FakeGst.FlowReturn.OK
    assert timeline.total == 320
    assert np.allclose(timeline.reference_window(mic_start=100.0, length=320), ramp(320))
    # A buffer the tap cannot read is dropped, never raised: the sound must keep playing.
    assert callback(object()) == FakeGst.FlowReturn.OK


# --------------------------------------------------------------------------- the capture path

def test_the_in_app_music_player_is_tapped_too() -> None:
    """Every output path, not only speech: music Voxa starts comes back on her microphone.

    VLC and Audacious are deliberately not tapped - they are other applications, and their audio is
    what the reference must never contain.
    """
    from voxa.agent import player as player_module

    class FakePlaybin(FakePipeline):
        def __init__(self) -> None:
            sink = FakeElement("autoaudiosink")
            upstream = FakeElement("audioresample")
            upstream.pads["src"].link(sink.pads["sink"])
            super().__init__(sink, upstream)

        def get_by_name(self, _name: str) -> None:
            return None  # a playbin holds its sink in a property, not by name

        def get_property(self, name: str) -> object:
            return self.elements[0] if name == "audio-sink" else None

    class Timeline:
        sample_rate = RATE
        channels = 1

        def __init__(self) -> None:
            self.opened: list[tuple[str, str]] = []
            self.closed: list[tuple[str, str]] = []

        def open(self, generation: str, path: str, **_kwargs) -> None:
            self.opened.append((generation, path))

        def cancel(self, generation: str, *, reason: str = "", **_kwargs) -> None:
            self.closed.append((generation, reason))

    original = player_module._ensure_gstreamer
    player_module._ensure_gstreamer = lambda: FakeGst
    try:
        media_player = player_module.AudioPlayer()
        timeline = Timeline()
        media_player.render_timeline = timeline
        playbin = FakePlaybin()
        media_player._tap_playback(playbin, start=True)
        assert media_player.tap.get("attached"), media_player.tap
        assert len(timeline.opened) == 1 and timeline.opened[0][1] == "media"
        # The sink is still fed, through the tee.
        assert playbin.elements[0].pads["sink"].peer is not None
        media_player._close_generation("stopped")
        assert timeline.closed == [(timeline.opened[0][0], "stopped")]
        # A player with no timeline plays exactly as before.
        plain = player_module.AudioPlayer()
        plain._tap_playback(playbin, start=True)
        assert plain.tap == {}
    finally:
        player_module._ensure_gstreamer = original


def test_the_capture_serves_its_reference_from_the_timeline_and_never_from_a_monitor() -> None:
    capture = AudioCapture()
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    capture.use_own_reference(timeline)
    assert capture.reference_source is None
    assert capture.reference_mode() == "own-render"

    for index in range(20):  # 0.4 s of render: the window asked for below is inside it
        timeline.submit(ramp(320, start=index * 320), generation="g1", path="tts-file",
                        sink=FakeSink(0.04), now=100.0 + index * 0.02)
    capture.set_capture_latency(0.02)
    capture._last_arrival = 100.30  # the moment the driver handed over this buffer
    window = capture._reference_window(320)
    assert np.allclose(window, ramp(320, start=3520))


def test_the_capture_falls_back_to_the_ring_when_there_is_no_timeline() -> None:
    capture = AudioCapture()
    capture.feed_reference(ramp(640))
    capture._ref_anchor = 0
    capture._mic_counter = 320
    window = capture._reference_window(320)
    assert np.allclose(window, ramp(320, start=320))
    assert capture.reference_mode() == "monitor"


def test_feeding_the_reference_with_a_generation_submits_to_the_timeline() -> None:
    capture = AudioCapture()
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    capture.use_own_reference(timeline)
    assert capture.feed_reference(ramp(320), generation="g1", path="tts-file",
                                  sink=FixedSinkClock(0.04)) == 320
    assert timeline.audible_time(0) == pytest.approx(100.04)
    # Without a generation the plain ring still works: the tap is not a new way to break an old one.
    plain = AudioCapture()
    assert plain.feed_reference(ramp(320)) == 320


def test_the_canceller_receives_the_timeline_window_and_the_mic_is_still_cleaned() -> None:
    class Canceller:
        def __init__(self) -> None:
            self.seen: list[np.ndarray] = []

        def process(self, mic, ref):
            self.seen.append(np.asarray(ref))
            return mic - 0.5 * ref

    capture = AudioCapture()
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-file")
    capture.use_own_reference(timeline)
    for index in range(20):  # 0.4 s of render: the window asked for below is inside it
        timeline.submit(ramp(320, start=index * 320), generation="g1", path="tts-file",
                        sink=FakeSink(0.04), now=100.0 + index * 0.02)
    capture.canceller = Canceller()
    capture._last_arrival = 100.30
    mic = np.full(320, 0.25, dtype="<i2")
    # Without the capture latency subtracted the window is served 20 ms late - 320 samples of the
    # wrong reference, which is the whole error the render timeline exists to remove.
    cleaned = capture._mix_or_passthrough(mic.tobytes())
    assert len(capture.canceller.seen) == 1
    assert np.allclose(capture.canceller.seen[0], ramp(320, start=3840))
    capture.set_capture_latency(0.02)
    capture._mix_or_passthrough(mic.tobytes())
    assert np.allclose(capture.canceller.seen[1], ramp(320, start=3520))
    assert np.frombuffer(cleaned, dtype="<i2").shape == (320,)


def test_an_empty_timeline_falls_back_to_the_ring_it_was_fed() -> None:
    """A tap that never attached costs the clock, not the reference.

    The soak path (VoxaTest) and any machine where the tee could not be inserted still get Voxa's own
    rendered samples; they are simply aligned the old way. Silence would be worse than an old alignment.
    """
    capture = AudioCapture()
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    capture.use_own_reference(timeline)
    capture.feed_reference(ramp(640))
    capture._ref_anchor = 0
    capture._mic_counter = 320
    capture._last_arrival = 100.30
    assert capture.reference_via() == "ring"
    assert capture.reference_mode() == "own-render"
    assert np.allclose(capture._reference_window(320), ramp(320, start=320))
    # Once the tap delivers, the clock takes over.
    timeline.open("g1", "tts-file")
    timeline.submit(ramp(320), generation="g1", path="tts-file", sink=FixedSinkClock(0.04),
                    now=100.0)
    assert capture.reference_via() == "timeline"


def test_clean_chunk_bypasses_a_silent_reference_and_mix_chunk_keeps_its_semantics() -> None:
    class Canceller:
        def process(self, mic, ref):
            return mic - ref

    mic = np.full(8, 1000, dtype="<i2").tobytes()
    assert clean_chunk(mic, np.zeros(8, np.float32), Canceller()) == mic
    assert clean_chunk(mic, np.full(8, 0.1, np.float32), Canceller()) != mic

    from voxa.audio import ReferenceRing

    ring = ReferenceRing(1024)
    ring.append(np.full(16, 0.1, np.float32))
    assert mix_chunk(mic, ring, 0, Canceller(), 0) != mic
    # A reference span the ring never held is silence: the chunk passes through untouched.
    assert mix_chunk(mic, ring, 4096, Canceller(), 0) == mic


# --------------------------------------------------------------------------- the playback paths

def test_the_speech_service_opens_one_generation_per_utterance_and_closes_it_on_stop() -> None:
    from voxa.speech import SpeechService

    class Timeline:
        sample_rate = RATE
        channels = 1

        def __init__(self) -> None:
            self.opened: list[tuple[str, str]] = []
            self.closed: list[tuple[str, str]] = []

        def open(self, generation: str, path: str, **_kwargs) -> None:
            self.opened.append((generation, path))

        def finish(self, generation: str, **_kwargs) -> None:
            self.closed.append((generation, "finished"))

        def cancel(self, generation: str, *, reason: str = "", **_kwargs) -> None:
            self.closed.append((generation, reason))

    speech = SpeechService()
    timeline = Timeline()
    speech.attach_render_timeline(timeline)
    first = speech._open_generation("tts-file")
    assert speech._attach_reference_tap(None, "tts-file") is None  # no pipeline: reported, not raised
    assert speech.tap["attached"] is False
    speech.stop()
    assert timeline.opened == [(first, "tts-file")]
    assert timeline.closed == [(first, "stopped")]
    second = speech._open_generation("tts-stream")
    assert second != first
    speech._close_generation("finished")
    assert timeline.closed[-1] == (second, "finished")
    # A service with no timeline behaves exactly as it did before the tap existed.
    plain = SpeechService()
    assert plain._open_generation("tts-file") is None
    plain.stop()
