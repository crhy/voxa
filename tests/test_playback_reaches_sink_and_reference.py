"""The render tap must reach BOTH the speaker and the reference on the real code path.

The 0.1.9 reference tap broke Voxa's voice in two ways that a structural fake could not see:

* GStreamer 1.28 dropped the ``async`` property from the ``queue`` element. Setting it
  unconditionally raised, and because :func:`attach_render_tap` catches every exception the
  whole tap silently failed: the reference stayed empty ("reference silent: no echo possible")
  and Voxa interrupted her own reply, so no audio was heard and the face's mouth never moved.
* ``Gst.Pad.link`` returns a :class:`Gst.PadLinkReturn` whose ``OK`` is the integer ``0``.
  ``if not upstream.link(...)`` therefore read a *successful* link as a failure and cut the
  playback path.

The fake GStreamer below models both behaviours faithfully (PyGObject semantics) so the tests
fail before the fix and pass after. The companion test at the bottom drives the *real*
GStreamer with a fakesink when it is importable, proving samples reach both the sink and the
reference on the real code path.

This is the same lesson the whole task lives by: a tee/appsink in front of the sink must never
be a reason for silence.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from voxa.timeline import attach_render_tap, RenderTimeline

RATE = 16000


# --------------------------------------------------------------------------- faithful fake Gst

class PadLinkReturn:
    """A Gst.PadLinkReturn stand-in: OK is the integer 0, exactly like the real enum."""

    OK = 0
    NOFORMAT = 1
    INCOMPATIBLE = 2


class Pad:
    def __init__(self, name: str) -> None:
        self.name = name
        self.peer: Pad | None = None

    def link(self, other: Pad) -> int:
        # Real Pad.link returns PadLinkReturn.OK (== 0), which is FALSY: the bug this guards.
        self.peer, other.peer = other, self
        return PadLinkReturn.OK

    def unlink(self, other: Pad) -> None:
        if self.peer is other:
            self.peer, other.peer = None, None

    def get_peer(self) -> Pad | None:
        return self.peer


# Properties this GStreamer build actually has, per element. ``async`` is absent from queue in
# 1.28; a faithful fake must refuse it, the way PyGObject does.
VALID_PROPERTIES = {
    "queue": {"max-size-buffers", "max-size-bytes", "max-size-time", "leaky", "silent"},
    "appsink": {"emit-signals", "sync", "max-buffers", "drop", "name", "wait-on-eos"},
    "capsfilter": {"caps"},
    "fakesink": {"sync", "name", "signal-handoffs"},
    "identity": {"sync", "name", "signal-handoffs"},
}


class Element:
    def __init__(self, factory: str) -> None:
        self.factory = factory
        self.properties: dict[str, object] = {}
        self.signals: dict[str, object] = {}
        self.pads = {"sink": Pad("sink"), "src": Pad("src")}
        self.extra_pads: list[Pad] = []

    def get_static_pad(self, name: str) -> Pad | None:
        return self.pads.get(name)

    def request_pad_simple(self, template: str) -> Pad:
        pad = Pad(template)
        self.extra_pads.append(pad)
        return pad

    def find_property(self, name: str) -> object | None:
        valid = VALID_PROPERTIES.get(self.factory)
        if valid is None:
            return object()  # an element we do not restrict: accept anything
        return object() if name in valid else None

    def set_property(self, name: str, value: object) -> None:
        if self.find_property(name) is None:
            raise TypeError(f"object of type `{self.factory}' does not have property `{name}'")
        self.properties[name] = value

    def connect(self, signal: str, callback) -> None:
        self.signals[signal] = callback

    def link(self, other: "Element") -> bool:
        return self.pads["src"].link(other.pads["sink"]) is PadLinkReturn.OK


class Pipeline:
    def __init__(self, sink_element: Element, upstream: Element) -> None:
        self.elements = [sink_element, upstream]
        self.by_name = {"sink": sink_element}

    def add(self, element: Element) -> None:
        self.elements.append(element)

    def get_by_name(self, name: str) -> Element | None:
        return self.by_name.get(name)

    def get_property(self, name: str) -> object:
        return None


class Gst:
    class ElementFactory:
        @staticmethod
        def make(name: str) -> Element:
            return Element(name)

        @staticmethod
        def find(name: str) -> Element | None:
            # webrtcechoprobe is not installed in this build (matches GStreamer 1.28 here).
            return None

    class Caps:
        @staticmethod
        def from_string(text: str) -> str:
            return text

    PadLinkReturn = PadLinkReturn
    MapFlags = type("MapFlags", (), {"READ": "read"})
    FlowReturn = type("FlowReturn", (), {"OK": "ok"})


def _wire(sink_factory: str = "identity") -> tuple[Element, Element]:
    sink = Element(sink_factory)
    upstream = Element("audioresample")
    upstream.pads["src"].link(sink.pads["sink"])
    return sink, upstream


def test_tap_attaches_when_queue_has_no_async_property() -> None:
    """GStreamer 1.28 removed queue.async: the tap must survive its absence, not fail silently."""
    sink, upstream = _wire()
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-stream")
    made = attach_render_tap(Pipeline(sink, upstream), timeline, "g1", "tts-stream", Gst)
    assert made["attached"], made  # False before the fix: set_property("async") raised
    assert not made.get("error"), made


def test_tap_survives_a_pad_link_that_returns_ok_zero() -> None:
    """A successful Pad.link returns PadLinkReturn.OK (0): it must not read as a failure."""
    sink, upstream = _wire()
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-stream")
    made = attach_render_tap(Pipeline(sink, upstream), timeline, "g1", "tts-stream", Gst)
    assert made["attached"], made
    # The sink is still fed, through the tee, and the tee is fed by the playback path.
    assert sink.pads["sink"].peer is not None
    assert sink.pads["sink"].peer.name.startswith("src_")
    assert upstream.pads["src"].peer is not None
    assert upstream.pads["src"].peer.name == "sink"


def test_samples_reach_both_the_sink_and_the_reference() -> None:
    """The fix's real contract: the same buffers reach the speaker AND the canceller."""
    sink, upstream = _wire()
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: 100.0)
    timeline.open("g1", "tts-stream")
    made = attach_render_tap(Pipeline(sink, upstream), timeline, "g1", "tts-stream", Gst)
    assert made["attached"], made

    # The reference gets the PCM the appsink callback is handed...
    callback = made["elements"]["appsink"].signals["new-sample"]
    payload = (np.arange(320, dtype=np.float32) / 1000.0)

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

    assert callback(TapSink(payload.astype("<f4").tobytes())) == Gst.FlowReturn.OK
    assert timeline.total == 320  # reached the reference

    # ...and the sink is still wired into the playback path (the tee hands it back).
    assert sink.pads["sink"].peer is not None


# --------------------------------------------------------------------------- the real code path

def _system_python_with_gst() -> str | None:
    """A CPython that can import gi/Gst, so the fix can be proven on the real library."""
    import shutil

    for candidate in (sys.executable, "/usr/bin/python3", shutil.which("python3")):
        if not candidate:
            continue
        probe = subprocess.run(
            [candidate, "-c", "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst"],
            capture_output=True, timeout=15,
        )
        if probe.returncode == 0:
            return candidate
    return None


REAL_PROBE = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np, gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
Gst.init(None)
from voxa.timeline import attach_render_tap, RenderTimeline, GstSinkClock
RATE = 16000
pipeline = Gst.parse_launch(
    "appsrc name=source format=bytes block=true max-bytes=1048576 ! "
    "queue max-size-bytes=2097152 ! audioconvert ! audioresample ! identity name=sink sync=false")
sink = pipeline.get_by_name("sink")
timeline = RenderTimeline(sample_rate=RATE, clock=time.monotonic)
timeline.open("g1", "tts-stream")
made = attach_render_tap(pipeline, timeline, "g1", "tts-stream", Gst,
                         sink_clock=GstSinkClock(pipeline, RATE))
seen = []
sink.get_static_pad("sink").add_probe(
    Gst.PadProbeType.BUFFER,
    lambda p, i: (seen.append(i.get_buffer().get_size()), Gst.PadProbeReturn.OK)[1])
appsrc = pipeline.get_by_name("source")
pipeline.set_state(Gst.State.PLAYING)
data = np.arange(8000, dtype=np.int16).tobytes()
appsrc.set_property("caps", Gst.Caps.from_string(
    "audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved"))
buf = Gst.Buffer.new_allocate(None, len(data), None)
buf.fill(0, data)
appsrc.emit("push-buffer", buf)
appsrc.emit("end-of-stream")
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", lambda b, m: loop.quit()
            if m.type in (Gst.MessageType.EOS, Gst.MessageType.ERROR) else None)
GLib.timeout_add_seconds(3, loop.quit)
loop.run()
pipeline.set_state(Gst.State.NULL)
print(int(made.get("attached", False)), len(seen), int(timeline.total))
"""


def test_real_gstreamer_hands_samples_to_the_sink_and_the_reference(tmp_path) -> None:
    """Drive the real attach_render_tap into a fakesink (no sound) and assert both paths feed."""
    python = _system_python_with_gst()
    if python is None:
        pytest.skip("no gi/GStreamer interpreter available")
    import os

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run([python, "-c", REAL_PROBE, repo], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    attached, sink_buffers, reference_total = (int(field) for field in proc.stdout.split())
    assert attached == 1, "the render tap did not attach"
    assert sink_buffers >= 1, "no audio reached the sink"
    assert reference_total >= 8000, "the reference did not get the played samples"
