"""Tests for the WebRTC echo mode in voxa/audio.py with fake GStreamer element factories."""

import types

from voxa.audio import AudioCapture, webrtc_available


def _element_names(pipeline):
    return [element.name for element in pipeline.elements]


def make_fake_gst(available=True, monitor_fails=False):
    """Build a fake Gst module; monitor_fails makes only the webrtc pipeline fail to play."""
    created = []

    class Element:
        def __init__(self, name):
            self.name = name
            self.props = {}

        def set_property(self, key, value):
            self.props[key] = value

        def find_property(self, key):
            return object() if key == "drop" else None

        def connect(self, _name, _handler, _extra=None):
            return True

        def link(self, _other):
            return True

        def link_full(self, other, _a, _b, _pad, _bool_):
            return True

    class Factory:
        def is_available(self, name):
            return available

        def make(self, name):
            return Element(name)

    class Pipeline:
        def __init__(self, name):
            self.name = name
            self.elements = []
            self.state = None
            created.append(self)

        def add(self, element):
            self.elements.append(element)

        def set_state(self, state):
            self.state = state
            if monitor_fails and self.name == "voxa-webrtc":
                return gst_state_change.FAILURE
            return gst_state_change.OK

        def get_state(self):
            if monitor_fails and self.name == "voxa-webrtc":
                return gst_state.STOPPED
            return gst_state.PLAYING

        def get_bus(self):
            return bus

    class Bus:
        def add_signal_watch(self):
            return None

        def connect(self, _name, _handler, _extra=None):
            return None

    class Caps:
        def from_string(self, text):
            return text

    gst = types.ModuleType("Gst")
    gst.ElementFactory = Factory()
    gst.Caps = Caps()
    gst.Pipeline = types.SimpleNamespace(new=lambda name: Pipeline(name))
    gst.Pad = types.SimpleNamespace(NEGOTIATE=0)
    gst.State = types.SimpleNamespace(PLAYING=1, NULL=0, STOPPED=2)
    gst.StateChangeReturn = types.SimpleNamespace(OK=0, LATER=1, FAILURE=2)
    gst.FlowReturn = types.SimpleNamespace(OK=0, ERROR=1)
    bus = Bus()
    gst_state = gst.State
    gst_state_change = gst.StateChangeReturn
    return gst, created


def test_webrtc_available(monkeypatch):
    gst, _created = make_fake_gst(available=True)
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    assert webrtc_available() is True


def test_webrtc_unavailable_is_false(monkeypatch):
    gst, _created = make_fake_gst(available=False)
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    assert webrtc_available() is False


def test_webrtc_available_survives_errors(monkeypatch):
    class BadFactory:
        def is_available(self, name):
            raise OSError("no factory")

    gst = types.ModuleType("Gst")
    gst.ElementFactory = BadFactory()
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    assert webrtc_available() is False


def test_webrtc_mode_builds_one_pipeline(monkeypatch):
    gst, created = make_fake_gst(available=True)
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    cap = AudioCapture()
    cap.reference_source = "alsa_output.default.monitor"
    cap.pulse_source = None
    cap.canceller = None
    cap.start("mic", None, None, None)
    webrtc = [p for p in created if p.name == "voxa-webrtc"]
    assert len(webrtc) == 1
    names = _element_names(webrtc[0])
    assert "webrtcechoprobe" in names and "webrtcdsp" in names
    assert not any(p.name == "voxa-reference" for p in created)
    dsp = [e for e in webrtc[0].elements if e.name == "webrtcdsp"][0]
    assert dsp.props["probe"] == "probe"
    assert dsp.props["echo-cancel"] is True
    probe = [e for e in webrtc[0].elements if e.name == "webrtcechoprobe"][0]
    assert probe.props["name"] == "probe"


def test_other_modes_have_no_webrtc_elements(monkeypatch):
    gst, created = make_fake_gst(available=True)
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    cap = AudioCapture()
    cap.reference_source = "alsa_output.default.monitor"
    cap.canceller = object()  # voxa mode: a canceller is set
    cap.start("mic", None, None, None)
    capture = [p for p in created if p.name == "voxa-capture"]
    assert len(capture) == 1
    names = _element_names(capture[0])
    assert "webrtcechoprobe" not in names and "webrtcdsp" not in names

    gst2, created2 = make_fake_gst(available=True)
    monkeypatch.setattr("voxa.audio.Gst", gst2, raising=False)
    off = AudioCapture()
    off.reference_source = None
    off.canceller = None
    off.start("mic", None, None, None)
    names2 = _element_names(created2[0])
    assert "webrtcechoprobe" not in names2 and "webrtcdsp" not in names2


def test_unavailable_falls_back_without_raising(monkeypatch):
    gst, created = make_fake_gst(available=False)
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    cap = AudioCapture()
    cap.reference_source = "alsa_output.default.monitor"
    cap.canceller = None
    cap.start("mic", None, None, None)  # must not raise
    assert any(p.name == "voxa-capture" for p in created)
    assert cap.pipeline is not None


def test_failed_monitor_falls_back(monkeypatch):
    gst, created = make_fake_gst(available=True, monitor_fails=True)
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    cap = AudioCapture()
    cap.reference_source = "alsa_output.default.monitor"
    cap.canceller = None
    cap.start("mic", None, None, None)  # must not raise
    assert any(p.name == "voxa-webrtc" for p in created)
    assert any(p.name == "voxa-capture" for p in created)
    assert cap.pipeline is not None


def test_set_echo_mode_webrtc_unavailable_behaves_as_off(monkeypatch):
    gst, _created = make_fake_gst(available=False)
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    cap = AudioCapture()
    cap.set_echo_mode("webrtc", "alsa_output.default.monitor")
    assert cap.reference_source is None
    assert cap.canceller is None


def test_set_echo_mode_accepts_old_bool_style(monkeypatch):
    gst, _created = make_fake_gst(available=True)
    monkeypatch.setattr("voxa.audio.Gst", gst, raising=False)
    cap = AudioCapture()
    cap.set_echo_mode(True, "alsa_output.default.monitor")
    assert cap.reference_source == "alsa_output.default.monitor"
    assert cap.canceller is not None
    cap.set_echo_mode(False)
    assert cap.reference_source is None
    assert cap.canceller is None
