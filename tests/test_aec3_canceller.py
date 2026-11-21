"""Roadmap task 2: WebRTC AEC3 as a qualified echo canceller in voxa/anc.py.

No sound card is involved: AEC3 runs on numpy arrays. The GStreamer route (the webrtcechoprobe on
Voxa's own playback branch) is checked against fake element factories, the way
tests/test_audio_webrtc.py checks the capture pipeline.
"""

import types

import numpy as np
import pytest

from voxa import anc
from voxa.anc import Aec3Canceller, EchoCanceller, aec3_available, aec3_build_info
from voxa.audio import AudioCapture, _device_base, clean_chunk


@pytest.fixture(autouse=True)
def _no_lingering_owner():
    """Every test starts and ends with nobody owning the echo path."""
    anc._AEC_OWNER = None
    yield
    anc._AEC_OWNER = None


needs_aec3 = pytest.mark.skipif(not aec3_available(), reason="pywebrtc-audio not installed")


def _echo_pair(seconds=2.0, delay=40, rate=16000, seed=7):
    """A far-end signal, its delayed echo in a room, and a quiet microphone noise floor."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    ref = (rng.standard_normal(n) * 0.2).astype(np.float32)
    ref[: rate // 4] = 0.0
    ref[-rate // 8:] = 0.0
    echo = np.concatenate([np.zeros(delay), ref])[:n] * 0.3
    mic = echo + (rng.standard_normal(n) * 2e-3).astype(np.float32)
    return mic.astype(np.float32), ref


# --- the backend really is AEC3 ----------------------------------------------------------------

def test_the_binding_reports_what_it_links():
    info = aec3_build_info()
    assert info["package"] == "pywebrtc-audio"
    assert info["version"] and info["version"].startswith("0.2")
    # The wrapper's NAME says "webrtc"; only these fields say whether AEC3's config is reachable.
    # pywebrtc-audio 0.2.0 exposes neither EchoCanceller3Config nor has_clock_drift, which is why
    # Aec3Canceller keeps Voxa's own drift flag instead of pushing one into AEC3.
    assert info["config_exposed"] is False
    assert info["clock_drift_exposed"] is False


@needs_aec3
def test_aec3_cancels_a_delayed_echo_through_the_canceller_interface():
    mic, ref = _echo_pair(seconds=2.5)
    canceller = Aec3Canceller()
    try:
        out = canceller.process(mic, ref)
    finally:
        canceller.close()
    assert out.dtype == np.float32 and out.size == mic.size
    tail = slice(int(1.5 * 16000), mic.size)
    assert anc.erle_db(mic[tail], out[tail]) > 15.0


@needs_aec3
def test_any_chunk_length_is_cut_into_aec3s_10ms_frames():
    mic, ref = _echo_pair(seconds=1.0)
    canceller = Aec3Canceller()
    try:
        pieces, i = [], 0
        sizes = (320, 37, 160, 1000, 1, 96)  # not multiples of 160: remainders flush whole
        k = 0
        while i < len(mic):
            size = sizes[k % len(sizes)]
            piece = canceller.process(mic[i:i + size], ref[i:i + size])
            pieces.append(piece)
            i += size
            k += 1
        assert sum(piece.size for piece in pieces) == len(mic)
        out = np.concatenate(pieces)
        assert out.size == len(mic)  # process() never swallows and never invents samples
        # 16000 samples: everything AEC3 saw was a whole 10 ms frame, and the vast majority of
        # the stream was cancelled by those frames (a sub-frame remainder flushes raw instead).
        assert canceller.frames_processed * 160 <= len(mic)
        assert canceller.frames_processed >= 0.7 * (len(mic) // 160)  # measured 77 of 100
    finally:
        canceller.close()


@needs_aec3
def test_a_frame_aec3_makes_loud_is_passed_through_raw():
    mic, ref = _echo_pair(seconds=0.5)
    canceller = Aec3Canceller()
    try:
        # pywebrtc-audio's cdef class forbids assigning its own process(): replace the impl.
        canceller._impl = types.SimpleNamespace(
            process=lambda _near, _far: np.full(160, 0.9, dtype=np.float32))
        out = canceller.process(mic, ref)
        assert canceller.guard_frames > 0
        assert np.allclose(out[:160], mic[:160])
    finally:
        canceller.close()


@needs_aec3
def test_the_double_talk_guard_keeps_the_user_instead_of_suppressing_her():
    """A user speaking over the echo must not be pushed down by AEC3's suppressor."""
    mic, ref = _echo_pair(seconds=3.0)
    rng = np.random.default_rng(11)
    user = (rng.standard_normal(mic.size) * 0.05).astype(np.float32)
    user[: 16000] = 0.0
    mic_dt = mic + user
    guarded = Aec3Canceller()
    try:
        out_guarded = guarded.process(mic_dt, ref)
    finally:
        guarded.close()
    plain = Aec3Canceller(double_talk_guard=False)
    try:
        out_plain = plain.process(mic_dt, ref)
    finally:
        plain.close()
    span = slice(int(2.0 * 16000), mic.size)
    def level(x):
        return float(np.mean(np.asarray(x[span], dtype=np.float64) ** 2))
    assert guarded.double_talk_frames > 0
    assert level(out_guarded) > level(out_plain)  # the user survives better with the guard


@needs_aec3
def test_aec3_rejects_a_sample_rate_it_cannot_run():
    with pytest.raises(ValueError):
        Aec3Canceller(sample_rate=44100)


# --- exactly one AEC owner ---------------------------------------------------------------------

@needs_aec3
def test_a_second_live_aec3_is_refused():
    first = Aec3Canceller()
    try:
        with pytest.raises(AssertionError):
            Aec3Canceller()
        assert anc.aec_owner_kind() == "aec3"
    finally:
        first.close()
    assert anc.aec_owner_kind() is None
    second = Aec3Canceller()
    second.close()


@needs_aec3
def test_the_gstreamer_dsp_and_aec3_cannot_both_own_the_path():
    capture = AudioCapture()
    anc.claim_aec_owner(capture, "gst-webrtcdsp")
    try:
        with pytest.raises(AssertionError):
            Aec3Canceller()
    finally:
        anc.release_aec_owner(capture)
    assert anc.aec_owner_kind() is None


def test_releasing_someone_elses_claim_does_nothing():
    class Holder:
        pass

    holder = Holder()
    anc.claim_aec_owner(holder, "holder")
    anc.release_aec_owner(Holder())
    assert anc.aec_owner_kind() == "holder"
    anc.release_aec_owner(holder)
    assert anc.aec_owner_kind() is None


# --- the interface the capture path and the bench rely on --------------------------------------

@needs_aec3
def test_aec3_answers_the_measurement_properties():
    canceller = Aec3Canceller()
    try:
        mic, ref = _echo_pair(seconds=0.5)
        canceller.process(mic, ref)
        assert canceller.delay_samples == 0
        assert canceller.delay_managed_internally is True
        assert canceller.continuous is True
        assert canceller.mic_power_db > -120.0
        assert canceller.frames_processed > 0
        assert canceller.export_state()["saved_filter"] is False
        canceller.load_state({"backend": "aec3"})  # accepted, does nothing
        canceller.reset()
        assert canceller.frames_processed == 0
    finally:
        canceller.close()


def test_aec3_declares_clock_drift_when_the_devices_differ():
    capture = AudioCapture()
    capture._mic_identifier = "alsa_input.usb-Blue_Snowball-00.analog-mono"
    capture.set_sink_identifier("alsa_output.pci-0000_00_1b.0.analog-stereo.monitor")
    assert capture.devices_differ() is True
    capture._mic_identifier = "alsa_input.pci-0000_00_1b.0.analog-stereo"
    assert capture.devices_differ() is False
    capture.set_sink_identifier(None)
    assert capture.devices_differ() is False  # unknown is not evidence


@needs_aec3
def test_the_capture_path_hands_the_drift_state_to_aec3():
    capture = AudioCapture()
    capture._mic_identifier = "alsa_input.usb-Snowball-00.analog-mono"
    capture.canceller = Aec3Canceller()
    try:
        capture.set_sink_identifier("alsa_output.pci-0000_00_1b.0.analog-stereo.monitor")
        assert capture.canceller.has_clock_drift is True
        capture.set_sink_identifier("alsa_output.usb-Snowball-00.analog-stereo.monitor")
        assert capture.canceller.has_clock_drift is False
    finally:
        capture.canceller.close()


def test_device_base_compares_the_card_not_the_profile():
    assert _device_base("alsa_output.pci-0000_00_1b.0.analog-stereo.monitor") == "pci-0000_00_1b"
    assert _device_base("alsa_input.pci-0000_00_1b.0.analog-stereo") == "pci-0000_00_1b"
    assert _device_base("device.serial:usb-BLUE_Microphones_297498") == "usb-blue_microphones_297498"


# --- the capture path must feed AEC3 every frame ------------------------------------------------

def test_a_continuous_canceller_is_never_bypassed_for_a_silent_reference():
    calls = []

    class Watcher:
        continuous = True

        def process(self, mic, ref):
            calls.append(len(mic))
            return mic

    mic_bytes = np.zeros(320, dtype="<i2").tobytes()
    out = clean_chunk(mic_bytes, np.zeros(320, dtype=np.float32), Watcher())
    assert calls == [320] and out == mic_bytes


def test_the_nlms_canceller_keeps_its_silent_reference_bypass():
    calls = []

    class Watcher:
        def process(self, mic, ref):
            calls.append(len(mic))
            return mic

    clean_chunk(np.zeros(320, dtype="<i2").tobytes(), np.zeros(320, dtype=np.float32), Watcher())
    assert calls == []


# --- set_echo_mode knows the new backend -------------------------------------------------------

def test_set_echo_mode_aec3_falls_back_to_the_nlms_canceller_without_the_binding(monkeypatch):
    monkeypatch.setattr("voxa.audio.aec3_available", lambda: False)
    capture = AudioCapture()
    capture.set_echo_mode("aec3")
    assert isinstance(capture.canceller, EchoCanceller)


@needs_aec3
def test_set_echo_mode_aec3_runs_aec3_and_switching_releases_it():
    capture = AudioCapture()
    capture.set_echo_mode("aec3")
    assert isinstance(capture.canceller, Aec3Canceller)
    assert anc.aec_owner_kind() == "aec3"
    capture.set_echo_mode("voxa")
    assert isinstance(capture.canceller, EchoCanceller)
    assert anc.aec_owner_kind() is None


@needs_aec3
def test_switching_to_the_gstreamer_route_gives_the_claim_up_first(monkeypatch):
    capture = AudioCapture()
    capture.set_echo_mode("aec3")
    monkeypatch.setattr("voxa.audio.webrtc_available", lambda: True)
    capture.set_echo_mode("webrtc")  # no capture running: only the state moves, no pipeline built
    assert capture.canceller is None
    assert anc.aec_owner_kind() is None


@needs_aec3
@needs_aec3
def test_the_delay_estimator_does_not_replace_aec3(monkeypatch):
    capture = AudioCapture()
    capture.set_echo_mode("aec3")
    try:
        capture._delay_done = False
        rate = 16000
        rng = np.random.default_rng(1)
        loud = rng.standard_normal(rate // 10).astype(np.float32) * 0.2
        monkeypatch.setattr(capture, "_reference_window", lambda samples: loud)
        for _ in range(30):  # 3 s of loud reference: the estimator's trigger
            capture._delay_mic.append(loud)
            capture._delay_ref.append(loud)
        capture._maybe_estimate_delay((loud * 32767).astype("<i2").tobytes())
        assert capture._delay_done is True  # the estimate was made...
        assert isinstance(capture.canceller, Aec3Canceller)  # ...and AEC3 was not replaced by NLMS
    finally:
        capture.canceller.close()


# --- the GStreamer route: the probe rides the playback branch ----------------------------------

class _Pad:
    def __init__(self, name, element):
        self.name = name
        self.element = element
        self.peer = None

    def get_peer(self):
        return self.peer

    def unlink(self, _other):
        return True

    def link(self, _other):
        return True


class _Element:
    def __init__(self, name):
        self.name = name
        self.props = {}
        self.pads = {}

    def set_property(self, key, value):
        self.props[key] = value

    def find_property(self, key):
        return object()

    def connect(self, _name, _handler, _extra=None):
        return True

    def get_static_pad(self, name):
        return self.pads.setdefault(name, _Pad(name, self))

    def request_pad_simple(self, pattern):
        pad = _Pad(pattern % len(self.pads), self)
        self.pads[pad.name] = pad
        return pad

    def link(self, other):
        return True

    def unlink(self, _pad):
        return True


class _Pipeline:
    def __init__(self, name):
        self.name = name
        self.elements = []
        self.state = None

    def add(self, element):
        self.elements.append(element)

    def set_state(self, state):
        self.state = state
        return 1  # SUCCESS

    def get_state(self, _timeout=0):
        return (1, 1, 1)

    def get_bus(self):
        return types.SimpleNamespace(add_signal_watch=lambda: None,
                                     connect=lambda *_a: None)


def _fake_gst(missing=()):
    """A tiny fake Gst module: enough for the capture pipeline and for the render tap."""
    created = []

    class Factory:
        def find(self, name):
            return None if name in missing else object()

        def make(self, name):
            element = _Element(name)
            created.append(element)
            return element

    def new_pipeline(name):
        pipeline = _Pipeline(name)
        created.append(pipeline)
        return pipeline

    gst = types.ModuleType("Gst")
    gst.ElementFactory = Factory()
    gst.Caps = types.SimpleNamespace(from_string=lambda text: text)
    gst.Pipeline = types.SimpleNamespace(new=new_pipeline)
    gst.State = types.SimpleNamespace(PLAYING=1, NULL=0, STOPPED=2)
    gst.StateChangeReturn = types.SimpleNamespace(SUCCESS=1, ASYNC=2, NO_PREROLL=3, FAILURE=0)
    gst.SECOND = 1_000_000_000
    gst.FlowReturn = types.SimpleNamespace(OK=0, ERROR=1)
    return gst, created


def test_the_webrtc_route_with_an_own_render_reference_opens_no_monitor():
    gst, created = _fake_gst()
    capture = AudioCapture()
    capture.use_own_reference()
    assert capture._start_webrtc(gst, "mic-1", lambda *a: None, lambda *a: None) is True
    pipeline = next(p for p in created if isinstance(p, _Pipeline) and p.name == "voxa-webrtc")
    names = [element.name for element in pipeline.elements]
    # One pulsesrc only: the microphone. The monitor branch that used to feed the probe is gone.
    assert names.count("pulsesrc") == 1
    dsp = next(element for element in pipeline.elements if element.name == "webrtcdsp")
    assert dsp.props["probe"] == "voxa-echo-probe"
    assert capture.echo_probe_name == "voxa-echo-probe"
    assert capture.timeline.echo_probe_name == "voxa-echo-probe"
    assert anc.aec_owner_kind() == "gst-webrtcdsp"
    capture.stop()
    assert anc.aec_owner_kind() is None


def test_the_webrtc_route_without_an_own_render_reference_keeps_the_monitor_branch():
    gst, created = _fake_gst()
    capture = AudioCapture()
    capture.reference_source = "alsa_output.pci-0.analog-stereo.monitor"
    assert capture._start_webrtc(gst, "mic-1", lambda *a: None, lambda *a: None) is True
    pipeline = next(p for p in created if isinstance(p, _Pipeline) and p.name == "voxa-webrtc")
    names = [element.name for element in pipeline.elements]
    assert names.count("pulsesrc") == 2  # microphone plus the monitor, as before task 2
    assert "webrtcechoprobe" in names
    assert capture.echo_probe_name is None


def test_the_render_tap_registers_the_probe_on_the_playback_branch():
    from voxa.timeline import RenderTimeline, attach_render_tap

    gst, _created = _fake_gst()
    timeline = RenderTimeline(sample_rate=16000)
    timeline.echo_probe_name = "voxa-echo-probe"
    pipeline = _Pipeline("speech")
    sink = _Element("autoaudiosink")
    upstream = _Element("audioconvert")
    pipeline.add(sink)
    pipeline.add(upstream)
    sink_pad = sink.get_static_pad("sink")
    sink_pad.peer = upstream.get_static_pad("src")
    made = attach_render_tap(pipeline, timeline, "gen-1", "tts-stream", gst=gst, sink_element=sink)
    assert made["attached"] is True
    names = [element.name for element in pipeline.elements]
    assert "webrtcechoprobe" in names
    probe = next(element for element in pipeline.elements if element.name == "webrtcechoprobe")
    assert probe.props["name"] == "voxa-echo-probe"
    assert made["echo_probe"]["attached"] is True


def test_the_render_tap_works_without_a_probe_name():
    from voxa.timeline import RenderTimeline, attach_render_tap

    gst, _created = _fake_gst()
    timeline = RenderTimeline(sample_rate=16000)
    pipeline = _Pipeline("speech")
    sink = _Element("autoaudiosink")
    upstream = _Element("audioconvert")
    pipeline.add(sink)
    pipeline.add(upstream)
    sink.get_static_pad("sink").peer = upstream.get_static_pad("src")
    made = attach_render_tap(pipeline, timeline, "gen-1", "tts-stream", gst=gst, sink_element=sink)
    assert made["attached"] is True
    assert "webrtcechoprobe" not in [element.name for element in pipeline.elements]


def test_a_missing_probe_element_is_reported_and_the_tap_still_works():
    from voxa.timeline import RenderTimeline, attach_render_tap

    gst, _created = _fake_gst(missing=("webrtcechoprobe",))
    timeline = RenderTimeline(sample_rate=16000)
    timeline.echo_probe_name = "voxa-echo-probe"
    pipeline = _Pipeline("speech")
    sink = _Element("autoaudiosink")
    upstream = _Element("audioconvert")
    pipeline.add(sink)
    pipeline.add(upstream)
    sink.get_static_pad("sink").peer = upstream.get_static_pad("src")
    made = attach_render_tap(pipeline, timeline, "gen-1", "tts-stream", gst=gst, sink_element=sink)
    assert made["attached"] is True
    assert made["echo_probe"]["attached"] is False
