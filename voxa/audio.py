from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .anc import (Aec3Canceller, DelayTracker, EchoCanceller, aec3_available, claim_aec_owner,
                  estimate_delay,
                  release_aec_owner)
from .timeline import CaptureClock, RenderTimeline

Gst: Any = None  # Initialized lazily so importing this module needs no GStreamer or PyGObject.

log = logging.getLogger("voxa.audio")


def _ensure_gstreamer() -> Any:
    """The GStreamer module, or None when PyGObject is not installed here (never raises, so a
    machine without GStreamer can still import and run every non-webrtc path in this module)."""
    global Gst
    if Gst is None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst as _Gst

            _Gst.init(None)
            Gst = _Gst
        except ImportError:
            log.info("GStreamer (PyGObject) is not importable: webrtc echo cancellation is off")
    return Gst

# Reference samples we keep: the last 3 seconds at 16 kHz.
REFERENCE_SECONDS = 3
REFERENCE_RATE = 16000


def webrtc_available() -> bool:
    """True when both WebRTC echo cancellation elements exist in this GStreamer runtime."""
    try:
        gst = _ensure_gstreamer()
        factory = gst.ElementFactory
        return all(factory.find(name) is not None for name in ("webrtcdsp", "webrtcechoprobe"))
    except Exception:
        return False


class ReferenceRing:
    """Thread-safe ring buffer of the most recent reference samples with a running counter.

    ``total`` is the number of samples ever appended, so a sample's global position in the
    reference stream is ``total - len(buffer) + index``. ``window`` returns the samples covering a
    span of global positions, zero-padding any position the buffer no longer holds (or has not
    produced yet).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._buf = np.zeros(0, dtype=np.float32)
        self._total = 0
        self._lock = threading.Lock()

    def append(self, samples) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return
        with self._lock:
            self._total += samples.size
            merged = np.concatenate([self._buf, samples])
            self._buf = merged[-self._capacity :]

    def window(self, start: int, length: int) -> np.ndarray:
        """Reference samples for global positions ``[start, start + length)``, zero-padded when absent."""
        out = np.zeros(length, dtype=np.float32)
        if length <= 0:
            return out
        with self._lock:
            oldest = self._total - self._buf.size
            lo = max(start, oldest, 0)
            hi = min(start + length, self._total)
            if hi > lo:
                out[lo - start : hi - start] = self._buf[lo - oldest : hi - oldest]
        return out

    @property
    def total(self) -> int:
        return self._total


def clean_chunk(mic_bytes: bytes, ref: np.ndarray, canceller) -> bytes:
    """Clean one microphone chunk against an ALREADY ALIGNED reference window.

    A chunk whose reference span is silence is passed through untouched (bypass): nothing of Voxa's was
    playing over that span, so there is no echo to fit and the adaptive filter is not fed a lie. Returns
    the cleaned chunk as int16 bytes. Raises if the canceller raises.

    A canceller that declares ``continuous`` (AEC3, :class:`voxa.anc.Aec3Canceller`) is NOT bypassed:
    its delay estimator, filter and suppressor keep state across frames, so it must see every frame of
    the stream, silence and echo tail included (handoff 6.2: the canceller runs continuously).
    """
    ref = np.asarray(ref, dtype=np.float32)
    if ref.size == 0:
        return mic_bytes
    if (not getattr(canceller, "continuous", False)
            and float(np.mean(ref * ref)) < 1e-9):
        return mic_bytes
    mic = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float32) / 32768.0
    cleaned = canceller.process(mic, ref)
    cleaned = np.clip(np.rint(np.asarray(cleaned, dtype=np.float32) * 32767.0), -32768, 32767)
    return cleaned.astype("<i2").tobytes()


def mix_chunk(mic_bytes: bytes, ref_ring: ReferenceRing, counter: int, canceller,
              ref_offset: int = 0) -> bytes:
    """Clean one microphone chunk against the reference samples covering the same span.

    ``counter`` is the global sample position where this microphone chunk begins; ``ref_offset``
    is the microphone position that corresponds to the reference stream's position 0, so the
    reference samples for the same wall-clock span start at ``counter - ref_offset`` (zero-padded
    when the reference has not produced enough yet). Returns the cleaned chunk as int16 bytes.
    """
    mic_samples = len(mic_bytes) // 2
    ref = ref_ring.window(counter - ref_offset, mic_samples)
    return clean_chunk(mic_bytes, ref, canceller)


@dataclass(slots=True)
class AudioDevice:
    identifier: str
    name: str
    device: Any
    properties: Any = None


def _stable_field(properties, key: str) -> str:
    """Read one property in a backend-agnostic way; never raises."""
    if properties is None:
        return ""
    with contextlib.suppress(Exception):
        if properties.has_field(key):
            value = properties.get_value(key)
            return value if value is not None else ""
    return ""


def is_monitor_source(properties, display_name: str = "") -> bool:
    """Return True when a source device is a loopback monitor, not a real microphone.

    Pipewire and PulseAudio expose "Monitor of …" nodes through the same
    Audio/Source device class as real microphones. Opening one captures the
    machine's own playback instead of the user's voice, so monitors are
    excluded from the microphone picker and refused at capture start.
    """
    stable = _stable_field(properties, "object.stable-id").casefold()
    if "monitor" in stable:
        return True
    name = (display_name or "").casefold()
    # Voxa's own echo-cancelling device is plumbing, not a microphone to choose: Voxa routes the chosen
    # microphone through it by itself.
    if "voxa-echo-cancel" in name or "voxa_echo_cancel" in name or "voxa_echo_cancel" in stable:
        return True
    return "monitor of" in name


def _device_base(identifier: str) -> str:
    """The device a source or sink name belongs to, with profile and monitor parts removed.

    ``alsa_output.pci-0000_00_1b.0.analog-stereo.monitor`` and ``alsa_input.pci-0000_00_1b.0
    .analog-stereo`` both answer ``pci-0000_00_1b``: one sound card, one clock.
    """
    text = str(identifier).strip()
    for prefix in ("alsa_output.", "alsa_input.", "auto_output.", "auto_input.", "of_output."):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.endswith(".monitor"):
        text = text[:-len(".monitor")]
    for suffix in (".analog-stereo", ".analog-mono", ".digital-stereo", ".digital-surround51",
                   ".digital-surround71", ".iec958-stereo", ".mono", ".stereo"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    if ":" in text:
        text = text.split(":", 1)[1]
    return (text.split(".")[0] or text).strip().lower()


class AudioCapture:
    """Native GStreamer microphone capture producing 16 kHz mono signed PCM.

    When :attr:`reference_source` is set, a second pipeline records the computer's own playback
    (the loopback monitor of the default sink) into a ring buffer. When :attr:`canceller` is set,
    each microphone chunk is cleaned against the reference samples covering the same span before it
    reaches the wake word, Whisper and barge-in.
    """

    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self._ref_pipeline: Any | None = None
        self._devices: list[AudioDevice] = []
        # Name of a PulseAudio source to record from instead of the selected device, or None.
        self.pulse_source: str | None = None
        # Loopback monitor to record the computer's own playback from, or None.
        self.reference_source: str | None = None
        # The reference policy that chose it (voxa/reference.ReferenceDecision), or None when unjudged.
        self.reference_decision: Any | None = None
        # True when the reference is Voxa's own rendered samples: no stream on the speakers is in it.
        self.own_reference: bool = False
        # Roadmap task 1: the render timeline (voxa/timeline.py) that owns the own-render reference.
        self.timeline: RenderTimeline | None = None
        # Stamps a capture buffer with the moment its samples left the microphone.
        self.capture_clock = CaptureClock(rate=REFERENCE_RATE)
        # Monotonic time the last capture buffer was delivered, or None before the first one.
        self._last_arrival: float | None = None
        # The canceller in voxa/anc.py that cleans the microphone, or None when inactive.
        self.canceller: EchoCanceller | None = None
        self._on_audio = None
        self._on_error = None
        self._last_level_emit = 0.0
        self._ref_ring: ReferenceRing | None = None
        self._mic_counter = 0
        # How far the reference stream's position 0 sits ahead of the microphone stream's,
        # or None until the monitor has produced its first sample. The monitor delivers
        # nothing while the sink is idle, so the two streams do not start together.
        self._ref_anchor: int | None = None
        self._canceller_enabled = True
        self._canceller_logged = False
        self._delay_done = True
        self._delay_mic: list = []
        self._delay_ref: list = []
        self._delay_tracker: DelayTracker | None = None
        self._start_args: tuple | None = None
        # Roadmap task 2: the device the microphone is on and the device the sound goes out on.
        # Different devices mean different clocks, which is what AEC3's drift handling wants to know.
        self._mic_identifier: str | None = None
        self._sink_identifier: str | None = None
        # The GStreamer route: the name of the webrtcechoprobe the playback tap registers (task 2).
        self.echo_probe_name: str | None = None

    def list_devices(self) -> list[AudioDevice]:
        gst = _ensure_gstreamer()
        monitor = gst.DeviceMonitor.new()
        monitor.add_filter("Audio/Source", None)
        if not monitor.start():
            return []
        try:
            found: list[AudioDevice] = []
            seen: set[str] = set()
            for index, device in enumerate(monitor.get_devices()):
                name = device.get_display_name() or f"Microphone {index + 1}"
                props = device.get_properties()
                if is_monitor_source(props, name):
                    continue
                stable = self._device_identifier(props, name)
                if stable in seen:
                    continue
                seen.add(stable)
                found.append(AudioDevice(identifier=stable, name=name, device=device, properties=props))
            self._devices = found
            return list(found)
        finally:
            monitor.stop()

    @staticmethod
    def _device_identifier(properties, fallback: str) -> str:
        if properties is None:
            return fallback
        for key in (
            "device.serial",
            "device.path",
            "node.name",
            "object.path",
            "alsa.card_name",
        ):
            value = _stable_field(properties, key)
            if value:
                return f"{key}:{value}"
        with contextlib.suppress(Exception):
            return properties.to_string()
        return fallback

    def start(self, device_id: str, on_audio, on_level, on_error) -> None:
        self.stop()
        self._start_args = (device_id, on_audio, on_level, on_error)
        self._on_audio = on_audio
        self._on_error = on_error
        self._mic_identifier = device_id
        self.apply_clock_drift()

        selected = next((item for item in self._devices if item.identifier == device_id), None)
        # A profile saved by an older build could still point at a monitor
        # node; refuse to open it rather than capture the machine's own audio.
        if selected is not None and is_monitor_source(selected.properties, selected.name):
            raise RuntimeError("The selected source is a monitor stream and cannot capture your voice.")
        gst = _ensure_gstreamer()
        if self.reference_source and self.canceller is None and webrtc_available():
            # "webrtc" mode: one pipeline with both branches, no numpy canceller.
            if self._start_webrtc(gst, device_id, on_audio, on_level):
                return
        source = None
        if self.pulse_source:
            # The echo-cancelled microphone (see voxa/echo.py): the computer's own sound is already removed.
            source = gst.ElementFactory.make("pulsesrc")
            if source is not None:
                source.set_property("device", self.pulse_source)
        if source is None:
            if selected is not None:
                source = selected.device.create_element(None)
            else:
                source = gst.ElementFactory.make("autoaudiosrc")
        convert = gst.ElementFactory.make("audioconvert")
        resample = gst.ElementFactory.make("audioresample")
        capsfilter = gst.ElementFactory.make("capsfilter")
        sink = gst.ElementFactory.make("appsink")
        if not all((source, convert, resample, capsfilter, sink)):
            raise RuntimeError("Required GStreamer audio elements are unavailable.")

        capsfilter.set_property("caps", gst.Caps.from_string("audio/x-raw,format=S16LE,channels=1,rate=16000"))
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 12)
        if sink.find_property("drop") is not None:
            sink.set_property("drop", True)
        sink.connect("new-sample", self._on_sample, on_level)

        pipeline = gst.Pipeline.new("voxa-capture")
        for element in (source, convert, resample, capsfilter, sink):
            pipeline.add(element)
        if not source.link(convert) or not convert.link(resample) or not resample.link(capsfilter) or not capsfilter.link(sink):
            pipeline.set_state(gst.State.NULL)
            raise RuntimeError("Could not connect the GStreamer audio pipeline.")

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        self.pipeline = pipeline
        result = pipeline.set_state(gst.State.PLAYING)
        if result == gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("The selected microphone could not be opened.")

        # Both counters start together now that the microphone is actually playing.
        self._mic_counter = 0
        self._ref_anchor = None
        self._canceller_enabled = True
        self._canceller_logged = False
        self._delay_done = self.canceller is None
        self._delay_mic = []
        self._delay_ref = []
        self._delay_tracker = None  # a fresh capture or mode change is an explicit epoch
        self._ref_ring = (ReferenceRing(REFERENCE_SECONDS * REFERENCE_RATE)
                           if (self.reference_source or self.own_reference) else None)
        if self.reference_source:
            self._start_reference(gst)
        elif self.own_reference:
            log.info("echo reference: Voxa's own rendered samples; the test voice cannot enter it")

    def use_own_reference(self, timeline: RenderTimeline | None = None) -> None:
        """Take the reference from Voxa's own render instead of from any monitor (roadmap task 1).

        Nothing that reaches the speakers can enter this reference, so the test voice VoxaTest plays is
        excluded by construction: her canceller can never cancel the test user (handoff 8.2). The
        timeline is the single authoritative map of what her sinks played and when (voxa/timeline.py);
        the playback paths submit to it, the capture path reads windows out of it.
        """
        self.own_reference = True
        self.reference_source = None
        if self.timeline is None:
            self.timeline = timeline or RenderTimeline(sample_rate=REFERENCE_RATE)

    def set_sink_identifier(self, identifier: str | None) -> None:
        """Record the output device the sound goes to, so clock drift can be judged (task 2)."""
        self._sink_identifier = identifier
        self.apply_clock_drift()

    def devices_differ(self) -> bool:
        """True when the microphone and the output are known to be different physical devices.

        Unknown on either side answers False: claiming drift without evidence would be a guess. A
        monitor source names the sink it belongs to, so ``alsa_output.pci-0000_00_1b.0.analog-stereo
        .monitor`` and that same sink count as ONE device, while a USB headset microphone and the
        built-in output count as two.
        """
        mic, sink = self._mic_identifier, self._sink_identifier
        if not mic or not sink:
            return False
        return _device_base(mic) != _device_base(sink)

    def apply_clock_drift(self) -> None:
        """Hand the drift state to the canceller when it can use one (AEC3, roadmap task 2)."""
        canceller = self.canceller
        if canceller is None or not hasattr(canceller, "set_clock_drift"):
            return
        canceller.set_clock_drift(self.devices_differ())

    def set_capture_latency(self, seconds: float | Any) -> None:
        """How long the microphone driver sits on a sample before the callback sees it.

        A capture buffer is delivered late; without subtracting that, the reference window is served
        for the wrong span by exactly the capture latency. ``seconds`` may be a number or a callable
        that asks the live source (a driver whose latency changes with a Bluetooth profile must not
        freeze the alignment). Unknown or unanswered means 0: no correction, the old behaviour.
        """
        self.capture_clock.latency = seconds

    def reference_mode(self) -> str:
        """``own-render`` / ``monitor`` / ``none``: what the canceller is being fed right now."""
        if self.own_reference:
            return "own-render"
        if self._ref_ring is not None or self.reference_source:
            return "monitor"
        return "none"

    def reference_via(self) -> str:
        """``timeline`` when the render timeline answers, ``ring`` when the plain ring does.

        A tap that never attached leaves the timeline empty; the capture path then keeps serving the
        ring it was always fed, so a missing tap costs the clock, not the reference.
        """
        if self.timeline is not None and self._last_arrival is not None and self.timeline.total > 0:
            return "timeline"
        if self._ref_ring is not None or self.timeline is not None:
            return "ring"
        return "none"

    def feed_reference(self, samples, *, generation: str | None = None, path: str = "tts-file",
                       sink: Any = None) -> int:
        """Hand Voxa's own rendered samples to the reference; returns the reference length.

        With a render timeline (roadmap task 1) the samples go to it with their generation id and the
        sink they were submitted to, so the timeline can say when each of them was audible. Without
        one they fall back to the plain ring, which has no clock and is aligned by the anchor trick.
        This is the only writer of the reference in own-reference mode, which is why a stream playing
        on the speakers - VoxaTest's test voice included - can never appear in it.
        """
        if self.timeline is not None and generation is not None:
            self.timeline.submit(samples, generation=generation, path=path, sink=sink)
            return self.timeline.total
        if self._ref_ring is None:
            self._ref_ring = ReferenceRing(REFERENCE_SECONDS * REFERENCE_RATE)
        self._ref_ring.append(samples)
        return self._ref_ring.total

    @property
    def reference_ring(self) -> ReferenceRing | None:
        return self._ref_ring

    def _start_reference(self, gst) -> None:
        """Open the second pipeline that records the computer's own playback into the ring buffer."""
        if self.own_reference:
            log.info("echo reference: Voxa's own rendered samples; no monitor pipeline is opened")
            return
        decision = self.reference_decision
        if decision is not None:
            from .reference import log_reference

            log_reference(decision)
            if decision.source and decision.source != self.reference_source:
                log.info("echo reference moved to the narrowest source offered: %s", decision.source)
                self.reference_source = decision.source
        src = gst.ElementFactory.make("pulsesrc")
        convert = gst.ElementFactory.make("audioconvert")
        resample = gst.ElementFactory.make("audioresample")
        capsfilter = gst.ElementFactory.make("capsfilter")
        sink = gst.ElementFactory.make("appsink")
        if not all((src, convert, resample, capsfilter, sink)):
            log.info("reference capture unavailable: GStreamer elements missing")
            return
        src.set_property("device", self.reference_source)
        if src.find_property("latency") is not None:
            # Default PulseAudio latency delivers huge late chunks; a small latency
            # keeps the monitor stream flowing in near-real-time pieces.
            src.set_property("latency", 20)
        capsfilter.set_property("caps", gst.Caps.from_string("audio/x-raw,format=S16LE,channels=1,rate=16000"))
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", self._on_ref_sample, None)
        pipeline = gst.Pipeline.new("voxa-reference")
        for element in (src, convert, resample, capsfilter, sink):
            pipeline.add(element)
        if not src.link(convert) or not convert.link(resample) or not resample.link(capsfilter) or not capsfilter.link(sink):
            pipeline.set_state(gst.State.NULL)
            log.info("reference capture unavailable: could not connect pipeline")
            return
        pipeline.set_state(gst.State.PLAYING)
        self._ref_pipeline = pipeline

    def _start_webrtc(self, gst, device_id: str, on_audio, on_level) -> bool:
        """Build ONE pipeline: the monitor branch feeds the probe, the mic branch runs webrtcdsp.

        With Voxa's own render as the reference (roadmap task 1) the monitor branch is NOT opened at
        all: the probe belongs on the playback branch, where her own samples actually go (handoff 5.6:
        "the monitor branch is not the actual TTS playback branch"). The render tap then registers a
        ``webrtcechoprobe`` under :attr:`echo_probe_name` and the DSP here reads it, so the reference
        the GStreamer canceller uses is Voxa's own playback and never another application's sound.

        Returns True when the pipeline is playing; False means the caller must fall back to the
        plain capture pipeline (never leaving the user without a microphone).
        """
        make = gst.ElementFactory.make
        playback_probe = bool(self.own_reference)
        monitor_src = None if playback_probe else make("pulsesrc")
        # The microphone is opened exactly as the plain capture pipeline opens it.
        selected = next((item for item in self._devices if item.identifier == device_id), None)
        mic_src = selected.device.create_element(None) if selected is not None else make("pulsesrc")
        convert_m, resample_m, caps_m = make("audioconvert"), make("audioresample"), make("capsfilter")
        probe, probe_sink = make("webrtcechoprobe"), make("fakesink")
        convert, resample, capsfilter = make("audioconvert"), make("audioresample"), make("capsfilter")
        dsp, sink = make("webrtcdsp"), make("appsink")
        monitor_branch = () if playback_probe else (monitor_src, convert_m, resample_m, caps_m, probe, probe_sink)
        mic_branch = (mic_src, convert, resample, capsfilter, dsp, sink)
        if not all(monitor_branch + mic_branch):
            log.info("webrtc echo cancellation unavailable: GStreamer elements missing")
            return False
        if playback_probe:
            self.echo_probe_name = self.echo_probe_name or "voxa-echo-probe"
            if self.timeline is not None:
                self.timeline.echo_probe_name = self.echo_probe_name
            dsp.set_property("probe", self.echo_probe_name)
            log.info("webrtc echo cancellation: the probe %s rides Voxa's own playback branch; "
                     "no monitor stream is opened", self.echo_probe_name)
        else:
            monitor_src.set_property("device", self.reference_source)
        for src in (monitor_src, mic_src):
            if src is None:
                continue
            # Default PulseAudio latency delivers huge late chunks; both branches need small ones.
            for key, value in (("latency-time", 20000), ("buffer-time", 40000)):
                if src.find_property(key) is not None:
                    src.set_property(key, value)
        # The probe and the processor must see the same format (verified live: 16 kHz mono S16LE).
        caps = gst.Caps.from_string("audio/x-raw,format=S16LE,layout=interleaved,channels=1,rate=16000")
        capsfilter.set_property("caps", caps)
        if not playback_probe:
            caps_m.set_property("caps", caps)
            probe.set_property("name", "probe")
            probe_sink.set_property("sync", False)
            dsp.set_property("probe", "probe")
        for key, value in (("echo-cancel", True), ("echo-suppression-level", 1), ("noise-suppression", False),
                           ("gain-control", False), ("high-pass-filter", False), ("voice-detection", False),
                           ("limiter", False)):
            if dsp.find_property(key) is not None:
                dsp.set_property(key, value)
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 12)
        if sink.find_property("drop") is not None:
            sink.set_property("drop", True)
        sink.connect("new-sample", self._on_sample, on_level)

        pipeline = gst.Pipeline.new("voxa-webrtc")
        for element in monitor_branch + mic_branch:
            pipeline.add(element)
        linked = all(a.link(b) for branch in (monitor_branch, mic_branch) for a, b in zip(branch, branch[1:], strict=False))
        if not linked:
            pipeline.set_state(gst.State.NULL)
            log.info("webrtc echo cancellation unavailable: could not connect pipeline")
            return False
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        result = pipeline.set_state(gst.State.PLAYING)
        if result == gst.StateChangeReturn.ASYNC:
            # Live sources report NO_PREROLL; anything else still pending gets 3 s to settle.
            result = pipeline.get_state(3 * gst.SECOND)[0]
        if result == gst.StateChangeReturn.FAILURE or result == gst.StateChangeReturn.ASYNC:
            pipeline.set_state(gst.State.NULL)
            log.info("webrtc echo cancellation failed: pipeline did not start (monitor or microphone unavailable)")
            return False
        self._mic_counter = 0
        self._ref_anchor = None
        self._canceller_enabled = False
        self._canceller_logged = False
        self._delay_done = True
        self._delay_mic = []
        self._delay_ref = []
        self._delay_tracker = None  # a fresh capture or mode change is an explicit epoch
        self._ref_ring = None
        self.pipeline = pipeline
        # One AEC owner: the GStreamer DSP owns the echo path for as long as this pipeline lives.
        claim_aec_owner(self, "gst-webrtcdsp")
        return True

    def _on_ref_sample(self, sink, _on_level):
        gst = _ensure_gstreamer()
        sample = sink.emit("pull-sample")
        if sample is None:
            return gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        success, mapped = buffer.map(gst.MapFlags.READ)
        if not success:
            return gst.FlowReturn.ERROR
        try:
            pcm = bytes(mapped.data)
        finally:
            buffer.unmap(mapped)
        if self._ref_ring is not None:
            ref = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
            self._ref_ring.append(ref)
            if self._ref_anchor is None:
                # The monitor only starts producing when the sink wakes; anchor its
                # position 0 to the microphone position of the same moment.
                self._ref_anchor = self._mic_counter - self._ref_ring.total
        return gst.FlowReturn.OK

    def _on_sample(self, sink, on_level):
        gst = _ensure_gstreamer()
        sample = sink.emit("pull-sample")
        if sample is None:
            return gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        success, mapped = buffer.map(gst.MapFlags.READ)
        if not success:
            return gst.FlowReturn.ERROR
        try:
            pcm = bytes(mapped.data)
        finally:
            buffer.unmap(mapped)
        # The moment this buffer reached us: the capture clock turns it into the moment the
        # microphone sampled the first of these samples (roadmap task 1).
        self._last_arrival = time.monotonic()

        if self.canceller is not None and self._canceller_enabled:
            self._maybe_estimate_delay(pcm)
        pcm = self._mix_or_passthrough(pcm)

        level = self._rms(pcm)
        try:
            if self._on_audio is not None:
                self._on_audio(pcm, level)
            now = time.monotonic()
            if on_level is not None and now - self._last_level_emit >= 0.05:
                self._last_level_emit = now
                on_level(level)
        except Exception as exc:  # noqa: BLE001 - callback boundary
            if self._on_error is not None:
                self._on_error(str(exc))
            return gst.FlowReturn.ERROR
        return gst.FlowReturn.OK

    def _reference_window(self, samples: int) -> np.ndarray | None:
        """The reference samples the microphone chunk of ``samples`` could be hearing.

        With a render timeline (roadmap task 1) the answer comes from the clock: the buffer's arrival
        time minus the capture latency is the moment its first sample was captured, and the timeline
        says which of her own rendered samples were audible then. Without one the ring is used, aligned
        by the anchor the monitor's first sample happened to land on.
        """
        if (self.timeline is not None and self._last_arrival is not None
                and self.timeline.total > 0):
            mic_start, _end = self.capture_clock.stamp(self._last_arrival, samples)
            return self.timeline.reference_window(mic_start=mic_start, length=samples)
        if self._ref_ring is None:
            return None
        return self._ref_ring.window(self._mic_counter - (self._ref_anchor or 0), samples)

    def _mix_or_passthrough(self, pcm: bytes) -> bytes:
        """Clean ``pcm`` against the reference when a canceller is active; never break recording."""
        if self.canceller is None or not self._canceller_enabled:
            self._mic_counter += len(pcm) // 2
            return pcm
        window = self._reference_window(len(pcm) // 2)
        if window is None:
            self._mic_counter += len(pcm) // 2
            return pcm
        try:
            if self.reference_via() == "timeline":
                cleaned = clean_chunk(pcm, window, self.canceller)
            else:
                cleaned = mix_chunk(pcm, self._ref_ring, self._mic_counter, self.canceller,
                                    self._ref_anchor or 0)
        except Exception as exc:  # noqa: BLE001 - recording must never break
            if not self._canceller_logged:
                log.warning("echo canceller failed; using raw microphone: %s", exc)
                self._canceller_logged = True
            self._canceller_enabled = False
            self._mic_counter += len(pcm) // 2
            return pcm
        self._mic_counter += len(pcm) // 2
        return cleaned

    def _maybe_estimate_delay(self, pcm: bytes) -> None:
        """Track the echo delay and the clock drift continuously (roadmap task 3).

        A GCC-PHAT over the first 0.5 s of every TTS utterance gives the coarse offset,
        re-estimated every 0.5 s after that: the slope of the estimates is the clock drift in
        ppm (above 50 ppm the drift flag latches and the interpolated offsets resample the
        reference), and an abrupt jump is an explicit epoch the drift never sees. The
        canceller NEVER rebuilds per sentence: only its reference offset is re-pointed, its
        filter, convergence and double-talk state carry across utterances."""
        if self._delay_done:
            return
        if getattr(self.canceller, "delay_managed_internally", False):
            # AEC3 estimates the delay itself and does not report one; the flag it carries
            # (apply_clock_drift) and its own estimator stay in charge. The estimator here
            # is done: nothing to build, AEC3 is never replaced by the NLMS canceller.
            self._delay_done = True
            return
        if self._delay_tracker is None:
            self._delay_tracker = DelayTracker(sample_rate=REFERENCE_RATE)
        window = self._reference_window(len(pcm) // 2)
        if window is None:
            return
        mic = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        update = self._delay_tracker.observe(mic, np.asarray(window, dtype=np.float32))
        if update is None:
            return
        self.canceller.set_offset(update["delay_samples"], *update["anchors"])
        if update["kind"] != "track":
            log.info("echo delay %s: %d samples (%.3f s), drift %.1f ppm", update["kind"],
                     update["delay_samples"], update["delay_samples"] / REFERENCE_RATE,
                     update["drift_ppm"])

    @staticmethod
    def _rms(pcm: bytes) -> float:
        samples = np.frombuffer(pcm, dtype="<i2")
        if not samples.size:
            return 0.0
        stride = max(1, samples.size // 2048)
        chosen = samples[::stride].astype(np.float64)
        return float(np.sqrt(np.mean(chosen * chosen)))

    def _on_bus_error(self, _bus, message) -> None:
        error, debug = message.parse_error()
        detail = f"{error.message}"
        if debug:
            detail = f"{detail} ({debug})"
        if self._on_error is not None:
            self._on_error(detail)
        self.stop()

    @property
    def is_active(self) -> bool:
        """True while a capture pipeline exists, i.e. the microphone may be recording."""
        return self.pipeline is not None

    def reference_power(self, samples: int = 1600) -> float:
        """Power of the reference over the most recent ~100 ms (0.0 with no reference source).

        Attribution evidence, read-only: "is the speaker path live at this moment" is handoff 6.3's
        first stream. The window is the canceller's own block size scale, so a reply that stopped
        a moment ago still counts while its echo is in flight.
        """
        window = self._reference_window(max(1, int(samples)))
        if window is None:
            return 0.0
        return float(np.mean(np.asarray(window, dtype=np.float64) ** 2)) if len(window) else 0.0

    def echo_measurements(self) -> tuple[float, float | None, float | None]:
        """(reference_power, echo_explained_db, residual_db) of the last block the canceller fitted.

        ``None`` for the two dB fields means this path has no echo model to measure with (echo off,
        or no reference sink): the attribution decision then says so instead of guessing.
        """
        if self.canceller is None or not self._canceller_enabled:
            return self.reference_power(), None, None
        return self.reference_power(), self.canceller.echo_explained_db, self.canceller.residual_power_db

    @property
    def delay_samples(self) -> int:
        """The delay the echo path was fitted with (0 = never estimated: alignment unhealthy)."""
        return self.canceller.delay_samples if self.canceller is not None else 0

    def stop(self) -> None:
        gst = _ensure_gstreamer() if (self.pipeline is not None or self._ref_pipeline is not None) else None
        pipeline, self.pipeline = self.pipeline, None
        ref_pipeline, self._ref_pipeline = self._ref_pipeline, None
        if pipeline is not None:
            if gst is not None:
                pipeline.set_state(gst.State.NULL)
            # The GStreamer DSP route owned the echo path while that pipeline lived.
            release_aec_owner(self)
        if ref_pipeline is not None and gst is not None:
            ref_pipeline.set_state(gst.State.NULL)

    def set_echo_mode(self, mode: str | bool = "voxa", reference_source: str | None = None) -> None:
        """Switch echo cancellation mode ("voxa", "aec3", "webrtc" or "off"), applying immediately if capture is running.

        A boolean is accepted for the old call style: ``True`` maps to "voxa", ``False`` to "off".
        "aec3" is WebRTC AEC3 in this process (roadmap task 2); without the binding it falls back to
        Voxa's own NLMS canceller and says so, exactly as "webrtc" falls back to "off".
        """
        if isinstance(mode, bool):
            mode = "voxa" if mode else "off"
        if mode == "aec3" and not aec3_available():
            log.info("AEC3 echo cancellation unavailable (pywebrtc-audio not installed): "
                     "using Voxa's own canceller")
            mode = "voxa"
        if mode == "webrtc" and not webrtc_available():
            log.info("webrtc echo cancellation unavailable: running with echo off")
            mode = "off"
        if mode == "webrtc":
            self.pulse_source = None
            self.reference_source = reference_source
            release_aec_owner(self.canceller)
            self.canceller = None
            self._canceller_enabled = False
            self._canceller_logged = False
            self._delay_done = True
            self._delay_mic = []
            self._delay_ref = []
            self._delay_tracker = None  # a mode change is an explicit epoch
            self._ref_ring = None
            self._ref_anchor = None
            if self._ref_pipeline is not None:
                gst = _ensure_gstreamer()
                self._ref_pipeline.set_state(gst.State.NULL)
                self._ref_pipeline = None
            if self.is_active and self._start_args is not None:
                device_id, on_audio, on_level, on_error = self._start_args
                self.stop()
                self.start(device_id, on_audio, on_level, on_error)
            return
        if mode not in ("voxa", "aec3"):
            self.pulse_source = None
            self.reference_source = None
            self.canceller = None
            self._canceller_enabled = False
            self._delay_done = True
            self._delay_mic = []
            self._delay_ref = []
            self._delay_tracker = None  # a mode change is an explicit epoch
            self._ref_ring = None
            self._ref_anchor = None
            if self._ref_pipeline is not None:
                gst = _ensure_gstreamer()
                self._ref_pipeline.set_state(gst.State.NULL)
                self._ref_pipeline = None
            return
        self.pulse_source = None
        self.reference_source = reference_source
        wanted = Aec3Canceller if mode == "aec3" else EchoCanceller
        if not isinstance(self.canceller, wanted):
            # Exactly one canceller owns the echo path: give the old one up before the new one claims it.
            release_aec_owner(self.canceller)
            self.canceller = wanted()
        self._canceller_enabled = True
        self._canceller_logged = False
        self.apply_clock_drift()
        if not self.is_active:
            self._delay_done = False
            self._delay_mic = []
            self._delay_ref = []
            self._delay_tracker = None  # a mode change is an explicit epoch
            self._ref_ring = None
            return
        if self._ref_ring is None:
            self._ref_ring = ReferenceRing(REFERENCE_SECONDS * REFERENCE_RATE)
        if self._ref_pipeline is None and reference_source:
            gst = _ensure_gstreamer()
            self._start_reference(gst)
        self._delay_done = False
        self._delay_mic = []
        self._delay_ref = []
        self._delay_tracker = None  # a fresh capture or mode change is an explicit epoch
