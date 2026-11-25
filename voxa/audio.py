from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import anccal
from .align import ExactDelayLock
from .anc import (Aec3Canceller, EchoCanceller, MIN_REFERENCE_POWER, aec3_available,
                  claim_aec_owner, erle_db, estimate_delay, release_aec_owner)
from .echo import OUTPUT_SPEAKERS, DeviceWatcher, OutputSnapshot, output_tuning
from .refmask import LockedResidualMask
from .timeline import CaptureClock, RenderTimeline
from .dtln import DtlnCanceller, dtln_available

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
# Roadmap task 5 (f173): the barge-in joint gate reads the cleaned stream the capture produced.
# The capture thread only STORES the frames (the canceller/mask already ran); the model (Silero)
# and the coherence run on the hearing worker, which drains this bounded ring. 100 x 20 ms = 2 s.
JOINT_FRAMES = 100


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
        self._delay_lock: ExactDelayLock | None = None
        # Roadmap task 4 on the exact-delay output (f171): the residual stage ON the shipping
        # canceller's own output, read at the delay the lock drives the canceller at. None =
        # the mode has no residual stage (echo off, the webrtc route, or AEC3, whose delay is
        # its own and was never handed to a lock).
        self._residual_mask: LockedResidualMask | None = None
        self._delay_lock_logged = False   # the exact lock's first success: logged once
        self._delay_degraded_logged = False  # the fallback estimate taking over: logged once
        # Roadmap task 5 (f173): the joint double-talk gate's inputs, staged by the CAPTURE thread
        # and consumed by the HEARING WORKER. The capture never runs a model here: it stores the
        # cleaned/mic/reference bytes plus the delay, band margin and ERLE the canceller and the
        # residual mask already measured. The hearing worker drains this bounded ring, runs the
        # cleaned-stream Silero and the DoubleTalkGate coherence, and returns the joint verdicts to
        # window._hearing_evidence (handoff 6.8: no inference in the audio callback, none on GTK).
        self._joint_frames: deque[tuple] = deque(maxlen=JOINT_FRAMES)
        self._joint_lock = threading.Lock()
        self._joint_dropped = 0
        self._start_args: tuple | None = None
        # Roadmap task 2: the device the microphone is on and the device the sound goes out on.
        # Different devices mean different clocks, which is what AEC3's drift handling wants to know.
        self._mic_identifier: str | None = None
        self._sink_identifier: str | None = None
        # The GStreamer route: the name of the webrtcechoprobe the playback tap registers (task 2).
        self.echo_probe_name: str | None = None
        # Roadmap task 10: what the sound goes out on, and the epoch it was learned under.
        # A device, headphone or Bluetooth-profile change is an EXPLICIT epoch: the reference
        # history, the canceller and the latency measurement all restart together, because the
        # acoustic path (and its codec buffering) is a different one. Never per sentence.
        self._echo_mode = "off"
        self.output_class: str = OUTPUT_SPEAKERS
        self.tuning = output_tuning(self.output_class)
        self.epoch = 0
        self._recovery_until = 0.0
        self._device_watcher: DeviceWatcher | None = None
        # The live capture source, kept so an epoch can re-read the driver's own latency.
        self._capture_src: Any = None
        # The speaker/microphone pair Voxa is hearing through (see voxa/anccal.py): the saved profile
        # for that pair is her starting filter, and what she learns during use is saved back under it.
        self.hearing_sink: str | None = None
        self.hearing_microphone: str | None = None
        self.hearing_profile_applied = False
        self.hearing_profile_stale = False
        self.hearing_needs_calibration = False
        self.hearing_level = 0.0

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
        self._capture_src = source  # an epoch re-reads this driver's latency (task 10)
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
        # Start from the hearing profile saved for this speaker and microphone, when there is one.
        self.apply_hearing_profile(self.reference_source, device_id)
        self._delay_done = self.canceller is None
        self._delay_mic = []
        self._delay_ref = []
        self._delay_lock = None  # a fresh capture or mode change is an explicit epoch
        self._residual_mask = (LockedResidualMask(rate=REFERENCE_RATE)
                               if isinstance(self.canceller, EchoCanceller)
                               else None)  # the residual stage re-learns the path with the capture
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

    # --- roadmap task 10: device, headphone and Bluetooth changes as explicit epochs -------

    def start_device_watch(self, *, poll_seconds: float = 2.0, runner: Any = None) -> None:
        """Watch the user's output device and react to changes with an epoch (roadmap task 10).

        Read-only: the watcher only asks PulseAudio/PipeWire what the default sink, source and
        Bluetooth profile ARE; Voxa never switches a profile, a port or the default device itself
        (WirePlumber's autoswitch to the headset profile is an event to REACT to). Safe to call
        twice, never raises: no daemon, no watcher, and the pipeline keeps its current epoch.
        """
        if self._device_watcher is not None:
            return None
        try:
            watcher = DeviceWatcher(self._on_device_change, runner=runner or subprocess.run,
                                    poll_seconds=poll_seconds, log=log)
        except Exception:  # noqa: BLE001 - a machine with no audio at all must still run
            log.info("device watcher unavailable: audio stays on its current epoch")
            return None
        self._device_watcher = watcher
        # One synchronous read first: the very first output is not an epoch, it is what this
        # capture is, so the tuning (and AEC3's delay seed) is right from the first sentence.
        watcher.poll_once()
        if watcher.snapshot is not None:
            self.note_output(watcher.snapshot.output_class)
        watcher.start()
        return None

    def stop_device_watch(self) -> None:
        watcher, self._device_watcher = self._device_watcher, None
        if watcher is not None:
            watcher.stop()

    def _on_device_change(self, change: dict) -> None:
        """The watcher's callback: one device change, one explicit epoch."""
        snap: OutputSnapshot | None = change.get("to")
        reason = str(change.get("reason") or "device-change")
        self.begin_epoch(getattr(snap, "output_class", None), reason=reason)

    def note_output(self, output_class: str | None, *, reason: str = "startup") -> str:
        """Learn WHICH output this is without an epoch (the watcher's first read).

        The very first read has nothing to recover from, and opening a recovery window there would
        hold the user's first request for no reason. The tuning still applies, because the AEC3
        delay seed below is chosen when the canceller is created.
        """
        if not output_class or output_class == self.output_class:
            return self.output_class
        self.output_class = output_class
        self.tuning = output_tuning(output_class)
        log.info("audio output %s (%s): delay search %.0f ms, residual margin %.1f dB",
                 output_class, reason, self.tuning.delay_search_ms, self.tuning.residual_margin_db)
        return output_class

    def echo_path_qualified(self) -> bool:
        """True when the echo path can say what the microphone's sound is made of RIGHT NOW.

        False is "unqualified audio": no delay estimate yet, or no canceller at all. Task 10's
        recovery rule holds automatic commands off exactly that, and keeps the audio."""
        canceller = self.canceller
        if canceller is None or not self._canceller_enabled:
            return False
        if getattr(canceller, "delay_managed_internally", False):
            return True  # AEC3 owns the alignment: it reports nothing because it is always searching
        return self.delay_samples > 0

    @property
    def recovering(self) -> bool:
        """True inside an epoch's recovery window (handoff 6.2's recovery process).

        During recovery the pipeline keeps every sample of audio - nothing is thrown away - but
        unqualified audio may not raise an automatic command: the canceller has not converged on the
        new path yet, and a decision made from audio it cannot explain is how a device change turns
        into a self-interruption.
        """
        return time.monotonic() < self._recovery_until

    def begin_epoch(self, output_class: str | None = None, *, reason: str = "device-change") -> int:
        """One explicit epoch: flush the reference, recreate the canceller, re-measure latency.

        Roadmap task 10 and handoff 6.2: a route, device or Bluetooth-profile change is the ONLY
        thing that reaches here (the canceller is never rebuilt per sentence, and the delay tracker
        re-estimates the delay without ever touching the filter). Everything acoustic restarts
        together: the reference history, the canceller instance, the latency measurement and the
        task-5 thresholds that belong to this output (``self.tuning``). The recovery window that
        opens holds automatic commands off unqualified audio while it converges.
        """
        self.epoch += 1
        if output_class and output_class != self.output_class:
            self.output_class = output_class
        self.tuning = output_tuning(self.output_class)
        log.info("audio epoch %d (%s): output %s, echo path %s, delay search %.0f ms, "
                 "residual margin %.1f dB", self.epoch, reason, self.output_class,
                 "present" if self.tuning.echo_path_present else "absent",
                 self.tuning.delay_search_ms, self.tuning.residual_margin_db)
        # 1. Flush the reference history: samples rendered into the OLD device's clock say nothing
        #    about this one. Both reference kinds flush together (task 1's timeline and the ring).
        if self.timeline is not None:
            self.timeline.new_epoch(reason)
        if self._ref_ring is not None:
            self._ref_ring = ReferenceRing(REFERENCE_SECONDS * REFERENCE_RATE)
        self._ref_anchor = None
        # 2. Re-measure the latency before the first chunk of the new path is aligned.
        self._remeasure_latency()
        # 3. Recreate the canceller for the new path (one owner: the old one gives the path up
        #    first), and restart the delay estimate - the delay itself is what changed.
        self._recreate_canceller()
        # The chain's delay machinery: an epoch is the only reset the learned delay ever gets
        # (task 3 re-estimates, never rebuilds). ExactDelayLock holds the lock AND the fallback
        # estimate, so dropping the lock drops both; ``_delay_tracker`` is the f137-era name of
        # that estimate, cleared with it. The residual stage starts ungifted too: no delay of
        # the old path may be handed to it.
        self._delay_lock = None
        self._delay_tracker = None
        self._residual_mask = (LockedResidualMask(rate=REFERENCE_RATE)
                               if self._echo_mode == "voxa" else None)
        self._delay_mic, self._delay_ref = [], []
        self._delay_done = self.canceller is None
        # 4. The recovery window (handoff 6.2): no automatic commands from unqualified audio.
        self._recovery_until = time.monotonic() + self.tuning.recovery_s
        return self.epoch

    def _recreate_canceller(self) -> None:
        """Build a fresh canceller of the mode the capture path asked for, with this output's delay."""
        mode = self._echo_mode
        old, self.canceller = self.canceller, None
        if old is not None:
            release_aec_owner(old)
            with contextlib.suppress(Exception):
                old.close()  # AEC3 exposes close(); the NLMS path does not
        if mode == "webrtc":
            # The GStreamer DSP owns this pipeline's echo path: an epoch restarts the pipeline
            # instead (``start`` re-runs it), which is also what re-opens its probe.
            self.canceller = None
            return
        try:
            if mode == "aec3":
                # A2DP/HFP buffer hundreds of milliseconds on top of the room: seed AEC3 with the
                # delay its estimator may expect (task 10's table). A normal output keeps today's
                # blind start, because a wrong seed costs more than it saves.
                self.canceller = Aec3Canceller(stream_delay_ms=int(self.tuning.aec3_seed_ms))
            elif mode in ("voxa", "off"):
                self.canceller = EchoCanceller()
            else:
                self.canceller = None
                return
        except Exception as exc:  # noqa: BLE001 - the microphone must survive a canceller failure
            log.warning("echo canceller could not restart (%s): raw microphone, degraded state", exc)
            self.canceller = None
        self._canceller_enabled = self.canceller is not None
        self._canceller_logged = False
        self.apply_clock_drift()

    def _remeasure_latency(self) -> float | None:
        """Read the live capture latency again (an epoch may have changed the driver's buffer)."""
        current = self.capture_clock.latency
        if callable(current):
            # The app handed us a live question, so ask it again: a Bluetooth profile changes the
            # answer, and this is exactly why set_capture_latency accepts a callable.
            try:
                value = float(current())
            except Exception:  # noqa: BLE001 - the source may be gone mid-switch
                return None
            log.debug("capture latency re-measured: %.1f ms", value * 1000.0)
            return value
        source = self._capture_src
        if source is not None:
            # The PulseAudio/PipeWire source driver tells its own latency in microseconds, the same
            # property _start_reference sets on the reference side.
            with contextlib.suppress(Exception):
                if source.find_property("latency") is not None:
                    microseconds = float(source.get_property("latency"))
                    self.capture_clock.latency = max(0.0, microseconds / 1_000_000.0)
                    log.debug("capture latency re-measured from the source: %.1f ms",
                              self.capture_clock.latency * 1000.0)
                    return self.capture_clock.latency
        return current if isinstance(current, (int, float)) else None

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

    def _microphone_identity(self) -> str | None:
        """The microphone Voxa listens with, as her settings and the running capture name it."""
        if self.hearing_microphone:
            return self.hearing_microphone
        return self._start_args[0] if self._start_args is not None else None

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
        self._capture_src = mic_src  # an epoch re-reads this driver's latency (task 10)
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
        self._delay_lock = None  # a fresh capture or mode change is an explicit epoch
        self._residual_mask = None  # the GStreamer DSP owns the whole echo path, stage included
        self._reset_joint_frames()  # the webrtc route feeds no joint frames: drop any NLMS residue
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
            self._note_mic_level(pcm)
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
        cleaned = self._residual_stage(cleaned, pcm, window)
        # f173: stage the frame the joint double-talk gate will read on the hearing worker. No
        # model runs here: the bytes plus the delay/margin/ERLE the canceller and mask measured.
        self._note_joint_frame(cleaned, pcm, window)
        self._mic_counter += len(pcm) // 2
        return cleaned

    def _note_mic_level(self, pcm: bytes) -> None:
        """Remember the loudest microphone level she is learning at, for :func:`anccal.profile_is_current`."""
        samples = np.frombuffer(pcm, dtype="<i2")
        if not samples.size:
            return
        peak = float(np.max(np.abs(samples.astype(np.float64)))) / 32768.0
        if peak > self.hearing_level:
            self.hearing_level = peak

    def _note_joint_frame(self, cleaned: bytes, mic_bytes: bytes, ref: np.ndarray) -> None:
        """Roadmap task 5 (f173): park one cleaned frame for the barge-in joint gate.

        The capture thread has no business running a model (handoff 6.8), so this only keeps the
        three sample streams the DoubleTalkGate needs plus the scalars the shipping chain already
        measured: the delay the lock drove the canceller at, the residual mask's per-band margin
        over its OWN prediction at that delay, and the running ERLE of the cleaned output. The
        hearing worker drains the ring (``drain_joint_frames``), runs the cleaned-stream Silero and
        the gate there, and hands the joint verdicts back to ``window._hearing_evidence``.
        """
        if self.canceller is None or not self._canceller_enabled:
            return
        try:
            mic = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float64) / 32768.0
            out = np.frombuffer(cleaned, dtype="<i2").astype(np.float64) / 32768.0
            erle = erle_db(mic, out)
            mask = self._residual_mask
            margin = (float(mask.band_margin_db)
                      if mask is not None and np.isfinite(mask.band_margin_db) else None)
            delay = int(self.canceller.delay_samples)
            # The gate's coherence half reads only where the lock PROVED the frame (the exact
            # render-timeline delay), never a drifting estimate - f132's blocker, f168 shipped the
            # lock. AEC3 owns its delay internally, so it always counts as healthy.
            align_healthy = bool(self.delay_confident
                                 or getattr(self.canceller, "delay_managed_internally", False))
            ref_power = float(np.mean(np.asarray(ref, dtype=np.float64) ** 2))
            ref_active = bool(ref_power > MIN_REFERENCE_POWER)
            with self._joint_lock:
                if len(self._joint_frames) == self._joint_frames.maxlen:
                    self._joint_dropped += 1
                self._joint_frames.append((cleaned, mic_bytes,
                                           np.asarray(ref, dtype=np.float64),
                                           delay if delay > 0 else None, margin, erle,
                                           align_healthy, ref_active))
        except Exception as exc:  # noqa: BLE001 - the tap must never break the recording
            log.debug("joint-evidence tap skipped: %s", exc)

    def drain_joint_frames(self) -> list[tuple]:
        """Take every frame staged since the last drain (the hearing worker consumes them).

        The capture appends under the lock and the worker drains it whole, so each frame is fed to
        the gate exactly once and contiguously; a full ring drops the oldest (a counted drop, the
        degraded path), never a silent loss of the start of a turn.
        """
        with self._joint_lock:
            frames = list(self._joint_frames)
            self._joint_frames.clear()
        return frames

    def _reset_joint_frames(self) -> None:
        """A capture/mode epoch is a new echo path: drop frames staged under the old one."""
        with self._joint_lock:
            self._joint_frames.clear()

    def _residual_stage(self, cleaned: bytes, mic_bytes: bytes, ref: np.ndarray) -> bytes:
        """Roadmap task 4 on the EXACT-DELAY output (task f171, issue #72).

        The residual suppressor runs ON THE CANCELLER'S OWN OUTPUT with the same reference
        window the canceller just heard: it predicts the residue the linear stage left per band
        (:class:`voxa.refmask.LockedResidualMask`, the f131 machinery with the delay the lock
        proved handed to it by :meth:`voxa.refmask.ReferenceMask.set_delay`, so the reference it
        reads and the canceller's residue are the SAME pair). The canceller is unchanged and
        stays the only AEC owner: the mask is a spectral gain on its output, not a second
        canceller, and it suppresses nothing outside TTS plus its echo tail (the mask's own
        ``tail_s`` rule) - a frame it cannot explain is kept, never deleted.

        A mask failure degrades the stage alone, visibly: the cleaned stream still carries the
        canceller's verdict, exactly as it did before this task.
        """
        if self._residual_mask is None:
            return cleaned
        try:
            mic = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float64) / 32768.0
            out = np.frombuffer(cleaned, dtype="<i2").astype(np.float64) / 32768.0
            # The running ERLE of the LINEAR canceller: where it removed little, the mask
            # subtracts deeper (roadmap task 4's over-subtraction ramp).
            self._residual_mask.set_running_erle(erle_db(mic, out))
            masked = np.clip(self._residual_mask.process(out, np.asarray(ref, dtype=np.float64)),
                             -1.0, 1.0)
            if masked.size == 0:
                return cleaned
            masked = np.clip(np.rint(masked * 32767.0), -32768, 32767)
            return masked.astype("<i2").tobytes()
        except Exception as exc:  # noqa: BLE001 - recording must never break
            log.warning("residual echo stage failed; keeping the canceller's output: %s", exc)
            self._residual_mask = None
            return cleaned

    def _maybe_estimate_delay(self, pcm: bytes) -> None:
        """Track the echo delay and the clock drift continuously (roadmap task 3), through
        the exact lock issue #72/task f168 shipped: the render timeline's :class:`ExactAligner`
        lock (roadmap Hearing 2) drives the canceller's reference while it is proven, and the
        task-3 estimate (:class:`voxa.anc.DelayTracker`) keeps it pointed while it is not -
        visibly degraded, never silently.

        The GCC-PHAT fallback over the first 0.5 s of every TTS utterance gives the coarse
        offset, re-estimated every 0.5 s after that: the slope of the estimates is the clock
        drift in ppm (above 50 ppm the drift flag latches and the interpolated offsets resample
        the reference), and an abrupt jump is an explicit epoch the drift never sees. The
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
        if self._delay_lock is None:
            self._delay_lock = ExactDelayLock(
                sample_rate=REFERENCE_RATE, canceller=self.canceller)
        window = self._reference_window(len(pcm) // 2)
        if window is None:
            return
        mic = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        update = self._delay_lock.observe(
            self._mic_counter / REFERENCE_RATE, mic, np.asarray(window, dtype=np.float32))
        if update == "lock" and self._residual_mask is not None:
            # f171 (roadmap task 4 on the exact-delay output): the mask reads its reference at
            # the SAME delay the canceller was just driven at, so every band ratio, every held
            # tail and the 6 dB near-end rule all see the pair the lock proved. Before the
            # first lock nothing is handed and the mask estimates its own lag (the pre-f169
            # behaviour); a degraded frame never hands the mask a delay the canceller is NOT
            # using, so it never passes a verdict measured against an unmeasured alignment.
            self._residual_mask.set_delay(float(self._delay_lock.delay_frac))
        if update == "lock":
            if not self._delay_lock_logged:
                log.info("echo delay: exact lock at %.3f ms (drift %.1f ppm)",
                         float(self._delay_lock.delay_frac) / REFERENCE_RATE * 1000.0,
                         self._delay_lock.drift_ppm)
                self._delay_lock_logged = True
        elif update == "estimate":
            if not self._delay_degraded_logged:
                log.info("echo delay: exact lock not yet confident; the estimated delay "
                         "drives the reference (degraded: lock unproven, never raw audio)")
                self._delay_degraded_logged = True
            est = self._delay_lock.tracker.last_update
            if est is not None and est["kind"] != "track":
                log.info("echo delay %s: %d samples (%.3f s), drift %.1f ppm", est["kind"],
                         est["delay_samples"], est["delay_samples"] / REFERENCE_RATE,
                         est["drift_ppm"])

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

        Issue #72 / task v08: while one of her turns is LIVE this read is the turn's own render
        (voxa/timeline.py ``turn_reference_window``) - continuous across the gaps between the
        pieces of a reply and over the echo tail - so a mid-reply moment never reads silent.
        That reading used to credit her own voice to the user, which is how she answered herself.
        """
        window = self._attribution_reference_window(max(1, int(samples)))
        if window is None:
            return 0.0
        return float(np.mean(np.asarray(window, dtype=np.float64) ** 2)) if len(window) else 0.0

    def _attribution_reference_window(self, samples: int) -> np.ndarray | None:
        """The reference window for ATTRIBUTION: the canceller keeps the exact-aligned read.

        Falls back to :meth:`_reference_window` when no render timeline is answering, or when no
        turn of hers is live (a silent reference is then honest: nothing is playing).
        """
        timeline = self.timeline
        if timeline is None or self._last_arrival is None or timeline.total <= 0:
            return self._reference_window(samples)
        mic_start, _end = self.capture_clock.stamp(self._last_arrival, samples)
        if not timeline.turn_is_live(mic_start):
            return timeline.reference_window(mic_start=mic_start, length=samples)
        return timeline.turn_reference_window(moment=mic_start, length=samples)

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

    @property
    def delay_frac(self) -> float:
        """The fractional reference read offset the lock holds (0.0 = never locked)."""
        return float(self._delay_lock.delay_frac) if (
            self._delay_lock is not None and self._delay_lock.delay_frac is not None) else 0.0

    @property
    def delay_confident(self) -> bool:
        """True only while the exact lock has proven itself (never on the fallback estimate)."""
        return bool(self._delay_lock is not None and self._delay_lock.delay_confident)

    @property
    def delay_degraded(self) -> bool:
        """True while the reference is driven by the estimate because the lock is unproven."""
        return bool(self._delay_lock is not None and self._delay_lock.degraded)

    def stop(self) -> None:
        self._save_hearing_profile()
        self.stop_device_watch()
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

    def _save_hearing_profile(self) -> None:
        """Save the filter the running canceller learned, under this speaker/microphone pair.

        Only a filter that actually did work is worth remembering: it must have adapted on real
        reference material and removed at least 3 dB of the computer's own sound.
        """
        canceller = self.canceller
        if canceller is None or not self.hearing_sink or not self.hearing_microphone:
            return
        if not self.hearing_profile_applied and getattr(canceller, "_blocks_adapted", 0) <= 0:
            return
        if getattr(canceller, "converged_db", 0.0) < 3.0:
            return
        export = getattr(canceller, "export_state", None)
        if export is None:
            return
        try:
            anccal.save_live_state(export(), getattr(canceller, "_delay", 0), self.hearing_sink,
                                   self.hearing_microphone, self.hearing_level)
        except (OSError, ValueError, TypeError):
            log.info("could not save the hearing profile for %s/%s", self.hearing_sink,
                     self.hearing_microphone)
            return
        log.info("saved the hearing profile for %s/%s (%.1f dB of the computer's own sound removed)",
                 self.hearing_sink, self.hearing_microphone, canceller.converged_db)

    def apply_hearing_profile(self, sink: str | None, source: str | None,
                              level: float | None = None) -> bool:
        """Start the canceller from the hearing profile saved for this speaker/microphone pair.

        Returns True when the saved filter was loaded, False when echo cancellation is off, when no
        profile exists for the pair, or when the saved one belongs to another speaker or a very
        different volume (then Voxa adapts from scratch and sets :attr:`hearing_profile_stale`).
        Either way the pair is remembered, so what the running canceller learns can be saved back
        by :meth:`stop`.
        """
        if not sink or not source:
            return False
        self.hearing_sink, self.hearing_microphone = sink, source
        if level is not None:
            self.hearing_level = max(self.hearing_level, float(level))
        if self.canceller is None:
            return False
        profile = anccal.load_profile(sink, source)
        if profile is None:
            return False
        canceller = EchoCanceller(delay=int(profile.get("delay", 0) or 0))
        state = profile.get("state")
        if isinstance(state, dict):
            canceller.load_state(state)
        if canceller.weights is None or not anccal.profile_is_current(profile, sink, source, level):
            self.hearing_profile_stale = True
            log.info("saved profile is for a different speaker or volume; adapting from scratch")
            return False
        self.canceller = canceller
        self.hearing_profile_applied = True
        log.info("starting from the hearing profile saved for %s/%s", sink, source)
        return True

    def set_echo_mode(self, mode: str | bool = "voxa", reference_source: str | None = None) -> None:
        """Switch echo cancellation mode ("voxa", "aec3", "webrtc" or "off"), applying immediately if capture is running.

        A boolean is accepted for the old call style: ``True`` maps to "voxa", ``False`` to "off".
        "aec3" is WebRTC AEC3 in this process (roadmap task 2); without the binding it falls back to
        Voxa's own NLMS canceller and says so, exactly as "webrtc" falls back to "off".
        """
        if isinstance(mode, bool):
            mode = "voxa" if mode else "off"
        if mode == "dtln" and not dtln_available()[0]:
            # Hearing 2 h04: the neural stage is the default, but a machine without onnxruntime or
            # the shipped weights falls back to Voxa's own canceller and says so exactly once.
            log.info("DTLN echo stage unavailable (%s): using Voxa's own canceller", dtln_available()[1])
            mode = "voxa"
        if mode == "aec3" and not aec3_available():
            log.info("AEC3 echo cancellation unavailable (pywebrtc-audio not installed): "
                     "using Voxa's own canceller")
            mode = "voxa"
        if mode == "webrtc" and not webrtc_available():
            log.info("webrtc echo cancellation unavailable: running with echo off")
            mode = "off"
        self._echo_mode = mode  # an epoch (task 10) recreates THIS mode's canceller
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
            self._delay_lock = None  # a mode change is an explicit epoch
            self._residual_mask = None  # no canceller output to put a stage on
            self._reset_joint_frames()
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
        if mode not in ("voxa", "aec3", "dtln"):
            self.pulse_source = None
            self.reference_source = None
            self.canceller = None
            self._canceller_enabled = False
            self._delay_done = True
            self._delay_mic = []
            self._delay_ref = []
            self._delay_lock = None  # a mode change is an explicit epoch
            self._residual_mask = None  # no canceller output to put a stage on
            self._reset_joint_frames()
            self._ref_ring = None
            self._ref_anchor = None
            if self._ref_pipeline is not None:
                gst = _ensure_gstreamer()
                self._ref_pipeline.set_state(gst.State.NULL)
                self._ref_pipeline = None
            return
        self.pulse_source = None
        self.reference_source = reference_source
        wanted = Aec3Canceller if mode == "aec3" else (
            DtlnCanceller if mode == "dtln" else EchoCanceller)
        if not isinstance(self.canceller, wanted):
            # Exactly one canceller owns the echo path: give the old one up before the new one claims it.
            release_aec_owner(self.canceller)
            if mode == "aec3":
                # Task 10: a Bluetooth link buffers hundreds of ms on top of the room, and the
                # output is already known here - seed the estimator instead of starting blind.
                self.canceller = wanted(stream_delay_ms=int(self.tuning.aec3_seed_ms))
            else:
                self.canceller = wanted()
            self.apply_hearing_profile(self.reference_source, self.hearing_microphone)
        # f171 (roadmap task 4 on the exact-delay output): the residual stage rides on THIS
        # canceller's output - it is a spectral gain, not a second AEC owner. A mode change is
        # an explicit epoch, so the stage starts fresh with it: no delay handed yet, and the
        # mask estimates its own lag until the exact lock proves one and hands it over.
        self._residual_mask = (LockedResidualMask(rate=REFERENCE_RATE)
                               if mode == "voxa" else None)
        self._canceller_enabled = True
        self._canceller_logged = False
        self.apply_clock_drift()
        if not self.is_active:
            self._delay_done = False
            self._delay_mic = []
            self._delay_ref = []
            self._delay_lock = None  # a mode change is an explicit epoch
            self._reset_joint_frames()
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
        self._delay_lock = None  # a fresh capture or mode change is an explicit epoch
