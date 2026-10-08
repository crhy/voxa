from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .anc import EchoCanceller, estimate_delay

Gst: Any = None  # Initialized lazily so importing this module needs no GStreamer or PyGObject.

log = logging.getLogger("voxa.audio")


def _ensure_gstreamer() -> Any:
    global Gst
    if Gst is None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst as _Gst

        _Gst.init(None)
        Gst = _Gst
    return Gst

# Reference samples we keep: the last 3 seconds at 16 kHz.
REFERENCE_SECONDS = 3
REFERENCE_RATE = 16000


def webrtc_available() -> bool:
    """True when both WebRTC echo cancellation elements exist in this GStreamer runtime."""
    try:
        gst = _ensure_gstreamer()
        factory = gst.ElementFactory
        return bool(factory.is_available("webrtcdsp") and factory.is_available("webrtcechoprobe"))
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


def mix_chunk(mic_bytes: bytes, ref_ring: ReferenceRing, counter: int, canceller,
              ref_offset: int = 0) -> bytes:
    """Clean one microphone chunk against the reference samples covering the same span.

    ``counter`` is the global sample position where this microphone chunk begins; ``ref_offset``
    is the microphone position that corresponds to the reference stream's position 0, so the
    reference samples for the same wall-clock span start at ``counter - ref_offset`` (zero-padded
    when the reference has not produced enough yet). An idle monitor delivers nothing, so a chunk
    whose reference span is silence is passed through untouched (bypass). Returns the cleaned
    chunk as int16 bytes. Raises if the canceller raises.
    """
    mic = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float32) / 32768.0
    ref = ref_ring.window(counter - ref_offset, mic.size)
    if float(np.mean(ref * ref)) < 1e-9:
        return mic_bytes
    cleaned = canceller.process(mic, ref)
    cleaned = np.clip(np.rint(np.asarray(cleaned, dtype=np.float32) * 32767.0), -32768, 32767)
    return cleaned.astype("<i2").tobytes()


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
        self._start_args: tuple | None = None

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
        self._ref_ring = ReferenceRing(REFERENCE_SECONDS * REFERENCE_RATE) if self.reference_source else None
        if self.reference_source:
            self._start_reference(gst)

    def _start_reference(self, gst) -> None:
        """Open the second pipeline that records the computer's own playback into the ring buffer."""
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

        Returns True when the pipeline is playing; False means the caller must fall back to the
        plain capture pipeline (never leaving the user without a microphone).
        """
        monitor_src = gst.ElementFactory.make("pulsesrc")
        mic_src = gst.ElementFactory.make("pulsesrc")
        convert_m = gst.ElementFactory.make("audioconvert")
        resample_m = gst.ElementFactory.make("audioresample")
        probe = gst.ElementFactory.make("webrtcechoprobe")
        dsp = gst.ElementFactory.make("webrtcdsp")
        convert = gst.ElementFactory.make("audioconvert")
        resample = gst.ElementFactory.make("audioresample")
        capsfilter = gst.ElementFactory.make("capsfilter")
        sink = gst.ElementFactory.make("appsink")
        if not all((monitor_src, mic_src, convert_m, resample_m, probe, dsp, convert, resample, capsfilter, sink)):
            log.info("webrtc echo cancellation unavailable: GStreamer elements missing")
            return False
        monitor_src.set_property("device", self.reference_source)
        monitor_src.set_property("latency-time", 20000)
        monitor_src.set_property("buffer-time", 40000)
        mic_src.set_property("device", self._mic_pulse_source(gst, device_id))
        mic_src.set_property("latency-time", 20000)
        mic_src.set_property("buffer-time", 40000)
        probe.set_property("name", "probe")
        dsp.set_property("probe", "probe")
        dsp.set_property("echo-cancel", True)
        dsp.set_property("echo-suppression-level", 1)
        dsp.set_property("noise-suppression", False)
        dsp.set_property("gain-control", False)
        dsp.set_property("high-pass-filter", False)
        dsp.set_property("voice-detection", False)
        dsp.set_property("limiter", False)
        capsfilter.set_property(
            "caps", gst.Caps.from_string("audio/x-raw,format=S16LE,layout=interleaved,channels=1,rate=16000")
        )
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 12)
        if sink.find_property("drop") is not None:
            sink.set_property("drop", True)
        sink.connect("new-sample", self._on_sample, on_level)

        pipeline = gst.Pipeline.new("voxa-webrtc")
        for element in (monitor_src, convert_m, resample_m, probe, mic_src, convert, resample, dsp, capsfilter, sink):
            pipeline.add(element)
        links = (
            monitor_src.link_full(convert_m, 0, 0, gst.Pad.NEGOTIATE, False)
            and convert_m.link_full(resample_m, 0, 0, gst.Pad.NEGOTIATE, False)
            and resample_m.link_full(probe, 0, 0, gst.Pad.NEGOTIATE, False)
            and mic_src.link_full(convert, 0, 0, gst.Pad.NEGOTIATE, False)
            and convert.link_full(resample, 0, 0, gst.Pad.NEGOTIATE, False)
            and resample.link_full(dsp, 0, 0, gst.Pad.NEGOTIATE, False)
            and dsp.link_full(capsfilter, 0, 0, gst.Pad.NEGOTIATE, False)
            and capsfilter.link_full(sink, 0, 0, gst.Pad.NEGOTIATE, False)
        )
        if not links:
            pipeline.set_state(gst.State.NULL)
            log.info("webrtc echo cancellation unavailable: could not connect pipeline")
            return False
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_bus_error)
        if pipeline.set_state(gst.State.PLAYING) == gst.StateChangeReturn.LATER:
            deadline = time.monotonic() + 3.0
            playing = False
            while time.monotonic() < deadline:
                if pipeline.get_state() == gst.State.PLAYING:
                    playing = True
                    break
                time.sleep(0.05)
            if not playing:
                pipeline.set_state(gst.State.NULL)
                log.info("webrtc echo cancellation failed: pipeline never reached PLAYING")
                return False
        elif pipeline.get_state() != gst.State.PLAYING:
            pipeline.set_state(gst.State.NULL)
            log.info("webrtc echo cancellation failed: monitor source could not be opened")
            return False
        self._mic_counter = 0
        self._ref_anchor = None
        self._canceller_enabled = False
        self._canceller_logged = False
        self._delay_done = True
        self._delay_mic = []
        self._delay_ref = []
        self._ref_ring = None
        self.pipeline = pipeline
        return True

    def _mic_pulse_source(self, gst, device_id: str) -> str:
        """The PulseAudio device name for the microphone chosen in Voxa."""
        if self.pulse_source:
            return self.pulse_source
        selected = next((item for item in self._devices if item.identifier == device_id), None)
        if selected is not None:
            props = selected.properties
            for key in ("alsa.device", "device.profile", "object.path", "device.path"):
                value = _stable_field(props, key) if props is not None else ""
                if value:
                    return value
        return ""

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

    def _mix_or_passthrough(self, pcm: bytes) -> bytes:
        """Clean ``pcm`` against the reference when a canceller is active; never break recording."""
        if self.canceller is None or not self._canceller_enabled or self._ref_ring is None:
            self._mic_counter += len(pcm) // 2
            return pcm
        try:
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
        """During the first loud-enough 3 seconds, measure the echo delay once and rebuild the canceller."""
        if self._delay_done or self._ref_ring is None:
            return
        mic = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        ref = self._ref_ring.window(self._mic_counter - (self._ref_anchor or 0), mic.size)
        if float(np.mean(ref * ref)) < 1e-6:
            return  # the reference is not loud enough yet
        self._delay_mic.append(mic)
        self._delay_ref.append(ref)
        collected = sum(chunk.size for chunk in self._delay_mic)
        if collected < REFERENCE_SECONDS * REFERENCE_RATE:
            return
        self._delay_done = True
        delay = estimate_delay(np.concatenate(self._delay_mic), np.concatenate(self._delay_ref))
        if delay > 0:
            self.canceller = EchoCanceller(delay=delay)
            log.info("echo delay estimated at %d samples (%.3f s)", delay, delay / REFERENCE_RATE)

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

    def stop(self) -> None:
        gst = _ensure_gstreamer() if (self.pipeline is not None or self._ref_pipeline is not None) else None
        pipeline, self.pipeline = self.pipeline, None
        ref_pipeline, self._ref_pipeline = self._ref_pipeline, None
        if pipeline is not None:
            pipeline.set_state(gst.State.NULL)
        if ref_pipeline is not None:
            ref_pipeline.set_state(gst.State.NULL)

    def set_echo_mode(self, mode: str | bool = "voxa", reference_source: str | None = None) -> None:
        """Switch echo cancellation mode ("voxa", "webrtc" or "off"), applying immediately if capture is running.

        A boolean is accepted for the old call style: ``True`` maps to "voxa", ``False`` to "off".
        """
        if isinstance(mode, bool):
            mode = "voxa" if mode else "off"
        if mode == "webrtc" and not webrtc_available():
            log.info("webrtc echo cancellation unavailable: running with echo off")
            mode = "off"
        if mode == "webrtc":
            self.pulse_source = None
            self.reference_source = reference_source
            self.canceller = None
            self._canceller_enabled = False
            self._canceller_logged = False
            self._delay_done = True
            self._delay_mic = []
            self._delay_ref = []
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
        if mode != "voxa":
            self.pulse_source = None
            self.reference_source = None
            self.canceller = None
            self._canceller_enabled = False
            self._delay_done = True
            self._delay_mic = []
            self._delay_ref = []
            self._ref_ring = None
            self._ref_anchor = None
            if self._ref_pipeline is not None:
                gst = _ensure_gstreamer()
                self._ref_pipeline.set_state(gst.State.NULL)
                self._ref_pipeline = None
            return
        self.pulse_source = None
        self.reference_source = reference_source
        if self.canceller is None:
            self.canceller = EchoCanceller()
        self._canceller_enabled = True
        self._canceller_logged = False
        if not self.is_active:
            self._delay_done = False
            self._delay_mic = []
            self._delay_ref = []
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
