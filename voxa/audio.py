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


def mix_chunk(mic_bytes: bytes, ref_ring: ReferenceRing, counter: int, canceller) -> bytes:
    """Clean one microphone chunk against the reference samples covering the same span.

    ``counter`` is the global sample position where this microphone chunk begins; the reference
    samples for that same span are taken from ``ref_ring`` (zero-padded when the reference has not
    produced enough yet). Returns the cleaned chunk as int16 bytes. Raises if the canceller raises.
    """
    mic = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float32) / 32768.0
    ref = ref_ring.window(counter, mic.size)
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
    return "monitor of" in (display_name or "").casefold()


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
        self._canceller_enabled = True
        self._canceller_logged = False
        self._delay_done = True
        self._delay_mic: list = []
        self._delay_ref: list = []

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
        self._on_audio = on_audio
        self._on_error = on_error

        selected = next((item for item in self._devices if item.identifier == device_id), None)
        # A profile saved by an older build could still point at a monitor
        # node; refuse to open it rather than capture the machine's own audio.
        if selected is not None and is_monitor_source(selected.properties, selected.name):
            raise RuntimeError("The selected source is a monitor stream and cannot capture your voice.")
        gst = _ensure_gstreamer()
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
            cleaned = mix_chunk(pcm, self._ref_ring, self._mic_counter, self.canceller)
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
        ref = self._ref_ring.window(self._mic_counter, mic.size)
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
