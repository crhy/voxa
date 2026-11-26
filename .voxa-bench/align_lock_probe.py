#!/usr/bin/env python3
"""Task v11d: run the SHIPPING exact lock on live (microphone, render-map) pairs.

The chirp probe proved the map-vs-microphone offset is one consistent NEGATIVE value
(-30.7..-31.0 ms), yet the live bench never fitted travel (echo-health travel_ms=None) and
still interrupted herself. The rolling buffer _measure sees is only _keep ~= 0.52 s, so this
probe monkey-patches ExactAligner._measure/_coherent and prints the EXACT gate chain
(prominence / second path / full-buffer coherence / recent-tail coherence) on the exact
arrays the shipping lock used, per feed that carries reference content.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from voxa.align import ExactAligner  # noqa: E402
from voxa.timeline import (CaptureClock, GstSinkClock, RenderTimeline,  # noqa: E402
                           attach_render_tap)

RATE = 16000
SECONDS = 2.2
REPEATS = 3
GAP = 0.15
MIC_DEVICE = sys.argv[1] if len(sys.argv) > 1 else None


def speech_like() -> np.ndarray:
    """Broadband, speech-like sweep, played REPEATS times back to back (one long reply)."""
    n = int(SECONDS * RATE)
    rng = np.random.default_rng(7)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    spec *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    sig = np.fft.irfft(spec, n)
    sig = (0.4 * sig / np.max(np.abs(sig))).astype(np.float32)
    pre = np.zeros(int(0.5 * RATE), dtype=np.float32)
    gap = np.zeros(int(GAP * RATE), dtype=np.float32)
    post = np.zeros(int(1.0 * RATE), dtype=np.float32)
    parts = [pre]
    for _ in range(REPEATS):
        parts.append(sig)
        parts.append(gap)
    parts.pop()  # no gap before the post-silence
    parts.append(post)
    return np.concatenate(parts)


class DiagnosticsAligner(ExactAligner):
    """The shipping lock, with every _measure refusal explained on the shipping arrays."""

    def _measure(self, mic, ref, recent=False):
        out = super()._measure(mic, ref, recent=recent)
        if out is not None or recent or self._locked is not None:
            return out
        if float(np.mean(ref * ref)) <= 1e-9:
            return out
        if mic.size < 1600:
            return out
        n = min(mic.size, ref.size)
        m, r = mic[:n], ref[:n]
        max_delay = int(self._search_ms / 1000.0 * self._sr)
        size = 1
        while size < n + 2 * max_delay:
            size <<= 1
        cross = np.fft.rfft(m, size) * np.conj(np.fft.rfft(r, size))
        cross /= np.abs(cross) + 1e-12
        corr = np.fft.irfft(cross, size)
        signed = np.concatenate((corr[size - max_delay:], corr[:max_delay + 1]))
        peak_i = int(np.argmax(signed))
        floor = float(np.median(signed))
        top = float(signed[peak_i])
        lag = peak_i - max_delay
        print(f"  REFUSE n={n} peak_lag={lag} ({lag / self._sr * 1000:.1f} ms) "
              f"top={top:.3e} floor={floor:.3e} prom_ok={top >= 8.0 * floor}")
        band_far = self._sr // 25
        far_lo = max(0, peak_i - band_far)
        far_hi = min(signed.size, peak_i + band_far)
        rest = np.concatenate((signed[:far_lo], signed[far_hi:]))
        refused2 = rest.size and float(rest.max()) >= 0.5 * top
        print(f"  second path refused={bool(refused2)} rest.max={float(rest.max()):.3e} "
              f"half_top={0.5 * top:.3e}")

        def _coh_val(lag_v: int, tail: int = 0) -> tuple:
            if lag_v == 0 or abs(lag_v) >= len(m) or abs(lag_v) >= len(r):
                return (-1.0, 0)
            if lag_v > 0:
                end = len(m)
                start = max(lag_v, len(m) - tail) if tail else lag_v
            else:
                end = min(len(m), len(r) + lag_v)
                start = max(0, end - tail) if tail else 0
            mm_win = end - start
            a = m[start:end]
            b = r[start - lag_v:end - lag_v]
            mm = float(np.dot(a, a))
            rr = float(np.dot(b, b))
            dot = float(np.dot(a, b))
            if mm <= 0.0 or rr <= 0.0:
                return (-1.0, mm_win)
            return (dot * dot / (mm * rr), mm_win)

        need = self._coherence_min ** 2
        for tail in (0, self._window + self._sr // 4):
            coh, w = _coh_val(lag, tail)
            print(f"  coh(lag={lag}, tail={tail}, win={w}): {coh:.4f} need>={need:.4f} "
                  f"pass={coh >= need}")
        return out

    def observe(self, t_s, mic, ref):
        out = super().observe(t_s, mic, ref)
        if self._locked is not None:
            print(f"  LOCK t={t_s:.3f} lag={self._locked:.1f} "
                  f"({self._locked / self._sr * 1000:.1f} ms) refined={self.refined} "
                  f"conf={self.confidence:.2f}")
        return out


def main() -> int:
    Gst.init(None)
    timeline = RenderTimeline(sample_rate=RATE)
    stream = speech_like()

    pipeline = Gst.parse_launch(
        "appsrc name=source format=time block=true "
        r"caps=audio/x-raw\,format=F32LE\,channels=1\,rate=%d\,layout=interleaved ! " % RATE +
        "audioconvert ! audioresample ! audio/x-raw,format=F32LE,channels=1,rate=%d,layout=interleaved ! "
        "autoaudiosink name=sink sync=true" % RATE)
    appsrc = pipeline.get_by_name("source")
    sink = pipeline.get_by_name("sink")
    made = attach_render_tap(pipeline, timeline, "probe-gen", "tts-stream", gst=Gst,
                             sink_element=sink, sink_clock=GstSinkClock(pipeline, RATE))
    print("render-tap attached=%s error=%s" % (made.get("attached"), made.get("error")))
    if not made.get("attached"):
        return 2

    clock = CaptureClock(rate=RATE)
    ring = []           # (mono_time_start, samples float32)
    lock = threading.Lock()

    mic_pipeline = Gst.parse_launch(
        "pulsesrc name=src do-timestamp=true ! audioconvert ! audioresample ! "
        "audio/x-raw,format=S16LE,channels=1,rate=%d ! appsink name=sink sync=false" % RATE)
    if MIC_DEVICE:
        mic_pipeline.get_by_name("src").set_property("device", MIC_DEVICE)
    mic_sink = mic_pipeline.get_by_name("sink")
    mic_sink.set_property("emit-signals", True)
    mic_sink.set_property("max-buffers", 12)

    def on_sample(sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, mapped = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            data = np.frombuffer(bytes(mapped.data), dtype="<i2").astype(np.float32) / 32768.0
        finally:
            buf.unmap(mapped)
        arrival = time.monotonic()
        start, _end = clock.stamp(arrival, data.size)
        with lock:
            ring.append((start, data))
        return Gst.FlowReturn.OK

    mic_sink.connect("new-sample", on_sample)

    if mic_pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("mic pipeline failed")
        return 2
    time.sleep(1.0)
    t0 = timeline.total
    base_mono = time.monotonic()
    print("render state ->", pipeline.set_state(Gst.State.PLAYING))

    def push():
        buf = Gst.Buffer.new_wrapped(stream.astype(np.float32).tobytes())
        buf.pts = 0
        buf.duration = int(len(stream) / RATE * Gst.SECOND)
        appsrc.emit("push-buffer", buf)
        appsrc.emit("end-of-stream")

    threading.Thread(target=push, daemon=True).start()
    deadline = time.monotonic() + 30.0
    while timeline.total == t0 and time.monotonic() < deadline:
        time.sleep(0.05)
    last_audible = timeline.audible_time(t0 + stream.size - 1, now=base_mono)
    deadline = (last_audible if last_audible is not None else time.monotonic()) + 1.2
    while time.monotonic() < deadline:
        time.sleep(0.2)
    pipeline.set_state(Gst.State.NULL)
    mic_pipeline.set_state(Gst.State.NULL)
    GLib.usleep(200_000)

    with lock:
        arrivals = sorted(ring)
    print(f"mic buffers={len(arrivals)} samples={sum(a.size for _, a in arrivals)}")
    if not arrivals:
        print("NO MIC")
        return 1

    aligner = DiagnosticsAligner(sample_rate=RATE)
    feeds = locked_feeds = 0
    for mic_start, mic in arrivals:
        ref = timeline.reference_window(mic_start=mic_start, length=mic.size)
        if ref is None:
            continue
        ref = np.asarray(ref, dtype=np.float32)
        if float(np.mean(ref * ref)) <= 1e-9:
            continue
        feeds += 1
        aligner.observe(mic_start, mic[: ref.size], ref[: mic.size])
        if aligner.locked_delay is not None:
            locked_feeds += 1
    print(f"feeds={feeds} locked_feeds={locked_feeds} final_lock={aligner.locked_delay} "
          f"refined={aligner.refined}")
    out = Path.home() / ".var/app/io.github.crhy.voxa/data/align_capture.npz"
    starts = np.array([a[0] for a in arrivals])
    lens = np.array([a[1].size for a in arrivals])
    refs = []
    for st, a in arrivals:
        r = timeline.reference_window(mic_start=st, length=a.size)
        refs.append(np.zeros(a.size, np.float32) if r is None else np.asarray(r, np.float32))
    np.savez(out, starts=starts, lens=lens,
             mic=np.concatenate([a[1] for a in arrivals]),
             ref=np.concatenate(refs))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
