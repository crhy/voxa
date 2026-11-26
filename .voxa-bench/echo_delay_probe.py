#!/usr/bin/env python3
"""Task v11c step 3: measure the render-map vs microphone clock error honestly.

v11b's blocker (issue #72): the render timeline's estimated audible time and the microphone
disagree by +-480 ms, straddling the 400 ms GCC-PHAT window, so the echo never fits. This probe
reproduces the app's TWO live pipelines with the app's own code:

* render:  appsrc -> ... -> autoaudiosink name=sink with ``attach_render_tap`` +
  ``GstSinkClock`` attached, exactly like ``voxa/speech.py`` (whole-reply/stream path);
* capture: pulsesrc -> audioconvert -> audioresample -> 16k S16LE appsink, exactly like
  ``voxa/audio.py start()``, stamped with the app's ``CaptureClock``.

It plays N chirps through the tapped sink and records the microphone through the capture
pipeline. For every chirp the offset is
``mic_onset(monotonic) - timeline.audible_time(chirp's first sample)``.
v11b measured -488/+480 (a straddle, sign flips); the v11c step-3 gate is ONE consistent
value: spread (max-min) <= 20 ms over >= 5 chirps.

Read-only on the user's audio: opens ONE stream, leaves the volume alone, closes everything.
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

from voxa.timeline import (CaptureClock, GstSinkClock, RenderTimeline,  # noqa: E402
                           attach_render_tap)

RATE = 16000
CHIRPS = 5
PRE_SILENCE = 0.6
CHIRP_SECONDS = 0.03
POST_SILENCE = 1.2
MIC_DEVICE = sys.argv[1] if len(sys.argv) > 1 else None


def chirp_stream() -> np.ndarray:
    parts = []
    for _ in range(CHIRPS):
        n_pre = int(PRE_SILENCE * RATE)
        n_c = int(CHIRP_SECONDS * RATE)
        n_post = int(POST_SILENCE * RATE)
        t = np.arange(n_c) / RATE
        f0, f1 = 500.0, 4000.0
        k = (f1 - f0) / CHIRP_SECONDS
        chunk = 0.5 * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
        parts.append(np.zeros(n_pre, dtype=np.float32))
        parts.append(chunk.astype(np.float32))
        parts.append(np.zeros(n_post, dtype=np.float32))
    return np.concatenate(parts)


def main() -> int:
    Gst.init(None)
    timeline = RenderTimeline(sample_rate=RATE)
    stream = chirp_stream()

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

    # Mic: the same capture pipeline and clock the app uses.
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

    # A couple of seconds of mic only, so the capture pipeline is warm and its counters settle.
    time.sleep(1.0)
    t0 = timeline.total
    base_mono = time.monotonic()
    print("render state ->", pipeline.set_state(Gst.State.PLAYING))
    bus = pipeline.get_bus()

    # Push the whole chirp stream in one buffer with PTS 0 (the app pushes as it streams; the
    # probe pushes its own clocked PCM so the map has the same timeline the app computes).
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
    with lock:
        nbuf = len(ring)
        size = sum(s.size for _t, s in ring)
    print("debug: timeline.total=%d (was %d) mic buffers=%d samples=%d (%.1f s)"
          % (timeline.total, t0, nbuf, size, size / RATE))
    print("debug: audible(0)=%s now=%.3f"
          % (timeline.audible_time(t0, now=base_mono), time.monotonic()))
    while True:
        msg = bus.poll(Gst.MessageType.EOS | Gst.MessageType.ERROR, 0)
        if msg is None:
            break
        if msg.type == Gst.MessageType.ERROR:
            print("render bus error:", msg.parse_error()[0].message)
        else:
            print("render bus:", msg.type)

    # Cross-correlate the microphone ring against the known chirp template once playback ends.
    offsets = []
    n_pre = int(PRE_SILENCE * RATE)
    n_c = int(CHIRP_SECONDS * RATE)
    n_post = int(POST_SILENCE * RATE)
    period = n_pre + n_c + n_post
    last_audible = timeline.audible_time(t0 + stream.size - 1, now=base_mono)
    deadline = (last_audible if last_audible is not None else time.monotonic()) + 2.0
    while time.monotonic() < deadline:
        time.sleep(0.2)

    pipeline.set_state(Gst.State.NULL)
    mic_pipeline.set_state(Gst.State.NULL)
    GLib.usleep(200_000)

    with lock:
        samples_all = np.concatenate([s for _t, s in ring]) if ring else np.zeros(0, np.float32)
        first_start = ring[0][0] if ring else None
    if first_start is None:
        print("measured only 0 of %d chirps" % CHIRPS)
        return 2
    template = stream[n_pre:n_pre + n_c]
    for k in range(CHIRPS):
        first_index = t0 + k * period + n_pre
        audible = timeline.audible_time(first_index, now=base_mono)
        if audible is None:
            continue
        lo = int((audible - 0.6 - first_start) * RATE)
        hi = int((audible + 1.2 - first_start) * RATE)
        if lo < 0 or hi > samples_all.size:
            continue
        window = samples_all[lo:hi]
        corr = np.correlate(window, template, mode="valid")
        peak_i = int(np.argmax(corr))
        snr = float(corr[peak_i]) / max(1e-9, float(np.median(np.abs(corr))))
        onset = first_start + (lo + peak_i) / RATE
        if snr < 5.0:
            print("chirp %d: xcorr snr %.1f (rejected)" % (k, snr))
            continue
        offsets.append((k, (onset - audible) * 1000.0))
        print("chirp %d: map audible=%.3f onset=%.3f offset=%.1f ms (snr %.1f)"
              % (k, audible, onset, (onset - audible) * 1000.0, snr))

    if len(offsets) < 5:
        print("measured only %d of %d chirps" % (len(offsets), CHIRPS))
        return 2
    values = [v for _k, v in offsets]
    spread = max(values) - min(values)
    print("offsets: " + ", ".join("%.1f" % v for v in values))
    print("spread: %.1f ms over %d chirps" % (spread, len(values)))
    return 0 if spread <= 20.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
