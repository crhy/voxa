"""v11b step 8 measurement: what is the real echo delay on THIS machine?

Play a chirp through the same pipeline shape the live whole-reply path uses (appsrc ->
autoaudiosink, tap attached, render timeline stamps audible_at through GstSinkClock), record the
microphone at the same time, and cross-correlate the mic against the render over a long lag
range. Prints the lag(s) in ms: if the true acoustic+device delay is outside the canceller's
0-400 ms search (voxa/anc.py GCCPHAT_SEARCH_MS), the delay lock can never lock and every
decision is 'alignment unhealthy: audio unqualified'.
"""
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np

sys.path.insert(0, ".")
Gst.init(None)
from voxa.timeline import GstSinkClock, RenderTimeline, attach_render_tap

RATE = 16000
CHIRPS = [(0.5, 1.0), (2.5, 1.0)]  # (start offset s, duration s)


def chirp(seconds, f0=400.0, f1=4000.0):
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    k = (f1 - f0) / seconds
    return (0.6 * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))).astype(np.float32)


timeline = RenderTimeline(sample_rate=RATE, clock=time.monotonic)
timeline.open("probe", "whole-reply")
timeline.open_turn()

play = Gst.parse_launch(
    "appsrc name=source format=bytes ! audioconvert ! audioresample ! autoaudiosink name=sink")
made = attach_render_tap(play, timeline, "probe", "whole-reply", Gst,
                         sink_clock=GstSinkClock(play, RATE))
print("tap attached:", made.get("attached"), made.get("error"))

mic_chunks = []
mic_t0 = [None]
rec = Gst.parse_launch("pulsesrc name=mic ! audioconvert ! audioresample ! "
                       "audio/x-raw,rate=16000,channels=1,format=F32LE ! appsink name=rec")
rec_sink = rec.get_by_name("rec")
rec_sink.set_property("emit-signals", True)
rec_sink.set_property("sync", False)


def on_mic(sink):
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK
    buf = sample.get_buffer()
    ok, mapped = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK
    data = np.frombuffer(bytes(mapped.data), dtype=np.float32)
    if mic_t0[0] is None:
        mic_t0[0] = time.monotonic()
    mic_chunks.append((mic_t0[0] if mic_t0[0] else time.monotonic(), data))
    buf.unmap(mapped)
    return Gst.FlowReturn.OK


rec_sink.connect("new-sample", on_mic)

appsrc = play.get_by_name("source")
appsrc.set_property("caps", Gst.Caps.from_string(
    "audio/x-raw,format=F32LE,channels=1,rate=16000,layout=interleaved"))

play.set_state(Gst.State.PLAYING)
rec.set_state(Gst.State.PLAYING)
time.sleep(0.5)

for start, dur in CHIRPS:
    buf = Gst.Buffer.new_allocate(None, (int(dur * RATE) * 4), None)
    buf.fill(0, chirp(dur).tobytes())
    appsrc.emit("push-buffer", buf)
    silence = np.zeros(int(2.0 * RATE), dtype=np.float32)
    sb = Gst.Buffer.new_allocate(None, silence.nbytes, None)
    sb.fill(0, silence.tobytes())
    appsrc.emit("push-buffer", sb)

loop = GLib.MainLoop()
GLib.timeout_add_seconds(7, loop.quit)
loop.run()
appsrc.emit("end-of-stream")
play.set_state(Gst.State.NULL)
rec.set_state(Gst.State.NULL)

mic = np.concatenate([d for _t, d in mic_chunks]) if mic_chunks else np.zeros(RATE)
ref = timeline.ring.window(timeline.total - mic.size if mic.size < timeline.total else 0,
                           min(mic.size, timeline.total))
ref = np.asarray(ref).reshape(-1)
if ref.size < mic.size:  # pad the front of the reference so both cover the same wall window
    ref = np.concatenate([np.zeros(mic.size - ref.size, dtype=np.float32), ref])

def expected_onset(offset_s: float) -> float | None:
    """Wall time the render map says the sample at render position offset_s*RATE leaves the speakers."""
    idx = int(offset_s * RATE)
    for chunk in list(timeline._anchors):
        if chunk.first_index <= idx < chunk.first_index + chunk.samples:
            return chunk.audible_at + (idx - chunk.first_index) / RATE
    return None


def mic_onset(offset_s: float, width_s: float = 1.2) -> float | None:
    """When the mic first heard the chirp inside [offset_s, offset_s+width_s]."""
    lo = int(offset_s * RATE)
    hi = min(mic.size, int((offset_s + width_s) * RATE))
    seg = mic[lo:hi]
    if seg.size == 0:
        return None
    env = np.convolve(np.abs(seg), np.ones(RATE // 50) / 20.0, mode="same")
    thresh = max(0.05 * float(np.max(env)), float(np.percentile(env, 20)) + 0.02)
    hits = np.nonzero(env > thresh)[0]
    if hits.size == 0:
        return None
    return mic_t0[0] + (lo + int(hits[0])) / RATE


print(f"mic={mic.size} rms_mic={np.sqrt(np.mean(mic**2)):.4f} mic_t0 offset anchors={len(timeline._anchors)}")
for start, dur in CHIRPS:
    exp = expected_onset(start)
    got = mic_onset(start)
    if exp is None or got is None or mic_t0[0] is None:
        print(f"chirp@{start}s: expected={exp} mic={got}")
        continue
    print(f"chirp@{start}s: mic arrived {1000.0 * (got - exp):.1f} ms after the render map's audible time")
