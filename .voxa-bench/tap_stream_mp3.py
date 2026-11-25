"""v11b: the exact whole-reply streaming shape (appsrc -> decodebin -> ... -> autoaudiosink)
with real MP3 bytes, the render tap attached, and the ATTRIBUTION read (moment window).

This is voxa/speech.py _start_natural_stream's pipeline: appsrc pushes encoded bytes and
decodebin picks the decoder, so the tap sees decoded PCM after the decoder chain.
"""
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np

MP3 = sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe_tone.mp3"
Gst.init(None)
from voxa.timeline import RenderTimeline, GstSinkClock, attach_render_tap

RATE = 16000
p = Gst.parse_launch(
    "appsrc name=source format=bytes block=true max-bytes=1048576 ! "
    "queue max-size-bytes=2097152 ! decodebin ! audioconvert ! audioresample ! "
    "autoaudiosink name=sink")
tl = RenderTimeline(sample_rate=RATE, clock=time.monotonic)
tl.open("g1", "tts-stream")
tl.open_turn()
clock = GstSinkClock(p, RATE)
made = attach_render_tap(p, tl, "g1", "tts-stream", Gst, sink_clock=clock)
print("attach:", made.get("attached"), made.get("error"))
els = made.get("elements") or {}
sink = els.get("sink")
print("sink element:", sink.get_factory().get_name() if sink else None)

bus = p.get_bus()
bus.add_signal_watch()


def on_msg(_b, m):
    if m.type in (Gst.MessageType.ERROR, Gst.MessageType.WARNING, Gst.MessageType.TAGS):
        print("BUS", m.type,
              m.parse_error() if m.type == Gst.MessageType.ERROR else None)
bus.connect("message", on_msg)

appsrc = p.get_by_name("source")
data = open(MP3, "rb").read()
# push the encoded bytes in chunks like the natural worker does
for i in range(0, len(data), 4096):
    buf = Gst.Buffer.new_allocate(None, len(data[i:i + 4096]), None)
    buf.fill(0, data[i:i + 4096])
    appsrc.emit("push-buffer", buf)

print("state ->", p.set_state(Gst.State.PLAYING))
start = time.monotonic()
for i in range(14):
    time.sleep(0.5)
    now = time.monotonic()
    w = tl.reference_window(mic_start=now, length=800)
    a = np.asarray(w, dtype=np.float64)
    pw = float(np.mean(a ** 2)) if a.size else 0.0
    g = tl.generation("g1")
    print(f"t={now - start:.1f} total={tl.total} window_power={pw:.6f} "
          f"submitted={getattr(g, 'submitted', None)} state={getattr(g, 'state', None)}")
    if now - start > 6.5:
        break
print("done", flush=True)
