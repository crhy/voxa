"""v11b: the playbin path (piece / live whole-reply file) with the tap on the REAL sink,
read the way ATTRIBUTION reads it (moment window). Installed app code, sandbox."""
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np

Gst.init(None)
from voxa.timeline import RenderTimeline, GstSinkClock, attach_render_tap

RATE = 16000
p = Gst.ElementFactory.make("playbin")
p.set_property("uri", "file:///tmp/probe_tone.mp3")
tl = RenderTimeline(sample_rate=RATE, clock=time.monotonic)
tl.open("g1", "tts-file")
tl.open_turn()
clock = GstSinkClock(p, RATE)
made = attach_render_tap(p, tl, "g1", "tts-file", Gst, sink_clock=clock)
print("attach:", made.get("attached"), made.get("error"), "elements:", sorted(made.get("elements") or {}))

bus = p.get_bus()
bus.add_signal_watch()


def on_msg(_b, m):
    if m.type in (Gst.MessageType.ERROR, Gst.MessageType.WARNING):
        print("BUS", m.type,
              m.parse_error() if m.type == Gst.MessageType.ERROR else m.parse_warning())
bus.connect("message", on_msg)

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
