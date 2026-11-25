"""v11b: does the render timeline answer the ATTRIBUTION read (moment-based window) while the
real sink plays? Filing at the wrong time reads silent even when the tap attached."""
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np

Gst.init(None)
from voxa.timeline import RenderTimeline, GstSinkClock, attach_render_tap

RATE = 16000
p = Gst.parse_launch("appsrc name=source format=bytes ! audioconvert ! "
                     "audioresample ! autoaudiosink name=sink")
tl = RenderTimeline(sample_rate=RATE, clock=time.monotonic)
tl.open("g1", "tts-stream")
tl.open_turn()
clock = GstSinkClock(p, RATE)
made = attach_render_tap(p, tl, "g1", "tts-stream", Gst, sink_clock=clock)
print("attach:", made.get("attached"), made.get("error"))

import threading

stop_push = threading.Event()


def pusher():
    appsrc = p.get_by_name("source")
    appsrc.set_property("caps", Gst.Caps.from_string(
        "audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved"))
    t = np.arange(RATE, dtype=np.float64)
    data = (np.sin(2 * 3.14159 * 440 * t / RATE) * 12000).astype("<i2")
    while not stop_push.is_set():
        for _ in range(10):
            buf = Gst.Buffer.new_allocate(None, data.nbytes, None)
            buf.fill(0, data.tobytes())
            appsrc.emit("push-buffer", buf)
        time.sleep(0.5)


threading.Thread(target=pusher, daemon=True).start()
print("state ->", p.set_state(Gst.State.PLAYING))
start = time.monotonic()
for i in range(12):
    time.sleep(0.5)
    now = time.monotonic()
    try:
        lat = clock.latency()
    except Exception as exc:
        lat = f"latency failed: {exc}"
    try:
        ql = p.query_latency()
    except Exception as exc:
        ql = f"query failed {exc}"
    w_moment = tl.reference_window(mic_start=now - 0.1, length=800) if hasattr(tl, "reference_window") else None
    try:
        tw = tl.turn_reference_window(moment=now, length=800)
    except Exception as exc:
        tw = f"turn window failed: {exc}"
    def power(x):
        if x is None:
            return None
        a = np.asarray(x, dtype=np.float64)
        return float(np.mean(a ** 2)) if a.size else 0.0
    print(f"t={now - start:.1f} latency={lat if not isinstance(lat, float) else round(lat, 4)} "
          f"query={ql if isinstance(ql, str) else (ql[0], ql[1], ql[2])} "
          f"total={tl.total} window_now={power(tw):.6f} "
          f"turn_live={tl.turn_is_live(now) if hasattr(tl, 'turn_is_live') else '?'} "
          f"turn_stats_samples={tl.turn_stats()['samples'] if hasattr(tl, 'turn_stats') else '?'}")
stop_push.set()
loop = GLib.MainLoop()
GLib.timeout_add_seconds(1, loop.quit)
loop.run()
