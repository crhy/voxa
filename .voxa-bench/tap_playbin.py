"""v11b: the playbin path (voxa/speech.py _play_file) with the render tap on the REAL sink.

Writes a short wav, hands a playbin its uri, attaches the tap exactly as the app does, and
counts buffers at every hop of the tap plus what the timeline ring ends up holding.
"""
import os
import time
import wave

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np

Gst.init(None)
from voxa.timeline import RenderTimeline, GstSinkClock, attach_render_tap

WAV = "/tmp/voxa_probe_tone.wav"
if not os.path.exists(WAV):
    rate = 16000
    t = np.arange(rate, dtype=np.float64)
    data = (np.sin(2 * 3.14159 * 440 * t / rate) * 12000).astype("<i2")
    with wave.open(WAV, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())

COUNT = {}


def probe(tag):
    def cb(pad, info):
        COUNT[tag] = COUNT.get(tag, 0) + 1
        return Gst.PadProbeReturn.OK
    return cb


def main():
    p = Gst.ElementFactory.make("playbin")
    p.set_property("uri", "file://" + WAV)
    print("get_by_name sink:", p.get_by_name("sink"), "audio-sink prop:",
          p.get_property("audio-sink"))
    tl = RenderTimeline(sample_rate=16000, clock=time.monotonic)
    tl.open("g1", "tts-file")
    if hasattr(tl, "open_turn"):
        tl.open_turn()
    made = attach_render_tap(p, tl, "g1", "tts-file", Gst, sink_clock=GstSinkClock(p, 16000))
    print("attach:", made.get("attached"), "error:", made.get("error"))
    els = made.get("elements") or {}
    print("elements:", sorted(els))
    for name in ("tee", "queue", "audioconvert", "audioresample", "capsfilter", "appsink",
                 "autoaudiosink", "bin"):
        el = els.get(name)
        if el is None:
            continue
        pad = el.get_static_pad("sink")
        if pad is not None:
            pad.add_probe(Gst.PadProbeType.BUFFER, probe(name))
    tap_sink = els.get("appsink")
    if tap_sink is not None:
        tap_sink.set_property("emit-signals", True)
        samples = []

        def cb(sink):
            s = sink.emit("pull-sample")
            if s is not None and s.get_buffer() is not None:
                samples.append(s.get_buffer().get_size())
            return Gst.FlowReturn.OK
        tap_sink.connect("new-sample", cb)

    bus = p.get_bus()
    bus.add_signal_watch()

    def on_msg(_b, m):
        if m.type in (Gst.MessageType.ERROR, Gst.MessageType.WARNING):
            err = m.parse_error() if m.type == Gst.MessageType.ERROR else m.parse_warning()
            print("BUS", m.type, err)
    bus.connect("message", on_msg)
    print("state ->", p.set_state(Gst.State.PLAYING))
    loop = GLib.MainLoop()
    GLib.timeout_add_seconds(3, loop.quit)
    loop.run()
    res, state, pending = p.get_state(1 * Gst.SECOND)
    print("state:", res, state, pending)
    print("counts:", COUNT, "tap new-sample:", len(samples), sum(samples))
    print("ring stats:", tl.turn_stats() if hasattr(tl, "turn_stats") else "n/a")
    try:
        peak = 0.0
        window = tl.ring.window(0, 16000)
        print("ring max abs:", float(np.max(np.abs(window))) if window is not None else None,
              "sum abs:", float(np.sum(np.abs(window))) if window is not None else None)
    except Exception as exc:
        print("ring read failed:", exc)
    p.set_state(Gst.State.NULL)
    print("done", flush=True)


main()
