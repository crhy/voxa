"""v11b offline reproduction (host, GST_AUDIOSINK UNSET, real sink path).

Both shipped speech pipelines with a REAL autoaudiosink (the default sink is a null sink the
test module creates), tap attached exactly as voxa/speech.py does, then: does the render
timeline hold non-silent samples? Prints the numbers; the test version asserts them.
"""
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np

sys.path.insert(0, ".")
Gst.init(None)
from voxa.timeline import RenderTimeline, GstSinkClock, attach_render_tap

RATE = 16000
MODE = sys.argv[1] if len(sys.argv) > 1 else "both"


def run(mode):
    if mode == "piece":
        import os
        import wave
        wav = ".voxa-bench/real_sink_probe_tone.wav"
        if not os.path.exists(wav):
            t = np.arange(RATE, dtype=np.float64)
            data = (np.sin(2 * 3.14159 * 440 * t / RATE) * 12000).astype("<i2")
            with wave.open(wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(data.tobytes())
        pipeline = Gst.ElementFactory.make("playbin")
        pipeline.set_property("uri", "file://" + os.path.abspath(wav))
    else:
        pipeline = Gst.parse_launch(
            "appsrc name=source format=bytes ! audioconvert ! audioresample ! "
            "autoaudiosink name=sink")
    tl = RenderTimeline(sample_rate=RATE, clock=time.monotonic)
    tl.open("g1", mode)
    tl.open_turn()
    made = attach_render_tap(pipeline, tl, "g1", mode, Gst, sink_clock=GstSinkClock(pipeline, RATE))
    print(mode, "attach:", made.get("attached"), "error:", made.get("error"))

    samples = []

    def cb(sink):
        s = sink.emit("pull-sample")
        if s is not None and s.get_buffer() is not None:
            samples.append(s.get_buffer().get_size())
        return Gst.FlowReturn.OK

    tap_sink = (made.get("elements") or {}).get("appsink")
    if tap_sink is not None:
        tap_sink.connect("new-sample", cb)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_msg(_b, m):
        if m.type in (Gst.MessageType.ERROR, Gst.MessageType.WARNING):
            print(mode, "BUS", m.type,
                  m.parse_error() if m.type == Gst.MessageType.ERROR else m.parse_warning())
    bus.connect("message", on_msg)

    pipeline.set_state(Gst.State.PLAYING)
    if mode != "piece":
        appsrc = pipeline.get_by_name("source")
        appsrc.set_property("caps", Gst.Caps.from_string(
            "audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved"))
        t = np.arange(RATE, dtype=np.float64)
        data = (np.sin(2 * 3.14159 * 440 * t / RATE) * 12000).astype("<i2")
        for _ in range(3):
            buf = Gst.Buffer.new_allocate(None, data.nbytes, None)
            buf.fill(0, data.tobytes())
            appsrc.emit("push-buffer", buf)
    loop = GLib.MainLoop()
    GLib.timeout_add_seconds(4, loop.quit)
    loop.run()
    stats = tl.turn_stats()
    print(mode, "tap samples:", len(samples), "turn:", stats)


for mode in (["piece", "whole-reply"] if MODE == "both" else [MODE]):
    run(mode)
