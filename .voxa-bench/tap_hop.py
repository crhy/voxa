"""v11b: where do the buffers stop when the tap sits in front of the REAL autoaudiosink?

Same pipeline shape as voxa/speech.py's streaming reply, tap attached the same way, but with
pad probes on every hop: upstream->tee, tee->sink, tee->tap branch, and the tap branch's own
elements. Prints a buffer count per hop plus the tap branch's pad state and caps.
"""
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)
from voxa.timeline import RenderTimeline, GstSinkClock, attach_render_tap

COUNT = {}


def probe(tag, pad, info):
    COUNT[tag] = COUNT.get(tag, 0) + 1
    return Gst.PadProbeReturn.OK


def pad_of(made, name):
    el = (made.get("elements") or {}).get(name)
    return el.get_static_pad("sink") if el is not None else None


def main():
    p = Gst.parse_launch("appsrc name=source format=bytes ! audioconvert ! "
                         "audioresample ! autoaudiosink name=sink")
    tl = RenderTimeline(sample_rate=16000, clock=time.monotonic)
    tl.open("g1", "tts-stream")
    if hasattr(tl, "open_turn"):
        tl.open_turn()
    made = attach_render_tap(p, tl, "g1", "tts-stream", Gst, sink_clock=GstSinkClock(p, 16000))
    print("attach:", made.get("attached"), made.get("error"))
    els = made.get("elements") or {}
    tee = els.get("tee")
    sink = els.get("sink")
    print("tee:", tee, "sink:", sink)

    # hop 1: the pad the tee feeds from the playback path
    for pad_name, pad in (("tee.sink", tee.get_static_pad("sink") if tee else None),):
        if pad is not None:
            pad.add_probe(Gst.PadProbeType.BUFFER, lambda _p, _i, n=pad_name: probe(n, _p, _i))
    # hop 2: the autoaudiosink's own sink pad (fed by the tee)
    sp = sink.get_static_pad("sink") if sink is not None else None
    if sp is not None:
        sp.add_probe(Gst.PadProbeType.BUFFER, lambda _p, _i: probe("autoaudiosink.sink", _p, _i))
    # hop 3..: the tap branch after the tee
    for name in ("queue", "audioconvert", "audioresample", "capsfilter", "appsink"):
        pad = pad_of(made, name)
        if pad is not None:
            pad.add_probe(Gst.PadProbeType.BUFFER,
                          lambda _p, _i, n=name: probe(n, _p, _i))

    import numpy as np
    print("state ->", p.set_state(Gst.State.PLAYING))
    appsrc = p.get_by_name("source")
    appsrc.set_property("caps", Gst.Caps.from_string(
        "audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved"))
    data = (np.sin(2 * 3.14159 * 440 * np.arange(16000) / 16000) * 12000).astype("<i2")
    chunk = data.tobytes()
    for i in range(20):
        buf = Gst.Buffer.new_allocate(None, len(chunk), None)
        buf.fill(0, chunk)
        appsrc.emit("push-buffer", buf)

    loop = GLib.MainLoop()
    GLib.timeout_add_seconds(3, loop.quit)
    loop.run()
    print("counts:", COUNT)
    res, state, pending = p.get_state(1 * Gst.SECOND)
    print("state:", res, state, pending)
    for name in ("queue", "audioconvert", "capsfilter", "appsink"):
        el = els.get(name)
        if el is None:
            continue
        pad = el.get_static_pad("sink")
        tmpl = pad.get_query_caps() if pad else None
        print(name, "pad tmpl caps:", tmpl)
    print("tee src pads:", [pp.get_name() for pp in tee.get_pads()] if tee else None)
    print("tap branch pad:", els.get("queue") and els["queue"].get_static_pad("sink").get_peer())


main()
