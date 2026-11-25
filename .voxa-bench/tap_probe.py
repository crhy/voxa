"""Issue #72 task v11b: which sink element does Voxa's render tap find on the REAL sink?

Run inside the installed flatpak with no GST_AUDIOSINK, replicating the shipped speech
pipelines exactly as voxa/speech.py builds them:

  stream: appsrc -> decodebin -> ... -> autoaudiosink name=sink  (whole-reply streaming)
  file:   playbin (uri)  (prepared pieces and the whole-reply file via _play_file)

Prints what the tap finds, whether it attaches, what real child autoaudiosink made after
the state change, and whether the tap branch ever sees a single sample.
"""
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

Gst.init(None)
print("gst", Gst.version_string())

from voxa.timeline import RenderTimeline, GstSinkClock, attach_render_tap  # noqa: E402

REAL_STREAM = ("appsrc name=source format=bytes block=true max-bytes=1048576 ! "
               "queue max-size-bytes=2097152 ! decodebin ! audioconvert ! audioresample ! "
               "autoaudiosink name=sink")


def _kids(el):
    if hasattr(el, "iterate_all_elems"):
        try:
            return [c.get_factory().get_name() for c in el.iterate_all_elems()]
        except Exception as exc:  # noqa: BLE001
            return ["iterate failed: %s" % exc]
    if hasattr(el, "get_children"):
        try:
            return [c.get_factory().get_name() for c in el.get_children()]
        except Exception as exc:  # noqa: BLE001
            return ["get_children failed: %s" % exc]
    return ["not a bin: " + el.get_factory().get_name()]


def timeline_stats(tl):
    if hasattr(tl, "turn_stats"):
        try:
            return tl.turn_stats()
        except Exception:  # noqa: BLE001
            pass
    try:
        g = tl.generation("g1")
        return {"samples": getattr(g, "samples", None), "peak": getattr(g, "peak", None)}
    except Exception as exc:  # noqa: BLE001
        return "stats unavailable: %s" % exc


def describe(tag, pipeline):
    found = None
    for n in ("sink", "audio-sink"):
        try:
            found = pipeline.get_by_name(n)
        except Exception as exc:  # noqa: BLE001
            print(tag, "get_by_name", n, "raised", exc)
        if found is not None:
            print(tag, "found by name:", n, "->", found.get_factory().get_name())
            break
    if found is None:
        try:
            found = pipeline.get_property("audio-sink")
        except Exception:  # noqa: BLE001
            found = None
        print(tag, "audio-sink property:", found)
    if found is None:
        print(tag, "-> NO sink element to tap in front of (attach would stop here)")
    else:
        pad = found.get_static_pad("sink")
        print(tag, "sink pad:", pad is not None, "peer:", pad.get_peer() if pad else None)
        kids = _kids(found)
        print(tag, "children at build time:", kids)


def flow(tag, pipeline, timeline, made, seconds=3):
    """Push audio and count what the tap branch's appsink actually receives."""
    seen = {"n": 0, "bytes": 0}
    tap_sink = (made.get("elements") or {}).get("appsink")

    def cb(sink):
        sample = sink.emit("pull-sample")
        if sample is not None:
            seen["n"] += 1
            seen["bytes"] += sample.get_buffer().get_size()
        return Gst.FlowReturn.OK

    if tap_sink is not None:
        tap_sink.set_property("emit-signals", True)
        tap_sink.connect("new-sample", cb)

    print(tag, "state ->", pipeline.set_state(Gst.State.PLAYING), flush=True)
    s = pipeline.get_by_name("sink")
    if s is not None:
        print(tag, "children while PLAYING:", _kids(s))

    appsrc = pipeline.get_by_name("source")
    if appsrc is not None:
        import numpy as np
        appsrc.set_property("caps", Gst.Caps.from_string(
            "audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved"))
        appsrc.set_property("block", False)
        t = np.arange(16000, dtype=np.float64)
        data = (np.sin(2 * 3.14159 * 440 * t / 16000) * 12000).astype("<i2")
        chunk = data.tobytes()
        for _ in range(seconds):
            buf = Gst.Buffer.new_allocate(None, len(chunk), None)
            buf.fill(0, chunk)
            appsrc.emit("push-buffer", buf)
    print(tag, "pushed, running loop", flush=True)

    loop = GLib.MainLoop()
    GLib.timeout_add_seconds(seconds + 1, loop.quit)
    loop.run()
    print(tag, "loop done", flush=True)
    print(tag, "TAP branch samples:", seen["n"], "bytes:", seen["bytes"],
          "timeline:", timeline_stats(timeline), flush=True)
    import contextlib
    with contextlib.suppress(Exception):
        pipeline.set_state(Gst.State.NULL)
    print(tag, "torn down", flush=True)



def main():
    # 1) the exact streaming chain from _start_natural_stream
    p = Gst.parse_launch(REAL_STREAM)
    describe("stream", p)
    tl = RenderTimeline(sample_rate=16000, clock=time.monotonic)
    tl.open("g1", "tts-stream")
    tl.open_turn()
    made = attach_render_tap(p, tl, "g1", "tts-stream", Gst, sink_clock=GstSinkClock(p, 16000))
    print("stream attach:", made.get("attached"), "error:", made.get("error"),
          "elements:", sorted(made.get("elements") or {}))
    # decodebin rejects raw: probe the flow with the same sink chain minus the decoder
    p.set_state(Gst.State.NULL)
    p2 = Gst.parse_launch("appsrc name=source format=bytes block=true max-bytes=1048576 ! "
                          "audioconvert ! audioresample ! autoaudiosink name=sink")
    tl2 = RenderTimeline(sample_rate=16000, clock=time.monotonic)
    tl2.open("g1", "tts-stream")
    tl2.open_turn()
    made2 = attach_render_tap(p2, tl2, "g1", "tts-stream", Gst, sink_clock=GstSinkClock(p2, 16000))
    print("stream(no-decodebin) attach:", made2.get("attached"), "error:", made2.get("error"))
    flow("stream", p2, tl2, made2)

    # 2) the file path from _play_file: a playbin on a real wav
    import os
    import numpy as np
    wav = "/tmp/voxa_probe_tone.wav"
    if not os.path.exists(wav):
        rate = 16000
        t = np.arange(rate, dtype=np.float64)
        data = (np.sin(2 * 3.14159 * 440 * t / rate) * 12000).astype("<i2")
        import wave
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data.tobytes())
    p3 = Gst.ElementFactory.make("playbin")
    p3.set_property("uri", "file://" + wav)
    describe("file", p3)
    tl3 = RenderTimeline(sample_rate=16000, clock=time.monotonic)
    tl3.open("g1", "tts-file")
    tl3.open_turn()
    made3 = attach_render_tap(p3, tl3, "g1", "tts-file", Gst, sink_clock=GstSinkClock(p3, 16000))
    print("file attach:", made3.get("attached"), "error:", made3.get("error"),
          "elements:", sorted(made3.get("elements") or {}))
    flow("file", p3, tl3, made3)


main()
