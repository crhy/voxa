"""Issue #72, step 2: on the REAL rc3 code paths the reference buffer holds her samples.

In the shipped rc3 every heard line said ``reference silent: no echo possible`` - the reference
of what Voxa played was EMPTY while she spoke, so no canceller had anything to cancel and she
interrupted herself 79 times. The reason: ``_play_file`` (voxa/speech.py) hands a **playbin**
the file, and a playbin builds its sink chain inside itself - the tap in ``attach_render_tap``
looked for a named sink, never found one, and reported "no sink element to tap in front of".

These tests drive the real GStreamer (a subprocess with a gi-capable interpreter; fakesink, so
no sound is played and no microphone is opened) over both shipped speech paths:

* ``piece`` - a prepared wav file through ``playbin`` (the path every prepared reply uses);
* ``whole-reply`` - the streaming appsrc pipeline the live reply uses,

each with the conversation muted True and False (the v05 log lines were exactly
``path='whole-reply' muted=True``; muting the microphone is a capture-side act and must never
change what the reference recorded).

The contract: while Voxa speaks, the reference buffer (the render timeline) holds her samples -
non-silent, complete, and aligned with what reached the sink to within 5 ms.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

RATE = 16000


def _system_python_with_gst() -> str | None:
    """A CPython that can import gi/Gst, so the fix can be proven on the real library."""
    import shutil

    for candidate in (sys.executable, "/usr/bin/python3", shutil.which("python3")):
        if not candidate:
            continue
        probe = subprocess.run(
            [candidate, "-c", "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst"],
            capture_output=True, timeout=15,
        )
        if probe.returncode == 0:
            return candidate
    return None


PROBE = r"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
os.environ["GST_AUDIOSINK"] = "fakesink"   # the tap must work with no sound device in use
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
Gst.init(None)
import numpy as np
from voxa.timeline import RenderTimeline, GstSinkClock, attach_render_tap

RATE = 16000
MODE = sys.argv[2]
pipeline = (Gst.parse_launch("playbin uri=file://" + sys.argv[3]) if MODE == "piece"
            else Gst.parse_launch(
                "appsrc name=source format=bytes block=true max-bytes=1048576 ! "
                "queue max-size-bytes=2097152 ! audioconvert ! audioresample ! identity name=sink sync=false"))
timeline = RenderTimeline(sample_rate=RATE, clock=time.monotonic)
timeline.open("g1", "tts-file" if MODE == "piece" else "tts-stream")
made = attach_render_tap(pipeline, timeline, "g1",
                         "tts-file" if MODE == "piece" else "tts-stream",
                         Gst, sink_clock=GstSinkClock(pipeline, RATE))
print("attached", int(bool(made.get("attached"))), made.get("error") or "")
if not made.get("attached"):
    sys.exit(1)

sink = made["elements"]["autoaudiosink"] if MODE == "piece" else pipeline.get_by_name("sink")
state = {"cum": 0}
def on_buffer(pad, info):
    # What the sink branch receives: the sample count tells us how far playback reached, and
    # the buffer CONTENTS (a ramp, where the value is the index) prove the tap branch stored
    # the same samples at the same positions - sample-exact, which is the 5 ms contract's
    # stronger form. Timing itself belongs to the sink clock (tested in test_tts_reference.py).
    state["cum"] += info.get_buffer().get_size() // 2
    return Gst.PadProbeReturn.OK
sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_buffer)

if MODE == "piece":
    pipeline.set_state(Gst.State.PLAYING)
else:
    appsrc = pipeline.get_by_name("source")
    appsrc.set_property("caps", Gst.Caps.from_string(
        "audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved"))
    pipeline.set_state(Gst.State.PLAYING)
    for i in range(0, len(np.arange(RATE, dtype="<i2")), 1600):
        chunk = np.arange(RATE, dtype="<i2")[i:i + 1600].tobytes()
        buf = Gst.Buffer.new_allocate(None, len(chunk), None)
        buf.fill(0, chunk)
        appsrc.emit("push-buffer", buf)
        time.sleep(0.02)
    appsrc.emit("end-of-stream")

loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
def on_message(b, m):
    if m.type in (Gst.MessageType.ERROR, Gst.MessageType.EOS):
        loop.quit()
bus.connect("message", on_message)
GLib.timeout_add_seconds(8, loop.quit)
loop.run()
pipeline.set_state(Gst.State.NULL)

cum = state["cum"]
heard = timeline.ring.window(0, RATE).reshape(-1)[:RATE]
if MODE == "piece":
    want = np.arange(RATE, dtype=np.float32) / 32768.0
else:
    want = np.arange(RATE, dtype=np.float32) / 32768.0
silent = bool(np.allclose(heard, 0.0))
aligned = bool(np.allclose(heard, want, atol=1e-3))
print("sink_samples", cum, "ring_total", int(timeline.total),
      "silent", int(silent), "aligned", int(aligned))
"""


def _ramp_wav(path: str) -> None:
    import wave

    values = np.arange(RATE, dtype="<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(values.tobytes())


@pytest.mark.parametrize("mode", ("piece", "whole-reply"))
@pytest.mark.parametrize("muted", (True, False), ids=("muted", "unmuted"))
def test_while_she_speaks_the_reference_holds_her_samples(tmp_path, mode, muted) -> None:
    """Non-silent, complete, and aligned to playback within 5 ms - muted or not.

    ``muted`` is the capture-side ConversationController act (the v05 lines were
    ``path='whole-reply' muted=True``): muting the microphone must never change what the
    render tap recorded, so both states must come out identical and non-silent.
    """
    python = _system_python_with_gst()
    if python is None:
        pytest.skip("no gi/GStreamer interpreter available")

    wav = tmp_path / "reply.wav"
    _ramp_wav(str(wav))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, VOXA_MUTED=repr(muted))
    proc = subprocess.run([python, "-c", PROBE, repo, mode, str(wav)],
                          capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fields = proc.stdout.split()
    attached = int(fields[1])
    sink_samples = int(fields[fields.index("sink_samples") + 1])
    ring_total = int(fields[fields.index("ring_total") + 1])
    silent = int(fields[fields.index("silent") + 1])
    aligned = int(fields[fields.index("aligned") + 1])

    assert attached == 1, f"the render tap did not attach on the {mode} path: {proc.stdout}"
    assert sink_samples == RATE, "her playback did not reach the sink"
    assert silent == 0, f"reference silent on the {mode} path (muted={muted})"
    assert ring_total >= RATE, "the reference buffer did not hold all of her samples"
    assert aligned == 1, (
        f"reference misaligned with playback on the {mode} path (muted={muted}): the samples "
        "the sink received are not the samples the ring holds at the same positions")
