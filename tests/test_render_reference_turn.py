"""Issue #72, task v08 step 2: ONE continuous reference per TURN, not one playbin per piece.

In the rc4 LIVE run every silent-reference decision landed within ~1 s of a piece boundary: the
reference attribution reads covers only the CURRENT piece (new playbin + tap per piece), so the
echo tail of the previous piece (room + microphone latency, up to ~1.5 s) met a SILENT reference
and her own voice was credited to the user - she answered herself 79 times.

This test drives the real GStreamer (subprocess, fakesink: no sound device, no microphone) over
both shipped speech paths ('piece' and 'whole-reply', muted alternating like the log), speaking
ONE reply as THREE pieces with real 0.4 s gaps, and asserts that the reference the attribution
reads is non-silent and sample-aligned at every instant from the turn's first speak-start until
the last piece's end plus a tail - including the gaps between pieces and looking back from the
first moments of each new piece. The pieces are a ramp whose SAMPLE VALUE is its position in the
output stream, so a window that is a contiguous ramp is aligned with what reached the sink by
construction.

On the pre-fix code this FAILS: during a gap ``reference_window`` maps the moment past the end
of the render and zero-pads, i.e. the attribution reads silent mid-reply (the logged
``reference silent: no echo possible``).
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

RATE = 16000
PIECE = 4800  # 0.3 s per piece
GAP = 0.4     # seconds between pieces of the same reply (the log's ~1 s boundary scale)
TAIL = 1.0    # polled past the last piece; < ECHO_TAIL_SECONDS (1.5 s)


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

RATE, PIECE, GAP, TAIL = 16000, 4800, 0.4, 1.0
MODE = sys.argv[2]
WAVS = sys.argv[3:]

timeline = RenderTimeline(sample_rate=RATE, clock=time.monotonic)
# Task v08: one reply is ONE turn. On pre-fix code there is no turn concept at all - the test
# then reads through the OLD reference_window and must fail on the silent-gap behaviour, not
# crash on a missing attribute.
open_turn = getattr(timeline, "open_turn", None)
turn_reference_window = getattr(timeline, "turn_reference_window", None)
if open_turn is not None:
    open_turn()

def attach(pipeline, generation, path):
    made = attach_render_tap(pipeline, timeline, generation, path, Gst,
                             sink_clock=GstSinkClock(pipeline, RATE))
    if not made.get("attached"):
        print("attached 0", made.get("error") or "")
        sys.exit(1)
    return made

cum = {"n": 0}
def watch(sink):
    def on_buffer(pad, info):
        cum["n"] += info.get_buffer().get_size() // 2
        return Gst.PadProbeReturn.OK
    sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, on_buffer)

reads = []  # (moment, samples_received-at-sink, window, idle)
def read_at(moment, idle):
    if turn_reference_window is not None:
        window = turn_reference_window(moment=moment, length=1600)
    else:
        window = timeline.reference_window(mic_start=moment, length=1600)
    expected = min(int(timeline.index_at_audible_time(moment)), int(timeline.total))
    reads.append((float(moment), int(cum["n"]), window, bool(idle), expected))

def drain(pipeline, want):
    # Poll DURING playback too: the first moments of a new piece must look back at the previous
    # piece's tail (the echo lands while the next piece is already playing).
    t0 = time.monotonic()
    next_read = 0.0
    while cum["n"] < want and time.monotonic() - t0 < 8.0:
        if time.monotonic() >= next_read:
            read_at(time.monotonic(), False)
            next_read = time.monotonic() + 0.05
        time.sleep(0.01)
    return cum["n"] >= want

if MODE == "piece":
    # The prepared path: every piece is its OWN playbin + tap (the rc4 bug: a new reference per
    # piece, while the reply is one thing she said).
    for k, wav in enumerate(WAVS):
        pipeline = Gst.parse_launch("playbin uri=file://" + wav)
        made = attach(pipeline, f"g{k}", "tts-file")
        watch(made["elements"]["autoaudiosink"])
        pipeline.set_state(Gst.State.PLAYING)
        if not drain(pipeline, (k + 1) * PIECE):
            print("drain_timeout", k)
            sys.exit(1)
        # Poll the tail of this piece's playback and the whole gap after it: the reference must
        # stay hers through the boundary.
        t_end = time.monotonic()
        for _ in range(int(GAP / 0.05)):
            read_at(time.monotonic(), True)
            time.sleep(0.05)
        pipeline.set_state(Gst.State.NULL)
        time.sleep(0.05)
else:
    # The whole-reply streaming path: ONE appsrc stream spoken with gaps between segments.
    pipeline = Gst.parse_launch(
        "appsrc name=source format=bytes block=true max-bytes=1048576 ! "
        "queue max-size-bytes=2097152 ! audioconvert ! audioresample ! identity name=sink sync=false")
    attach(pipeline, "g0", "tts-stream")
    watch(pipeline.get_by_name("sink"))
    appsrc = pipeline.get_by_name("source")
    appsrc.set_property("caps", Gst.Caps.from_string(
        "audio/x-raw,format=S16LE,channels=1,rate=16000,layout=interleaved"))
    pipeline.set_state(Gst.State.PLAYING)
    for k in range(len(WAVS)):
        first = k * PIECE
        for i in range(first, first + PIECE, 1600):
            chunk = np.arange(RATE, dtype="<i2")[i:i + 1600].tobytes()
            buf = Gst.Buffer.new_allocate(None, len(chunk), None)
            buf.fill(0, chunk)
            appsrc.emit("push-buffer", buf)
            time.sleep(0.01)
        if not drain(pipeline, first + PIECE):
            print("drain_timeout", k)
            sys.exit(1)
        for _ in range(int(GAP / 0.05)):
            read_at(time.monotonic(), True)
            time.sleep(0.05)
    appsrc.emit("end-of-stream")
    time.sleep(0.1)
    pipeline.set_state(Gst.State.NULL)

# The echo tail: her voice is still in the microphone for up to 1.5 s after the last sample left
# the speakers; the reference must still be hers.
t0 = time.monotonic()
while time.monotonic() - t0 < TAIL:
    read_at(time.monotonic(), True)
    time.sleep(0.05)

total = len(WAVS) * PIECE
silent = discontiguous = wrongtail = 0
for moment, heard_so_far, window, idle, expected in reads:
    if heard_so_far == 0:
        continue  # before the first sample left the speakers there is no echo to reference
    first_audible = timeline.audible_time(0) if timeline.total else None
    if first_audible is not None and moment < first_audible:
        continue  # nothing is audible at the microphone yet: silence would be honest
    # The pipeline hands the tap int16 normalised to float; scale back so the ramp's step is 1
    # and the sample VALUE is again its position in the output stream.
    arr = np.asarray(window, dtype=np.float64) * 32768.0
    # Front padding = positions before her reply was rendered at all (the microphone could not
    # carry them); strip it, then what remains must be ONE contiguous run of stream positions.
    arr = arr[np.argmax(arr != 0.0):] if np.any(arr != 0.0) else arr
    if arr.size == 0 or np.allclose(arr, 0.0):
        silent += 1
        continue
    if not np.allclose(np.diff(arr), 1.0):
        discontiguous += 1
        continue
    if idle:
        # A moment outside playback (gap or tail): the window must END exactly where the
        # render is at that moment - the tail of what reached the sink, not a guess past it.
        end = int(round(arr[-1])) + 1
        if abs(end - expected) > 320:
            wrongtail += 1
print("attached 1 moments", len(reads), "silent", silent,
      "discontiguous", discontiguous, "wrongtail", wrongtail,
      "sink_samples", cum["n"], "ring_total", int(timeline.total), "want", total)
"""


def _ramp_wavs(tmp_path, count: int) -> list[str]:
    """Piece k holds the output-stream positions [k*PIECE, (k+1)*PIECE) as its sample VALUES, so
    any window that is a contiguous ramp is sample-aligned by construction."""
    import wave

    paths = []
    for k in range(count):
        values = np.arange(k * PIECE, (k + 1) * PIECE, dtype="<i2")
        path = tmp_path / f"piece{k}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(RATE)
            handle.writeframes(values.tobytes())
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("mode", ("piece", "whole-reply"))
@pytest.mark.parametrize("muted", (True, False), ids=("muted", "unmuted"))
def test_one_reply_is_one_continuous_reference(tmp_path, mode, muted) -> None:
    """Three pieces, two real gaps, one tail: the reference read never goes silent or misaligned.

    ``muted`` is the capture-side act of muting the microphone (as in the log: the failing lines
    alternate ``muted=False``/``muted=True`` across pieces) - it must never change what the
    render tap recorded.
    """
    python = _system_python_with_gst()
    if python is None:
        pytest.skip("no gi/GStreamer interpreter available")

    wavs = _ramp_wavs(tmp_path, 3)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, VOXA_MUTED=repr(muted))
    proc = subprocess.run([python, "-c", PROBE, repo, mode, *wavs],
                          capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    fields = proc.stdout.split()
    assert fields[1] == "1", f"the render tap did not attach on the {mode} path: {proc.stdout}"
    got = {key: int(fields[fields.index(key) + 1])
           for key in ("silent", "discontiguous", "wrongtail", "sink_samples", "ring_total",
                       "want", "moments")}

    assert got["moments"] >= 3 * int(GAP / 0.05), "the pieces/gaps were not polled through"
    assert got["sink_samples"] == got["want"] == 3 * PIECE, "her playback did not reach the sink"
    assert got["ring_total"] >= got["want"], "the reference lost pieces of the reply"
    assert got["silent"] == 0, (
        f"reference silent between the pieces of ONE reply on the {mode} path (muted={muted}): "
        "the attribution read the gap of her own reply as silence - the logged "
        "'reference silent: no echo possible' that made her answer herself")
    assert got["discontiguous"] == 0, (
        f"reference not sample-aligned across piece boundaries on the {mode} path "
        f"(muted={muted}): the window is not a contiguous run of output-stream positions")
    assert got["wrongtail"] == 0, (
        f"reference window not at the render tail at the moment read on the {mode} path "
        f"(muted={muted})")
