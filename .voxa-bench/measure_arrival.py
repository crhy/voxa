#!/usr/bin/env python3
"""Task v11e step 2: what reaches a given microphone when VoxaTest-style speech is played.

Plays the harness's own test phrase (edge TTS if online, else the shipped sample) at the
FULL stream gain (100%) through the default sink, records the named source, and prints
the noise floor, peak and gated-RMS of the arrival in dBFS. Changes no volume.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path("/home/rhy/strata-fleet/repos/voxa")
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from voxatest_volume_check import (  # noqa: E402
    MicCapture, RATE, dbfs, gated_rms_dbfs, make_phrase, rms_dbfs)


class SourceCapture(MicCapture):
    """Same capture, but from an explicit PulseAudio source instead of the default."""

    def record(self, seconds: float, after_start=None):
        import numpy as np

        self.chunks = []
        gst = self.gst
        src = gst.ElementFactory.make("pulsesrc")
        src.set_property("device", self.source)
        conv, res, caps, sink = (gst.ElementFactory.make(n) for n in
                                 ("audioconvert", "audioresample", "capsfilter", "appsink"))
        caps.set_property("caps", gst.Caps.from_string("audio/x-raw,format=S16LE,channels=1,rate=16000"))
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)

        def on_sample(pad):
            sample = pad.emit("pull-sample")
            buf = sample.get_buffer()
            data = np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype=np.int16)
            self.chunks.append(data.astype(np.float32) / 32768.0)
            return gst.FlowReturn.OK

        sink.connect("new-sample", on_sample)
        pipe = gst.Pipeline.new("arrival")
        for el in (src, conv, res, caps, sink):
            pipe.add(el)
        src.link(conv)
        conv.link(res)
        res.link(caps)
        caps.link(sink)
        if pipe.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"could not open {self.source}")
        if after_start is not None:
            time.sleep(0.3)
            after_start()
        time.sleep(seconds + 0.3)
        pipe.set_state(gst.State.NULL)
        if not self.chunks:
            raise RuntimeError("microphone produced no audio")
        return np.concatenate(self.chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="PulseAudio source name to record from")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    cap = SourceCapture()
    cap.source = args.source
    out = {"source": args.source, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    vol = subprocess.run(["pactl", "get-source-volume", args.source],
                         capture_output=True, text=True).stdout.strip().replace("\n", " ")
    out["source_volume"] = vol
    sink = subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                          capture_output=True, text=True).stdout.strip().replace("\n", " ")
    out["sink_volume"] = sink
    with tempfile.TemporaryDirectory() as tmp:
        phrase = make_phrase(Path(tmp))
        rounds = []
        for _ in range(args.repeat):
            noise = cap.record(3.0)
            noise_db = rms_dbfs(noise)
            proc: list = []

            def start_play() -> None:
                proc.append(subprocess.Popen(["paplay", str(phrase)],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

            arrived = cap.record(6.0, after_start=start_play)
            if proc:
                proc[0].wait()
            rounds.append({"noise_rms_dbfs": round(noise_db, 1),
                           "arrival_peak_dbfs": round(dbfs(arrived), 1),
                           "arrival_rms_dbfs": round(gated_rms_dbfs(arrived, 10.0 ** (noise_db / 20.0)), 1)})
        out["rounds"] = rounds
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
