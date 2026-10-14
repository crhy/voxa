"""Render head-pose-locked clips for portraits: speech and silence.

One process, models loaded once. Run inside the sandbox from the
ditto-talkinghead folder.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, ".")
from inference import run, StreamSDK

N = "/home/rhy/voxa-neural"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("characters", nargs="*", help="optional character IDs to render")
parser.add_argument("--inputs", default=f"{N}/inputs", help="portrait PNG directory")
parser.add_argument("--output", default=f"{N}/out/locked", help="clip output directory")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
sdk = StreamSDK("./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", "./checkpoints/ditto_pytorch")
lock = {"setup_kwargs": {"use_d_keys": ("exp",)}}
only = set(args.characters)
for src in sorted(glob.glob(os.path.join(args.inputs, "*.png"))):
    cid = os.path.splitext(os.path.basename(src))[0]
    if only and cid not in only: continue
    for kind, wav in (("speech", "./example/audio.wav"), ("silence", f"{N}/out/silence.wav")):
        t = time.time()
        output = os.path.join(args.output, f"{cid}-{kind}.mp4")
        run(sdk, wav, src, output, dict(lock))
        # Ditto writes the rendered video before optionally muxing its source
        # audio with FFmpeg.  Audio is unnecessary for face-pack extraction, so
        # preserve the fresh video when FFmpeg is not installed in the sandbox.
        temporary_output = f"{output}.tmp.mp4"
        if os.path.exists(temporary_output) and (
            not os.path.exists(output) or os.path.getmtime(output) < t
        ):
            os.replace(temporary_output, output)
        print(f"DONE {cid} {kind} {time.time() - t:.1f}s", flush=True)
