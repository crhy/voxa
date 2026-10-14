"""Run recorded microphone + reference pairs through the WebRTC echo canceller (GStreamer webrtcdsp), offline.

The elements live in the Voxa Flatpak runtime:
    flatpak run --filesystem=/Voxa --command=python3 io.github.crhy.voxa /Voxa/tools/webrtc_lab.py selftest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SR = 16000
FRAME = SR // 100  # webrtcdsp works in 10 ms frames
ONLY_ECHO = {"echo-cancel": True, "noise-suppression": False, "gain-control": False,
             "high-pass-filter": False, "voice-detection": False, "limiter": False}


def _gst():
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    return Gst


def available() -> bool:
    try:
        gst = _gst()
    except (AttributeError, ImportError, ValueError):
        return False
    return all(gst.ElementFactory.find(name) is not None for name in ("webrtcdsp", "webrtcechoprobe"))


def _push(gst, src, samples: np.ndarray) -> None:
    pcm = np.clip(samples * 32768.0, -32768, 32767).astype("<i2")
    step = gst.SECOND // 100
    for k in range(len(pcm) // FRAME):
        buf = gst.Buffer.new_wrapped(pcm[k * FRAME:(k + 1) * FRAME].tobytes())
        buf.pts = k * step
        buf.duration = step
        src.emit("push-buffer", buf)
    src.emit("end-of-stream")


def process(mic, ref, rate: int = SR, **props) -> np.ndarray:
    """Clean ``mic`` against ``ref`` (float arrays, same time span). Takes as long as the clip (real-time sync)."""
    gst = _gst()
    caps = f"audio/x-raw,format=S16LE,layout=interleaved,channels=1,rate={rate}"
    pipeline = gst.parse_launch(
        f"appsrc name=far format=time caps={caps} ! webrtcechoprobe name=probe ! fakesink sync=true "
        f"appsrc name=near format=time caps={caps} ! webrtcdsp name=dsp probe=probe ! "
        f"appsink name=out sync=true max-buffers=0 drop=false"
    )
    dsp = pipeline.get_by_name("dsp")
    for key, value in {**ONLY_ECHO, **props}.items():
        if dsp.find_property(key) is not None:
            dsp.set_property(key, value)
    n = (min(len(mic), len(ref)) // FRAME) * FRAME
    _push(gst, pipeline.get_by_name("far"), np.asarray(ref[:n], dtype=np.float64))
    _push(gst, pipeline.get_by_name("near"), np.asarray(mic[:n], dtype=np.float64))
    out = pipeline.get_by_name("out")
    pipeline.set_state(gst.State.PLAYING)
    chunks = []
    deadline = time.monotonic() + n / rate + 20.0
    while time.monotonic() < deadline:
        sample = out.emit("try-pull-sample", gst.SECOND)
        if sample is None:
            if out.get_property("eos"):
                break
            continue
        buf = sample.get_buffer()
        chunks.append(np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype="<i2"))
    pipeline.set_state(gst.State.NULL)
    cleaned = np.concatenate(chunks).astype(np.float32) / 32768.0 if chunks else np.zeros(0, np.float32)
    result = np.zeros(len(mic), dtype=np.float32)
    result[:min(len(cleaned), len(result))] = cleaned[:len(result)]
    return result


def erle_tail_db(mic, cleaned, share: float = 0.6) -> float:
    """Echo removed (dB) over the last ``share`` of the clip, after the canceller has had time to converge."""
    n = min(len(mic), len(cleaned))
    a = int(n * (1.0 - share))
    return float(10 * np.log10((np.mean(np.square(mic[a:n])) + 1e-20) / (np.mean(np.square(cleaned[a:n])) + 1e-20)))


def _selftest() -> int:
    if not available():
        print("webrtcdsp / webrtcechoprobe not available here")
        return 1
    dsp = _gst().ElementFactory.make("webrtcdsp")
    print("properties:", ", ".join(f"{p.name}={dsp.get_property(p.name)}" for p in dsp.list_properties()
                                   if p.name not in ("name", "parent", "qos", "probe")))
    rng = np.random.default_rng(1)
    ref = (0.1 * rng.standard_normal(8 * SR)).astype(np.float32)
    h = rng.standard_normal(100) * np.exp(-np.arange(100) / 20.0)
    echo = np.convolve(ref, 0.5 * h / np.linalg.norm(h))[:len(ref)]
    mic = np.concatenate([np.zeros(480), echo])[:len(ref)] + 0.03 * np.std(echo) * rng.standard_normal(len(ref))
    print(f"synthetic echo (30 ms delay, noise 30 dB down): removed {erle_tail_db(mic, process(mic, ref)):.1f} dB")
    for path in sys.argv[2:]:
        m = np.load(path + ".mic.npy" if not path.endswith(".npy") else path)
        r = np.load(path + ".ref.npy" if not path.endswith(".npy") else path.replace("mic", "ref"))
        print(f"{path.rsplit('/', 1)[-1]}: removed {erle_tail_db(m, process(m, r)):.1f} dB")
    return 0


def _batch(jobs_path: str) -> int:
    """Run a list of {"mic","ref","out","props"} jobs through ``process``, saving float32 to "out".

    Jobs whose "out" already exists are skipped, so an interrupted batch can be resumed.
    Prints one progress line per job with wall seconds and ``time.process_time()`` used.
    """
    jobs = json.loads(Path(jobs_path).read_text())
    for j, job in enumerate(jobs):
        out = Path(job["out"])
        if out.exists():
            print(f"[{j}] skip {out.name}")
            continue
        mic = np.load(job["mic"])
        ref = np.load(job["ref"])
        t0 = time.monotonic()
        c0 = time.process_time()
        cleaned = process(mic, ref, **job.get("props", {}))
        wall = time.monotonic() - t0
        cpu = time.process_time() - c0
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.asarray(cleaned, dtype=np.float32))
        print(f"[{j}] {out.name}: {wall:.1f}s wall, {cpu:.1f}s cpu")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["selftest"]:
        raise SystemExit(_selftest())
    if cmd == ["batch"] and len(sys.argv) > 2:
        raise SystemExit(_batch(sys.argv[2]))
    raise SystemExit(2)
