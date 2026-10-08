#!/usr/bin/env python3
"""Q12 (issue #88): probe whether ONE shared clock fixes mic/monitor alignment.

Run as: PYTHONPATH=$PWD/.venv/lib/python3.14/site-packages timeout 180 python3 tools/clock_probe.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tools.anc_lab import (
    SR,
    align_ends,
    correlation_peak,
    dbfs,
    make_pink,
    oracle_erle_db,
    set_sink_volume,
    shift_forward,
    sink_volume_pct,
    write_wav,
)
from voxa.sync import band_coherence, resample_to

OUT = Path(".voxa-spec/out/anclab")
CLOCK_MD = OUT / "CLOCK.md"
MIC_SOURCE = "alsa_input.usb-Generic_Blue_Microphones_2111BAB03T18-00.analog-stereo"


def _gst():
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


def record_shared(seconds, mic_source, monitor_source, play_wav):
    """One pipeline, one clock, both pulsesrcs; returns (mic_buffers, ref_buffers).

    Every buffer is (pts_ns, samples). Plays play_wav with paplay on the default sink
    0.5 s after the pipeline reaches PLAYING.
    """
    gst = _gst()
    collected = {"mic": [], "ref": []}

    def pull(sink, key):
        sample = sink.emit("pull-sample")
        if sample is None:
            return gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        ok, mapped = buf.map(gst.MapFlags.READ)
        if not ok:
            return gst.FlowReturn.ERROR
        try:
            pcm = bytes(mapped.data)
        finally:
            buf.unmap(mapped)
        pts = int(buf.pts)
        collected[key].append((pts, np.frombuffer(pcm, dtype="<i2").astype(np.float32)))
        return gst.FlowReturn.OK

    pipeline = gst.Pipeline.new("clock-probe")
    for key, device in (("mic", mic_source), ("ref", monitor_source)):
        src = gst.ElementFactory.make("pulsesrc", f"src-{key}")
        conv = gst.ElementFactory.make("audioconvert", f"conv-{key}")
        res = gst.ElementFactory.make("audioresample", f"res-{key}")
        caps = gst.ElementFactory.make("capsfilter", f"caps-{key}")
        sink = gst.ElementFactory.make("appsink", f"sink-{key}")
        if not all((src, conv, res, caps, sink)):
            pipeline.set_state(gst.State.NULL)
            raise RuntimeError("GStreamer elements missing")
        src.set_property("device", device)
        src.set_property("do-timestamp", True)
        src.set_property("provide-clock", False)
        if src.find_property("latency") is not None:
            src.set_property("latency", 20)
        caps.set_property(
            "caps", gst.Caps.from_string(f"audio/x-raw,format=S16LE,channels=1,rate={SR}")
        )
        sink.set_property("sync", False)
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", pull, key)
        for element in (src, conv, res, caps, sink):
            pipeline.add(element)
        if not (src.link(conv) and conv.link(res) and res.link(caps) and caps.link(sink)):
            pipeline.set_state(gst.State.NULL)
            raise RuntimeError("could not connect pipeline")
    pipeline.use_clock(gst.SystemClock.obtain())
    pipeline.set_state(gst.State.PLAYING)
    time.sleep(0.5)
    proc = subprocess.Popen(["paplay", str(play_wav)])
    time.sleep(seconds)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    pipeline.set_state(gst.State.PAUSED)
    return collected["mic"], collected["ref"]


def to_timeline(buffers, rate, t0_ns):
    """Place each (pts_ns, samples) buffer at round((pts - t0)/1e9*rate) in one array.

    Gaps stay zero. Returns (array, number of gaps over 5 ms, number of overlaps).
    """
    placed = []
    for pts, samples in buffers:
        pos = round((pts - t0_ns) / 1e9 * rate)
        placed.append((pos, np.asarray(samples, dtype=np.float64)))
    if not placed:
        return np.zeros(0, dtype=np.float64), 0, 0
    start = min(p for p, _ in placed)
    end = max(p + len(s) for p, s in placed)
    out = np.zeros(end - start, dtype=np.float64)
    written = np.zeros(end - start, dtype=bool)
    overlaps = 0
    for pos, s in placed:
        i = pos - start
        if bool(np.any(written[i : i + len(s)])):
            overlaps += 1
        out[i : i + len(s)] = s
        written[i : i + len(s)] = True
    min_gap = int(round(0.005 * rate))
    gaps = 0
    i = 0
    n = len(written)
    while i < n:
        if written[i]:
            i += 1
            continue
        j = i
        while j < n and not written[j]:
            j += 1
        if j - i > min_gap:
            gaps += 1
        i = j
    return out, gaps, overlaps


def lag_track(mic, ref, win_s=2.0):
    """Cross-correlation lag (0 to 300 ms, causal) per window; returns (lags, slope ppm, largest jump)."""
    win = int(round(win_s * SR))
    maxd = int(round(0.3 * SR))
    n = min(len(mic), len(ref))
    lags = []
    centers = []
    start = 0
    while start + win <= n:
        m = np.asarray(mic[start : start + win], dtype=np.float64)
        r = np.asarray(ref[start : start + win], dtype=np.float64)
        if float(np.mean(m**2)) > 1e-8 and float(np.mean(r**2)) > 1e-8:
            size = 1
            while size < 2 * win:
                size <<= 1
            corr = np.fft.irfft(np.fft.rfft(m, size) * np.conj(np.fft.rfft(r, size)))
            window = np.abs(corr[: maxd + 1])
            lags.append(int(np.argmax(window)))
            centers.append(start + win / 2.0)
        start += win
    if len(lags) < 2:
        return lags, 0.0, 0
    ts = np.array(centers, dtype=np.float64)
    ds = np.array(lags, dtype=np.float64)
    tc = ts - float(ts.mean())
    tt = float(np.dot(tc, tc))
    slope = float(np.dot(tc, ds - float(ds.mean())) / tt) if tt > 0 else 0.0
    jumps = np.diff(ds)
    largest = int(np.max(np.abs(jumps))) if len(jumps) else 0
    return lags, slope * 1e6, largest


def row(mic, ref, gaps):
    """Metrics for one pairing: echo/noise dB, coherence, oracle ERLE, slope, jump, gaps."""
    mic, ref = align_ends(mic, ref)
    floor = dbfs(mic[: 2 * SR])
    echo_db = dbfs(mic) - floor
    lag, _ = correlation_peak(mic, ref)
    coh = band_coherence(mic, shift_forward(ref, lag))
    erle = oracle_erle_db(mic, ref, lag)
    _, slope_ppm, jump = lag_track(mic, ref)
    return {
        "echo_db": echo_db,
        "coh": coh,
        "erle": erle,
        "slope_ppm": slope_ppm,
        "jump": jump,
        "gaps": gaps,
    }


def main():
    sink = subprocess.run(
        ["pactl", "get-default-sink"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not sink:
        raise SystemExit("no default sink")
    monitor = f"{sink}.monitor"
    old = sink_volume_pct(sink)
    try:
        set_sink_volume(sink, 100.0)
        OUT.mkdir(parents=True, exist_ok=True)
        wav = OUT / "clock_probe_pink.wav"
        pink = make_pink(7, 20.0)
        write_wav(wav, pink)
        mic_bufs, ref_bufs = record_shared(21.0, MIC_SOURCE, monitor, wav)
    finally:
        if old is not None:
            set_sink_volume(sink, old)

    all_pts = [p for p, _ in mic_bufs] + [p for p, _ in ref_bufs]
    if not all_pts:
        raise SystemExit("no buffers captured")
    t0 = min(all_pts)
    mic_a, gaps_m, _ = to_timeline(mic_bufs, SR, t0)
    ref_a, gaps_r, _ = to_timeline(ref_bufs, SR, t0)
    mic_b = np.concatenate([s for _, s in mic_bufs]) if mic_bufs else np.zeros(0)
    ref_b = np.concatenate([s for _, s in ref_bufs]) if ref_bufs else np.zeros(0)

    rows = []
    rows.append(("a shared clock, timestamp timeline", row(mic_a, ref_a, gaps_m + gaps_r)))
    rows.append(("b concatenated by sample count", row(mic_b, ref_b, 0)))
    rows.append(("c monitor vs played file", row(ref_a, pink, gaps_r)))

    slope_a = rows[0][1]["slope_ppm"]
    if abs(slope_a) > 20.0:
        mic_d = resample_to(mic_a, len(mic_a), ratio=1.0 + slope_a * 1e-6)
        rows.append(("d mic resampled by fitted slope", row(mic_d, ref_a, gaps_m + gaps_r)))

    header = (
        "| pairing | echo/noise dB | coherence | oracle ERLE dB | slope ppm | max jump | gaps |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {name} | {r['echo_db']:.1f} | {r['coh']:.3f} | {r['erle']:.1f} "
        f"| {r['slope_ppm']:.1f} | {r['jump']} | {r['gaps']} |"
        for name, r in rows
    ]
    block = header + "\n".join(lines) + "\n"
    print(block)
    with CLOCK_MD.open("a", encoding="utf-8") as fh:
        fh.write(block)


if __name__ == "__main__":
    main()
