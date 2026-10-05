#!/usr/bin/env python3
"""Summarise face-tracking CSVs into talking-face statistics.

Usage: face_stats.py <csv>... --out stats.json

Computes, over frames where a face was found, blink / gaze / head / brow / mouth
statistics and a few correlations. Pure Python + numpy; importable for tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EYE_BLINK = ("eyeBlinkLeft", "eyeBlinkRight")
EYE_LOOK = ("eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
            "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight")
BROWS = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight")


def load_frames(path):
    rows = []
    with Path(path).open(newline="") as fh:
        import csv

        reader = csv.DictReader(fh)
        for r in reader:
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (ValueError, TypeError):
                continue
    return rows


def found(frames):
    return [f for f in frames if f.get("face_found", 0) >= 0.5]


def _pct(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _median(values):
    return _pct(values, 50)


BLINK_RATE_LO, BLINK_RATE_HI = 6.0, 40.0
BLINK_DUR_LO_MS, BLINK_DUR_HI_MS = 80.0, 450.0


def _blink_events(f):
    """Detect blinks in one clip's face-found frames.

    Returns a list of ``(onset_time, closing_s, opening_s)``. A blink is a
    rise of the mean eyeBlink score above ``baseline + 0.6 * (closed -
    baseline)`` that returns below ``baseline + 0.25 * (closed - baseline)``.
    Closures longer than 500 ms, shorter than 50 ms, or with eyeLookDown
    above 0.5 for the whole event while the score never reaches 0.8 * closed
    are rejected.
    """
    times = [x["time"] for x in f]
    score = [0.5 * (x[EYE_BLINK[0]] + x[EYE_BLINK[1]]) for x in f]
    look_down = [0.5 * (x["eyeLookDownLeft"] + x["eyeLookDownRight"]) for x in f]
    baseline = _pct(score, 20)
    closed = _pct(score, 99.5)
    if closed is None or baseline is None or closed <= baseline:
        return []
    rise = baseline + 0.6 * (closed - baseline)
    fall = baseline + 0.25 * (closed - baseline)
    events = []
    i = 0
    while i < len(score):
        if score[i] > rise:
            start = i
            peak = score[i]
            j = i
            while j + 1 < len(score) and score[j + 1] >= fall:
                j += 1
                peak = max(peak, score[j])
            end = j
            onset, finish = times[start], times[end]
            duration = finish - onset
            if duration > 0.5 or duration < 0.05:
                i = end + 1
                continue
            if (all(ld > 0.5 for ld in look_down[start:end + 1])
                    and peak < 0.8 * closed):
                i = end + 1
                continue
            amp = peak - baseline
            lo10, hi90 = baseline + 0.1 * amp, baseline + 0.9 * amp
            t10_up = next(times[k] for k in range(start, end + 1) if score[k] >= lo10)
            t90_up = next(times[k] for k in range(start, end + 1) if score[k] >= hi90)
            t90_dn = next(times[k] for k in range(end, start - 1, -1) if score[k] >= hi90)
            t10_dn = next(times[k] for k in range(end, start - 1, -1) if score[k] >= lo10)
            events.append((onset, t90_up - t10_up, t10_dn - t90_dn))
            i = end + 1
        else:
            i += 1
    return events


def _blink_summary(clip_events, spans):
    events = [e for ev in clip_events for e in ev]
    if not events:
        return {"rate_per_min": 0.0, "double_fraction": 0.0}
    total_span = sum(spans)
    rate = len(events) / (total_span / 60.0) if total_span > 0 else float(len(events))
    intervals = []
    doubles = 0
    for ev in clip_events:
        starts = [e[0] for e in ev]
        for k in range(len(starts) - 1):
            gap = starts[k + 1] - starts[k]
            intervals.append(gap)
            if gap < 0.5:
                doubles += 1
    durations_ms = [(e[1] + e[2]) * 1000.0 for e in events]
    closing_ms = [e[1] * 1000.0 for e in events]
    opening_ms = [e[2] * 1000.0 for e in events]
    return {
        "rate_per_min": rate,
        "duration_median_ms": _median(durations_ms),
        "duration_p10_ms": _pct(durations_ms, 10),
        "duration_p90_ms": _pct(durations_ms, 90),
        "closing_median_ms": _median(closing_ms),
        "opening_median_ms": _median(opening_ms),
        "double_fraction": doubles / len(events),
        "interval_median_s": _median(intervals),
        "interval_p10_s": _pct(intervals, 10),
        "interval_p90_s": _pct(intervals, 90),
        "blink_onset_times": [e[0] for e in events],
    }


def blink_stats(frames):
    f = found(frames)
    if not f:
        return {}
    span = f[-1]["time"] - f[0]["time"] if len(f) > 1 else 0.0
    return _blink_summary([_blink_events(f)], [span])


def _gaze_vec(x):
    ix = 0.5 * (x["eyeLookInLeft"] + x["eyeLookInRight"])
    ox = 0.5 * (x["eyeLookOutLeft"] + x["eyeLookOutRight"])
    up = 0.5 * (x["eyeLookUpLeft"] + x["eyeLookUpRight"])
    dn = 0.5 * (x["eyeLookDownLeft"] + x["eyeLookDownRight"])
    return np.array([ix - ox, up - dn])


def gaze_stats(frames):
    f = found(frames)
    if not f:
        return {}
    times = [x["time"] for x in f]
    vecs = [_gaze_vec(x) for x in f]
    shifts = []
    for k in range(1, len(f)):
        if times[k] - times[k - 1] <= 0.1 and np.linalg.norm(vecs[k] - vecs[k - 1]) > 0.15:
            shifts.append(times[k])
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    rate = len(shifts) / (span / 60.0) if span > 0 else float(len(shifts))
    fixs = [shifts[k + 1] - shifts[k] for k in range(len(shifts) - 1)]
    at_cam = sum(1 for v in vecs if np.linalg.norm(v) < 0.15) / len(vecs)
    return {
        "shifts_per_min": rate,
        "fixation_median_s": _median(fixs),
        "fixation_p10_s": _pct(fixs, 10),
        "fixation_p90_s": _pct(fixs, 90),
        "at_camera_fraction": at_cam,
        "gaze_shift_times": shifts,
    }


def head_stats(frames):
    f = found(frames)
    if not f:
        return {}
    yaw = np.array([x["yaw"] for x in f])
    pitch = np.array([x["pitch"] for x in f])
    roll = np.array([x["roll"] for x in f])
    times = np.array([x["time"] for x in f])
    speeds = []
    for k in range(1, len(f)):
        dt = times[k] - times[k - 1]
        if dt > 0:
            d = np.array([yaw[k] - yaw[k - 1], pitch[k] - pitch[k - 1], roll[k] - roll[k - 1]])
            speeds.append(np.linalg.norm(d) / dt)
    period = None
    if len(pitch) > 8:
        fps = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 30.0
        sig = pitch - np.mean(pitch)
        mag = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
        mask = (freqs >= 0.2) & (freqs <= 3.0)
        if mask.any():
            idx = np.argmax(np.where(mask, mag, 0.0))
            if freqs[idx] > 0:
                period = 1.0 / freqs[idx]
    return {
        "yaw_std": float(np.std(yaw)),
        "pitch_std": float(np.std(pitch)),
        "roll_std": float(np.std(roll)),
        "angular_speed_median_dps": _median(speeds),
        "pitch_period_s": period,
    }


def brow_stats(frames):
    f = found(frames)
    if not f:
        return {}
    times = [x["time"] for x in f]
    up = [any(x[b] > 0.3 for b in BROWS) for x in f]
    runs = []
    i = 0
    while i < len(up):
        if up[i]:
            j = i
            while j + 1 < len(up) and up[j + 1]:
                j += 1
            runs.append(times[j] - times[i])
            i = j + 1
        else:
            i += 1
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    rate = len(runs) / (span / 60.0) if span > 0 else float(len(runs))
    return {
        "raises_per_min": rate,
        "duration_median_ms": _median([r * 1000.0 for r in runs]),
    }


def mouth_stats(frames):
    f = found(frames)
    if not f:
        return {}
    jaw = np.array([x["jawOpen"] for x in f])
    talking = jaw[jaw > 0.05]
    smile = np.array([0.5 * (x["mouthSmileLeft"] + x["mouthSmileRight"]) for x in f])
    closed = float(np.mean(jaw < 0.05))
    return {
        "jawOpen_mean_talking": float(np.mean(talking)) if talking.size else 0.0,
        "jawOpen_p90_talking": float(np.percentile(talking, 90)) if talking.size else 0.0,
        "mouth_closed_fraction": closed,
        "mouthSmile_mean": float(np.mean(smile)),
    }


def correlations(frames):
    f = found(frames)
    if not f:
        return {}
    jaw = np.array([x["jawOpen"] for x in f])
    brow = np.array([x["browInnerUp"] for x in f])
    if jaw.std() > 0 and brow.std() > 0:
        corr = float(np.corrcoef(jaw, brow)[0, 1])
    else:
        corr = 0.0
    b = blink_stats(frames)
    g = gaze_stats(frames)
    onset = b.get("blink_onset_times", [])
    shifts = g.get("gaze_shift_times", [])
    within = 0
    for s in shifts:
        if any(abs(o - s) <= 0.2 for o in onset):
            within += 1
    frac = within / len(shifts) if shifts else 0.0
    return {"jawOpen_browInnerUp": corr, "blink_on_gaze_shift_fraction": frac}


def _blink_plausible(b):
    if not b:
        return False
    rate = b.get("rate_per_min")
    dur = b.get("duration_median_ms")
    close = b.get("closing_median_ms")
    open_ = b.get("opening_median_ms")
    if rate is None or dur is None or close is None or open_ is None:
        return False
    return (BLINK_RATE_LO <= rate <= BLINK_RATE_HI
            and BLINK_DUR_LO_MS <= dur <= BLINK_DUR_HI_MS
            and close < open_)


def _gaze_plausible(g):
    if not g:
        return False
    rate = g.get("shifts_per_min")
    fix = g.get("fixation_median_s")
    at_cam = g.get("at_camera_fraction")
    return (rate is not None and 5.0 <= rate <= 60.0
            and fix is not None and 0.2 <= fix <= 10.0
            and at_cam is not None and 0.0 <= at_cam <= 1.0)


def _head_plausible(h):
    if not h:
        return False
    vals = [h.get(k) for k in ("yaw_std", "pitch_std", "roll_std",
                               "angular_speed_median_dps")]
    return all(v is not None and 0.0 <= v <= 60.0 for v in vals)


def _brow_plausible(b):
    if not b:
        return False
    rate = b.get("raises_per_min")
    dur = b.get("duration_median_ms")
    return (rate is not None and 0.0 <= rate <= 60.0
            and dur is not None and 100.0 <= dur <= 3000.0)


def _mouth_plausible(m):
    if not m:
        return False
    return all(m.get(k) is not None and 0.0 <= m[k] <= 1.0
               for k in ("jawOpen_mean_talking", "jawOpen_p90_talking",
                         "mouth_closed_fraction", "mouthSmile_mean"))


def _corr_plausible(c):
    if not c:
        return False
    return all(c.get(k) is not None and -1.0 <= c[k] <= 1.0 for k in c)


def summarize(csv_paths, out_path):
    clips = []
    stats = {}
    sections = []
    for p in csv_paths:
        frames = load_frames(p)
        f = found(frames)
        clips.append({
            "file": str(Path(p).name),
            "frames": len(frames),
            "face_coverage": (len(f) / len(frames)) if frames else 0.0,
        })
        sections.append(frames)
    all_frames = []
    clip_events = []
    spans = []
    for frames in sections:
        f = found(frames)
        clip_events.append(_blink_events(f))
        spans.append(f[-1]["time"] - f[0]["time"] if len(f) > 1 else 0.0)
        all_frames.extend(frames)
    blink = _blink_summary(clip_events, spans)
    blink["plausible"] = _blink_plausible(blink)
    gaze = gaze_stats(all_frames)
    gaze["plausible"] = _gaze_plausible(gaze)
    head = head_stats(all_frames)
    head["plausible"] = _head_plausible(head)
    brow = brow_stats(all_frames)
    brow["plausible"] = _brow_plausible(brow)
    mouth = mouth_stats(all_frames)
    mouth["plausible"] = _mouth_plausible(mouth)
    corr = correlations(all_frames)
    corr["plausible"] = _corr_plausible(corr)
    stats = {
        "blink": blink,
        "gaze": gaze,
        "head": head,
        "brow": brow,
        "mouth": mouth,
        "correlations": corr,
        "clips": clips,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(stats, indent=2))
    return stats


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    summarize(args.csv, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
