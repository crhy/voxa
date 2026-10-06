from __future__ import annotations

import csv
from pathlib import Path

from tools import face_stats


def _write_csv(path, rows):
    cols = ["time", "face_found", "eyeBlinkLeft", "eyeBlinkRight",
            "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
            "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
            "yaw", "pitch", "roll"]
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, 0.0) for c in cols})
    return path


def _blank(t):
    return {"time": t, "face_found": 1}


def test_blink_stats(tmp_path):
    rows = []
    fps = 30.0
    pattern = [0.6, 0.9, 0.9, 0.45, 0.2]
    for i in range(int(60 * fps)):
        t = i / fps
        r = _blank(t)
        for k in range(12):
            start = 2.0 + k * 5.0
            idx = int((t - start) * fps)
            if 0 <= idx < len(pattern):
                v = pattern[idx]
                r["eyeBlinkLeft"] = v
                r["eyeBlinkRight"] = v
        for g in range(3):
            gs = 4.0 + g * 5.0
            if gs <= t < gs + 2.0:
                r["eyeBlinkLeft"] = 0.6
                r["eyeBlinkRight"] = 0.6
                r["eyeLookDownLeft"] = 0.8
                r["eyeLookDownRight"] = 0.8
        rows.append(r)
    frames = face_stats.load_frames(_write_csv(tmp_path / "blink.csv", rows))
    s = face_stats.blink_stats(frames)
    assert len(s["blink_onset_times"]) == 12
    assert 11.0 <= s["rate_per_min"] <= 13.0
    assert s["double_fraction"] == 0.0
    assert 4.5 <= s["interval_median_s"] <= 5.5


def test_gaze_stats(tmp_path):
    rows = []
    fps = 30.0
    for i in range(int(10 * fps)):
        t = i / fps
        r = _blank(t)
        if int(t) % 2 == 1:
            r["eyeLookInLeft"] = 0.6
            r["eyeLookInRight"] = 0.6
        rows.append(r)
    frames = face_stats.load_frames(_write_csv(tmp_path / "gaze.csv", rows))
    s = face_stats.gaze_stats(frames)
    assert 40.0 <= s["shifts_per_min"] <= 60.0
    assert 0.8 <= s["fixation_median_s"] <= 1.2
    assert 0.4 <= s["at_camera_fraction"] <= 0.6


def test_head_stats(tmp_path):
    rows = []
    fps = 30.0
    for i in range(int(10 * fps)):
        t = i / fps
        r = _blank(t)
        r["yaw"] = 10.0 if i % 2 == 0 else -10.0
        r["pitch"] = 5.0 * (1 if (i % 30) < 15 else -1)
        rows.append(r)
    frames = face_stats.load_frames(_write_csv(tmp_path / "head.csv", rows))
    s = face_stats.head_stats(frames)
    assert 9.0 <= s["yaw_std"] <= 11.0
    assert s["roll_std"] == 0.0
    assert 500.0 <= s["angular_speed_median_dps"] <= 700.0
