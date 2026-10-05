#!/usr/bin/env python3
"""Measure real talking faces: turn a video into per-frame blendshape numbers.

Usage: track_faces.py <video-url-or-file> <out.csv> [--seconds 90] [--start 30] [--fps 30]

Runs MediaPipe FaceLandmarker (VIDEO mode, one face, blendshapes + transformation
matrixes) over a clip and writes one CSV row per sampled frame: time, the 52
blendshape scores by name, head yaw/pitch/roll in degrees, and face_found (0/1).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from yt_dlp import YoutubeDL

MODEL = "/home/rhy/voxa-characters/analysis/models/face_landmarker.task"
CACHE_DIR = "/Voxa/.voxa-spec/out/analysis/cache"
NODE = "/home/rhy/.local/bin/node"

BLEND_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekBoneLeft", "cheekBoneRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthDown", "mouthFunnel", "mouthPout", "mouthPressDown", "mouthPressUp",
    "mouthSmileLeft", "mouthSmileRight", "mouthSquareLeft", "mouthSquareRight", "mouthWide",
    "noseScratchLeft", "noseScratchRight",
    "lipLowerInnerLeft", "lipLowerInnerRight", "lipLowerOuterLeft", "lipLowerOuterRight",
    "lipRaiseLeft", "lipRaiseRight", "lipRollInLowerLeft", "lipRollInLowerRight",
    "lipRollInUpperLeft", "lipRollInUpperRight", "lipRollOutLowerLeft", "lipRollOutLowerRight",
    "lipRollOutUpperLeft", "lipRollOutUpperRight", "lipShapeLowerLeft", "lipShapeLowerRight",
    "lipShapeUpperLeft", "lipShapeUpperRight",
]


def euler_deg(matrix):
    """Yaw/pitch/roll in degrees from a 4x4 row-major transformation matrix."""
    r = np.asarray(matrix, dtype=np.float64).reshape(4, 4)[:3, :3]
    pitch = np.degrees(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return float(yaw), float(pitch), float(roll)


def download(url, start, seconds, cache):
    """Download at most 360p of [start, start+seconds] into cache; reuse if present."""
    if cache.exists():
        return str(cache), 0.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    end = start + seconds
    opts = {
        "format": "mp4[height<=360][vcodec^=avc]/best[height<=360][vcodec^=avc]",
        "outtmpl": str(cache),
        "js_runtimes": {"node": {"path": NODE}},
        "quiet": True,
        "no_progress": True,
    }
    opts["download_sections"] = [f"*{int(start)}-{int(end)}"]
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    return str(cache), 0.0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--start", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(argv)

    if Path(args.src).exists():
        video, seek = args.src, 0.0
    else:
        tag = hashlib.sha1(f"{args.src}|{args.start}|{args.seconds}".encode()).hexdigest()[:16]
        cache = Path(CACHE_DIR) / f"{tag}.mp4"
        video, seek = download(args.src, args.start, args.seconds, cache)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native / args.fps)))

    header = ["time"] + BLEND_NAMES + ["yaw", "pitch", "roll", "face_found"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        import csv

        w = csv.writer(fh)
        w.writerow(header)
        frame_no = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if frame_no % step == 0:
                t = seek + frame_no / native
                if t > args.start + args.seconds:
                    break
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect_for_video(mp_img, int(round(t * 1000)))
                scores = {n: 0.0 for n in BLEND_NAMES}
                yaw = pitch = roll = 0.0
                found = 0
                if res.face_blendshapes:
                    found = 1
                    for cat in res.face_blendshapes[0]:
                        name = cat.category_name.lstrip("_")
                        if name in scores:
                            scores[name] = float(cat.score)
                    if res.facial_transformation_matrixes:
                        m = res.facial_transformation_matrixes[0]
                        yaw, pitch, roll = euler_deg(m.matrix if hasattr(m, "matrix") else m)
                row = [round(t, 3)] + [round(scores[n], 4) for n in BLEND_NAMES]
                row += [round(yaw, 2), round(pitch, 2), round(roll, 2), found]
                w.writerow(row)
            frame_no += 1
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
