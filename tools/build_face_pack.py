#!/usr/bin/env python3
"""Build a GPU-free "face pack" (mouth frames + blink patches) from locked clips.

Usage: build_face_pack.py <id> [--clips DIR] [--out DIR] [--size 512] [--max-mouth 120]

Reads ``<id>-speech.mp4`` and ``<id>-silence.mp4`` (head-locked, 25 fps), measures
every frame with MediaPipe FaceLandmarker (IMAGE mode, pixel coordinates), then
writes a folder of JPEG mouth frames, three eye-blink crops and an ``index.json``
that :mod:`voxa.ui.face_pack` can play.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL = "/home/rhy/voxa-characters/analysis/models/face_landmarker.task"
DEFAULT_CLIPS = "/home/rhy/voxa-neural/out/locked"

EYE_LANDMARKS = (33, 133, 159, 145, 263, 362, 386, 374)
BROW_LANDMARKS = (70, 105, 300, 334)
JPEG_QUALITY = 88


def _options() -> mp.tasks.vision.FaceLandmarker:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _measure(landmarks: list) -> tuple[float, float, float, float] | None:
    """Return (E, open, width, eye) in pixel units, or None if a landmark is missing."""
    if len(landmarks) < 478:
        return None

    def point(index: int) -> complex:
        lm = landmarks[index]
        return complex(lm.x, lm.y)

    e = abs(point(33) - point(263))
    if e <= 1e-6:
        return None
    open_ = float(abs(point(13) - point(14)) / e)
    width = float(abs(point(61) - point(291)) / e)
    eye = float((abs(point(159) - point(145)) + abs(point(386) - point(374))) / (2.0 * e))
    return (float(e), open_, width, eye)


def _read_clip(landmarker: mp.tasks.vision.FaceLandmarker, path: Path, size: int) -> list:
    """Return a list of (image_rgb_uint8, measurement_or_None) for every frame."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Square crop first (top-biased, the same crop the still portraits use) so faces are never squashed.
        h, w = rgb.shape[:2]
        if h > w:
            top = int((h - w) * 0.25)
            rgb = rgb[top:top + w]
        elif w > h:
            left = (w - h) // 2
            rgb = rgb[:, left:left + h]
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)
        if res.face_landmarks:
            frames.append((rgb, _measure(res.face_landmarks[0])))
        else:
            frames.append((rgb, None))
    cap.release()
    return frames


def _farthest_point(candidates: list, max_keep: int) -> list:
    """Greedy farthest-point sampling over (open, width); start at smallest open."""
    if not candidates:
        return []
    start = min(range(len(candidates)), key=lambda i: candidates[i][0])
    selected = [start]
    remaining = set(range(len(candidates))) - {start}
    while remaining and len(selected) < max_keep:
        best_i = None
        best_d = -1.0
        for i in remaining:
            ox, ow = candidates[i][0], candidates[i][1]
            min_d = min(
                (ox - candidates[j][0]) ** 2 + (ow - candidates[j][1]) ** 2 for j in selected
            )
            if min_d > best_d:
                best_d = min_d
                best_i = i
        selected.append(best_i)
        remaining.discard(best_i)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("id")
    parser.add_argument("--clips", default=DEFAULT_CLIPS)
    parser.add_argument("--out", default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-mouth", type=int, default=120)
    args = parser.parse_args()

    clips_dir = Path(args.clips)
    out_dir = Path(args.out) if args.out else Path.home() / ".local/share/voxa/faces" / args.id
    size = args.size
    out_dir.mkdir(parents=True, exist_ok=True)

    landmarker = _options()
    speech = _read_clip(landmarker, clips_dir / f"{args.id}-speech.mp4", size)
    silence = _read_clip(landmarker, clips_dir / f"{args.id}-silence.mp4", size)

    all_eye = [m[3] for _, m in speech + silence if m is not None]
    if not all_eye:
        raise SystemExit("no faces found in either clip")
    eye_open = statistics.median(all_eye)

    candidates = [
        (rgb, m) for rgb, m in speech if m is not None and m[3] >= 0.8 * eye_open
    ]
    if not candidates:
        raise SystemExit("no eyes-open speech frames")

    opens = [m[1] for _, m in candidates]
    widths = [m[2] for _, m in candidates]
    o_min, o_max = min(opens), max(opens)
    w_min, w_max = min(widths), max(widths)
    o_span = (o_max - o_min) or 1.0
    w_span = (w_max - w_min) or 1.0

    norm = [
        (float((m[1] - o_min) / o_span), float((m[2] - w_min) / w_span), rgb, m)
        for rgb, m in candidates
    ]
    chosen = _farthest_point(norm, args.max_mouth)
    # chosen[0] is the rest frame (smallest open -> normalised open 0.0)

    mouth_entries = []
    for order, idx in enumerate(chosen):
        on, ow, rgb, m = norm[idx]
        name = f"m{order:03d}.jpg"
        cv2.imwrite(str(out_dir / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        mouth_entries.append({"file": name, "open": round(on, 5), "width": round(ow, 5)})

    # eye box from the rest frame
    rest_rgb, _rest_m = candidates[chosen[0]]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rest_rgb)
    res = landmarker.detect(mp_img)
    lm = res.face_landmarks[0]
    pts = []
    for i in EYE_LANDMARKS + BROW_LANDMARKS:
        pts.append((lm[i].x * size, lm[i].y * size))
    e_rest = abs(lm[33].x - lm[263].x) * size
    pad = 0.18 * e_rest
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = max(0, int(round(min(xs) - pad)))
    y0 = max(0, int(round(min(ys) - pad)))
    x1 = min(size, int(round(max(xs) + pad)))
    y1 = min(size, int(round(max(ys) + pad)))
    eye_box = [x0, y0, x1 - x0, y1 - y0]

    def _crop(rgb: np.ndarray) -> np.ndarray:
        return rgb[eye_box[1]:eye_box[1] + eye_box[3], eye_box[0]:eye_box[0] + eye_box[2]]

    sil_measured = [(rgb, m) for rgb, m in silence if m is not None]
    blink_names = []
    if sil_measured:
        closed = min(sil_measured, key=lambda t: t[1][3])
        targets = [0.33 * eye_open, 0.66 * eye_open]
        picks = [closed]
        for t in targets:
            picks.append(min(sil_measured, key=lambda c: abs(c[1][3] - t)))
        # Each patch is RGBA: opaque over the two eyes, fading to nothing well inside the box, so pasting it
        # changes the eyelids only and never shows as a rectangle.
        mask = np.zeros((size, size), np.float32)
        for ids in ((33, 133, 159, 145), (263, 362, 386, 374)):
            ex = [lm[i].x * size for i in ids]
            ey = [lm[i].y * size for i in ids]
            centre = (int(round(sum(ex) / 4)), int(round(sum(ey) / 4)))
            span = max(ex) - min(ex)
            cv2.ellipse(mask, centre, (int(round(span * 0.80)), int(round(span * 0.48))), 0, 0, 360, 1.0, -1)
        blur = max(3, int(e_rest * 0.16) | 1)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        for order, (rgb, _m) in enumerate(picks):
            name = f"b{order}.png"
            alpha = (_crop(mask) * 255).astype(np.uint8)
            rgba = np.dstack([cv2.cvtColor(_crop(rgb), cv2.COLOR_RGB2BGR), alpha])
            cv2.imwrite(str(out_dir / name), rgba)
            blink_names.append(name)

    index = {
        "id": args.id,
        "size": size,
        "fps": 25,
        "eye_box": eye_box,
        "eye_open": round(float(eye_open), 6),
        "mouth": mouth_entries,
        "blink": blink_names,
        "open_range": [round(o_min, 6), round(o_max, 6)],
        "width_range": [round(w_min, 6), round(w_max, 6)],
    }
    (out_dir / "index.json").write_text(json.dumps(index))

    pack_bytes = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    frames_read = len(speech) + len(silence)
    print(
        f"{args.id}: read {frames_read} frames, kept {len(mouth_entries)} mouth frames, "
        f"pack {pack_bytes / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    main()
