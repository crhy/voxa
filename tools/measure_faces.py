"""Measure face proportions and colouring from portrait photos.

Pure helpers (proportions, colour sampling, filtering) are importable without
mediapipe; the mediapipe FaceLandmarker is only loaded in main().

Landmark indices (MediaPipe FaceMesh, 478 points; 468-472 left iris,
473-477 right iris):
  iris centres          468 (left), 473 (right)  -> inter-pupil distance
  left eye              outer 33, inner 133, top 145, bottom 154
  right eye             inner 363, outer 372, top 390, bottom 375
  left brow             70 (outer) .. 76 (inner)
  right brow            121 (inner) .. 126 (outer)
  nose                  top 1, bottom 2, tip 6, nostril edges 4, 5
  lips                  corners 61, 291; upper centre 185; lower centre 17
  chin                  152
  cheekbones            234 (left), 454 (right)
  jaw at mouth level    172 (left), 397 (right)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REF_DIR = Path("/Voxa/.voxa-spec/out/reference")

IPD_L, IPD_R = 468, 473
IRIS_L = list(range(468, 473))
IRIS_R = list(range(473, 478))
LB = list(range(70, 77))          # 70 outer .. 76 inner
RB = list(range(121, 127))        # 121 inner .. 126 outer
CHEEK_L, CHEEK_R = 234, 454
JAW_L, JAW_R = 172, 397
LIP_L, LIP_R = 61, 291
LIP_UP, LIP_DN = 185, 17


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _perp_signed(p, a, b):
    """Signed perpendicular distance from p to line a-b; positive when p is
    above the line (smaller y, since y grows downward)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    seg = math.hypot(dx, dy)
    if seg <= 1e-9:
        return 0.0
    cross = dx * (p[1] - a[1]) - dy * (p[0] - a[0])
    return (-cross / seg) if dx > 0 else (cross / seg)


EXPECTED_RANGES = {
    "eye_spacing": (0.45, 0.65),
    "eye_width": (0.38, 0.55),
    "eye_height": (0.12, 0.24),
    "brow_height": (0.22, 0.50),
    "nose_length": (0.65, 0.95),
    "nose_width": (0.50, 0.75),
    "philtrum_length": (0.18, 0.35),
    "mouth_width": (0.70, 1.00),
    "upper_lip_height": (0.07, 0.20),
    "lower_lip_height": (0.12, 0.28),
    "chin_height": (0.45, 0.70),
    "face_height": (1.75, 2.25),
    "face_width": (2.05, 2.45),
    "jaw_width": (1.55, 2.05),
}


def sanity_violations(props):
    """Names of proportions outside their expected adult-face range."""
    out = []
    for name, (lo, hi) in EXPECTED_RANGES.items():
        v = props.get(name)
        if v is not None and not (lo <= v <= hi):
            out.append(name)
    return out


def proportions(landmarks, image_width, image_height):
    """Scale-free proportion vector: each measurement / inter-pupil distance.

    landmarks: sequence of 478 (x, y) tuples in normalised (0..1) coords.
    y grows downward. Coordinates are converted to pixels (x*image_width,
    y*image_height) before Euclidean distances are taken.
    """
    w, h = image_width, image_height
    lm = [(p[0] * w, p[1] * h) for p in landmarks]
    ipd = _dist(lm[IPD_L], lm[IPD_R])
    if ipd <= 1e-6:
        raise ValueError("degenerate inter-pupil distance")

    eye_spacing = _dist(lm[133], lm[362])
    eye_width = (_dist(lm[33], lm[133]) + _dist(lm[362], lm[263])) / 2
    eye_height = (_dist(lm[159], lm[145]) + _dist(lm[386], lm[374])) / 2
    brow_height = (_dist(lm[105], lm[159]) + _dist(lm[334], lm[386])) / 2
    brow_arch = (_perp_signed(lm[105], lm[107], lm[70])
                 + _perp_signed(lm[334], lm[336], lm[300])) / 2
    nose_length = _dist(lm[168], lm[2])
    nose_width = _dist(lm[129], lm[358])
    philtrum_length = _dist(lm[2], lm[0])
    mouth_width = _dist(lm[61], lm[291])
    upper_lip_height = _dist(lm[0], lm[13])
    lower_lip_height = _dist(lm[14], lm[17])
    chin_height = _dist(lm[17], lm[152])
    face_height = _dist(lm[9], lm[152])
    face_width = _dist(lm[CHEEK_L], lm[CHEEK_R])
    jaw_width = _dist(lm[JAW_L], lm[JAW_R])
    taper = face_width / jaw_width if jaw_width > 1e-6 else 0.0
    fh_fw = face_height / face_width if face_width > 1e-6 else 0.0

    return {
        "face_height": face_height / ipd,
        "face_width": face_width / ipd,
        "jaw_width": jaw_width / ipd,
        "chin_height": chin_height / ipd,
        "nose_length": nose_length / ipd,
        "nose_width": nose_width / ipd,
        "mouth_width": mouth_width / ipd,
        "upper_lip_height": upper_lip_height / ipd,
        "lower_lip_height": lower_lip_height / ipd,
        "eye_width": eye_width / ipd,
        "eye_height": eye_height / ipd,
        "eye_spacing": eye_spacing / ipd,
        "brow_height": brow_height / ipd,
        "brow_arch": brow_arch / ipd,
        "philtrum_length": philtrum_length / ipd,
        "cheekbone_to_jaw_taper": taper,
        "face_height_over_width": fh_fw,
    }


def _median_rgb(pixels):
    arr = np.asarray(pixels, dtype=float)
    return [round(float(v), 1) for v in np.median(arr, axis=0)]


def _patch(image, cx, cy, half):
    h, w = image.shape[:2]
    x0, x1 = max(0, int(cx - half)), min(w, int(cx + half) + 1)
    y0, y1 = max(0, int(cy - half)), min(h, int(cy + half) + 1)
    if x1 <= x0 or y1 <= y0:
        return []
    return list(image[y0:y1, x0:x1].reshape(-1, image.shape[2])[:, :3])


def sample_colors(image, landmarks):
    """Median RGB for skin (cheek patches), lips, iris, brows, hair band."""
    lm = [tuple(p[:2]) for p in landmarks]
    h, w = image.shape[:2]

    def to_px(p):
        return p[0] * w, p[1] * h

    skin = []
    for cx, cy in (
        ((lm[CHEEK_L][0] + lm[LIP_L][0]) / 2, (lm[CHEEK_L][1] + lm[LIP_L][1]) / 2),
        ((lm[CHEEK_R][0] + lm[LIP_R][0]) / 2, (lm[CHEEK_R][1] + lm[LIP_R][1]) / 2),
    ):
        px, py = to_px((cx, cy))
        skin.extend(_patch(image, px, py, max(4, int(0.06 * w))))
    lips = []
    px, py = to_px(((lm[LIP_L][0] + lm[LIP_R][0] + lm[LIP_UP][0] + lm[LIP_DN][0]) / 4,
                    (lm[LIP_L][1] + lm[LIP_R][1] + lm[LIP_UP][1] + lm[LIP_DN][1]) / 4))
    lips.extend(_patch(image, px, py, max(3, int(0.03 * w))))
    iris = []
    for i in IRIS_L + IRIS_R:
        px, py = to_px(lm[i])
        iris.extend(_patch(image, px, py, 2))
    brows = []
    for i in LB + RB:
        px, py = to_px(lm[i])
        brows.extend(_patch(image, px, py, 2))
    hair = []
    brow_y = max(to_px(lm[i])[1] for i in LB + RB)
    band_top = max(0, int(brow_y - 0.18 * h))
    cx = w / 2
    if band_top < brow_y - 2:
        hair.extend(image[band_top:max(band_top + 1, int(brow_y - 2)),
                         max(0, int(cx - 0.15 * w)):int(cx + 0.15 * w)].reshape(-1, 3))
    out = {}
    for name, pix in (("skin", skin), ("lips", lips), ("iris", iris),
                      ("brow", brows), ("hair", hair)):
        out[name] = _median_rgb(pix) if pix else None
    return out


def _pose_ok(yaw_deg, pitch_deg, jaw_open, mouth_smile):
    return abs(yaw_deg) <= 15 and abs(pitch_deg) <= 15 and jaw_open < 0.25


def _euler_from_matrix(m):
    """Yaw/pitch in degrees from the 4x4 transformation matrix (row-major flat)."""
    m = list(np.asarray(m).ravel())
    pitch = math.asin(max(-1.0, min(1.0, -m[2])))
    yaw = math.atan2(m[6], m[10])
    return math.degrees(yaw), math.degrees(pitch)


def main():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker,
        FaceLandmarkerOptions,
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="/home/rhy/voxa-characters/analysis/models/face_landmarker.task",
        ),
        num_faces=2,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    marker = FaceLandmarker.create_from_options(options)

    index = json.loads((REF_DIR / "photos" / "index.json").read_text())
    results = {}
    for path in sorted(index):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = marker.detect(img)
        if len(res.face_landmarks) != 1:
            continue
        lm = [(p.x, p.y) for p in res.face_landmarks[0]]
        xs = [p[0] for p in lm]
        face_w_frac = max(xs) - min(xs)
        if face_w_frac * img.width < 160:
            continue
        blend = {b.category_name: b.score for b in res.face_blendshapes[0]} if res.face_blendshapes else {}
        jaw_open = blend.get("jawOpen", 0.0)
        smile = blend.get("mouthSmileLeft", 0.0) + blend.get("mouthSmileRight", 0.0)
        if jaw_open >= 0.25:
            continue
        mats = res.facial_transformation_matrixes[0] if res.facial_transformation_matrixes else None
        if mats is None:
            continue
        yaw, pitch = _euler_from_matrix(list(mats))
        if not _pose_ok(yaw, pitch, jaw_open, smile):
            continue
        image = rgb
        props = proportions(lm, img.width, img.height)
        bad = sanity_violations(props)
        if len(bad) >= 3:
            print(f"UNUSABLE {path}: {len(bad)} proportions out of range "
                  f"({', '.join(sorted(bad))})")
            continue
        results[path] = {
            "proportions": props,
            "colors": sample_colors(image, lm),
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "jaw_open": jaw_open,
            "smile": smile,
        }
    (REF_DIR / "measurements.json").write_text(json.dumps(results, indent=1))
    print(f"kept {len(results)} photos")


if __name__ == "__main__":
    main()
