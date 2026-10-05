"""Play a photoreal "face pack" without any GPU or MediaPipe.

A face pack is a small folder of JPEG mouth frames plus three eye-blink crops
produced once by :mod:`tools.build_face_pack` from head-locked talking clips.
At runtime Voxa only has to choose which mouth frame to show for a viseme
target and which blink patch to overlay, so this module is pure Python: no
cv2, no mediapipe, no GTK.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


def _faces_root() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local/share"))
    return base / "voxa" / "faces"


FACES_DIRECTORY: Path = _faces_root()


@dataclass(frozen=True, slots=True)
class MouthFrame:
    file: str
    open: float
    width: float


@dataclass(frozen=True, slots=True)
class FacePack:
    id: str
    directory: Path
    size: int
    eye_box: tuple[int, int, int, int]
    mouth: tuple[MouthFrame, ...]
    blink: tuple[str, ...]


VISEME_SHAPES: dict[str, tuple[float, float]] = {
    "viseme_sil": (0.0, 0.5),
    "viseme_PP": (0.0, 0.45),
    "viseme_FF": (0.12, 0.5),
    "viseme_TH": (0.2, 0.5),
    "viseme_DD": (0.25, 0.55),
    "viseme_kk": (0.3, 0.55),
    "viseme_CH": (0.25, 0.35),
    "viseme_SS": (0.12, 0.65),
    "viseme_nn": (0.2, 0.55),
    "viseme_RR": (0.25, 0.35),
    "viseme_aa": (1.0, 0.6),
    "viseme_E": (0.55, 0.75),
    "viseme_I": (0.35, 0.85),
    "viseme_O": (0.7, 0.25),
    "viseme_U": (0.35, 0.1),
}

REST_TARGET = (0.0, 0.5)


def load_pack(character_id: str, directory: Path | None = None) -> FacePack | None:
    """Load a face pack from disk, or ``None`` when it is missing or invalid."""
    root = directory if directory is not None else FACES_DIRECTORY
    pack_dir = Path(root) / character_id
    if directory is None and not (pack_dir / "index.json").exists():
        # Inside the Flatpak XDG_DATA_HOME is the app's private folder; the packs live in the host's
        # ~/.local/share/voxa/faces, which the sandbox can read.
        host_dir = Path.home() / ".local" / "share" / "voxa" / "faces" / character_id
        if (host_dir / "index.json").exists():
            pack_dir = host_dir
    index = pack_dir / "index.json"
    try:
        raw = index.read_text()
        data = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        pack_id = str(data["id"])
        size = int(data["size"])
        box = data["eye_box"]
        mouth_raw = data["mouth"]
        blink_raw = data["blink"]
        if len(box) != 4:
            return None
        eye_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        if not isinstance(mouth_raw, list) or not mouth_raw:
            return None
        mouth = tuple(
            MouthFrame(
                file=str(entry["file"]),
                open=float(entry["open"]),
                width=float(entry["width"]),
            )
            for entry in mouth_raw
        )
        if not isinstance(blink_raw, list):
            return None
        blink = tuple(str(name) for name in blink_raw)
    except (KeyError, TypeError, ValueError):
        return None
    return FacePack(
        id=pack_id,
        directory=pack_dir,
        size=size,
        eye_box=eye_box,
        mouth=mouth,
        blink=blink,
    )


def mouth_target(weights: dict[str, float]) -> tuple[float, float]:
    """Weighted average of the viseme shapes, or the rest shape when empty."""
    total = 0.0
    open_acc = 0.0
    width_acc = 0.0
    for name, weight in weights.items():
        shape = VISEME_SHAPES.get(name)
        if shape is None:
            continue
        total += weight
        open_acc += weight * shape[0]
        width_acc += weight * shape[1]
    if total <= 1e-9:
        return REST_TARGET
    return (open_acc / total, width_acc / total)


def _frame_distance(frame: MouthFrame, target: tuple[float, float]) -> float:
    open_d = frame.open - target[0]
    width_d = frame.width - target[1]
    return 2.0 * open_d * open_d + width_d * width_d


def pick_frame(
    pack: FacePack,
    target: tuple[float, float],
    previous: int = 0,
    stickiness: float = 0.03,
) -> int:
    """Index of the mouth frame nearest ``target``, with anti-flicker bias."""
    if not pack.mouth:
        return 0
    best = 0
    best_dist = math.inf
    for index, frame in enumerate(pack.mouth):
        dist = _frame_distance(frame, target)
        if dist < best_dist:
            best_dist = dist
            best = index
    if 0 <= previous < len(pack.mouth) and previous != best:
        prev_dist = _frame_distance(pack.mouth[previous], target)
        if best_dist + stickiness >= prev_dist:
            return previous
    return best


def blink_patch(pack: FacePack, amount: float) -> str | None:
    """Blink crop for ``amount`` (0 open .. 1 closed), or ``None`` when open."""
    if amount < 0.15 or not pack.blink:
        return None
    span = 1.0 - 0.15
    first = 0.15 + span / 3.0
    second = 0.15 + 2.0 * span / 3.0
    if amount < first:
        slot = 0
    elif amount < second:
        slot = 1
    else:
        slot = 2
    index = len(pack.blink) - 1 - slot
    if index < 0:
        index = 0
    return pack.blink[index]
