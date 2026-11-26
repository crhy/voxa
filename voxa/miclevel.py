"""Voxa's own microphone input level (issue #72, task v11e step 3b).

rhY: "voxa should be able to adjust her own volume(s) so she can definitely hear."
The webcam microphone sits on top of a very large monitor with the speakers behind it,
so a person's voice (and VoxaTest's prompts, from farther away) reaches it too quietly
to keep her name. This calibrates the INPUT gain of the ONE source Voxa records from -
that microphone, never the system output and never another app's stream - so her speech
peaks near :data:`TARGET_PEAK_DBFS` without clipping.

The level is remembered per ``microphone_id`` (``Settings.mic_levels``); a change the
user made by hand wins and stops the automatic tracking for that microphone. It is
never touched while she is speaking. Everything is offline-testable: the pactl
conversation is the injected ``runner``, exactly like :mod:`voxa.echo`.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable

from .echo import _pactl, source_for

# Speech should peak around here: loud enough to beat the room, never near 0 dBFS.
TARGET_PEAK_DBFS = -20.0
# A peak this hot is clipping: the gain must come down even if that is a big step.
CLIP_DBFS = -1.0
# PulseAudio source-volume range Voxa will ever set (percent, over-amplification allowed).
MIN_SOURCE_PERCENT = 1
MAX_SOURCE_PERCENT = 150
# A change this small in percent is not worth writing to the sound server.
STEP_TOLERANCE_PERCENT = 3


def parse_source_volume(text: str) -> int | None:
    """The highest percent from ``pactl get-source-volume`` output (``… / 85% / …``)."""
    percents = [int(p) for p in re.findall(r"(\d+)%", text)]
    return max(percents) if percents else None


def target_percent(current: int, peak_dbfs: float,
                   target: float = TARGET_PEAK_DBFS) -> int:
    """The source volume that would move a measured ``peak_dbfs`` to ``target``.

    PulseAudio source volume is a linear amplitude scale, so a dB move is a ratio:
    ``new = current * 10**((target - peak) / 20)``. Clipped speech (a peak at or above
    :data:`CLIP_DBFS`) always comes down. The result is kept in the sound server's range.
    """
    if current <= 0:
        current = MIN_SOURCE_PERCENT
    delta = target - peak_dbfs
    # Never push gain up past the target by more than the clip guard wants; and pull it
    # down whenever the capture clips.
    if peak_dbfs >= CLIP_DBFS:
        delta = min(delta, target - peak_dbfs)
    scale = 10.0 ** (delta / 20.0)
    scaled = current * scale
    return int(max(MIN_SOURCE_PERCENT, min(MAX_SOURCE_PERCENT, round(scaled))))


def stored_level(microphone_id: str, mic_levels: dict) -> dict:
    entry = mic_levels.get(microphone_id)
    return entry if isinstance(entry, dict) else {}


def user_owns_level(microphone_id: str, mic_levels: dict) -> bool:
    return bool(stored_level(microphone_id, mic_levels).get("user_picked"))


def note_user_change(microphone_id: str, mic_levels: dict, percent: int) -> None:
    """The user set this microphone by hand: remember it and stop auto-adjusting it."""
    mic_levels[microphone_id] = {
        "volume_percent": int(percent),
        "user_picked": True,
    }


def read_source_volume(source: str, runner: Callable = subprocess.run) -> int | None:
    code, out = _pactl("get-source-volume", source, runner=runner)
    return parse_source_volume(out) if code == 0 else None


def apply_level(microphone_id: str, mic_levels: dict, percent: int,
                *, user_picked: bool = False) -> dict:
    mic_levels[microphone_id] = {
        "volume_percent": int(percent),
        "user_picked": bool(user_picked),
    }
    return mic_levels[microphone_id]


def calibrate(microphone_id: str, mic_levels: dict, peak_dbfs: float,
              *, speaking: bool = False,
              source: str | None = None,
              runner: Callable = subprocess.run) -> dict:
    """One calibration step for the microphone Voxa records.

    ``peak_dbfs`` is the peak of a short, real speech sample measured at this source at
    its CURRENT gain. Returns a dict with ``changed`` and a ``reason``. Refuses while she
    is speaking, when the user has taken over the level, or when the source cannot be
    found; and it only ever writes that one source's volume.
    """
    result: dict = {"microphone_id": microphone_id, "changed": False}
    if speaking:
        result["reason"] = "she is speaking"
        return result
    if user_owns_level(microphone_id, mic_levels):
        result["reason"] = "the user set this microphone's level"
        return result
    if not microphone_id:
        result["reason"] = "no microphone selected"
        return result
    source = source or source_for(microphone_id, runner=runner)
    if not source:
        result["reason"] = "microphone source not found"
        return result
    result["source"] = source
    current = read_source_volume(source, runner=runner)
    if current is None:
        result["reason"] = "source volume unreadable"
        return result
    want = target_percent(current, peak_dbfs)
    result["current"] = current
    result["target"] = want
    if abs(want - current) < STEP_TOLERANCE_PERCENT:
        result["reason"] = "already at target"
        apply_level(microphone_id, mic_levels, current)
        return result
    code, _ = _pactl("set-source-volume", source, f"{want}%", runner=runner)
    if code != 0:
        result["reason"] = "sound server refused the change"
        return result
    apply_level(microphone_id, mic_levels, want)
    result["changed"] = True
    result["reason"] = "set to reach the target"
    return result
