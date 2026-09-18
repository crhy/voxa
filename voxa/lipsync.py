"""Mouth-shape timelines for the avatar via Rhubarb Lip Sync (optional).

Rhubarb (MIT, ``/app/bin/rhubarb`` in the Flatpak -- see the
``rhubarb-lip-sync`` module in ``io.github.crhy.voxa.yml``) turns a TTS wav
plus its dialog text into timestamped Preston-Blair mouth shapes. By the
time ``speech.py`` plays audio we already hold both inputs, so the timeline
can be baked ahead of playback and scheduled against the audio clock.

A missing binary is normal (dev machines, minimal installs): :func:`analyze`
returns ``None`` and callers fall back to RMS jaw flap. See ``docs/AVATAR.md``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

#: Shapes Rhubarb can emit with default ``GHX`` extended shapes. ``X`` is rest.
VALID_SHAPES = frozenset("ABCDEFGHX")


class LipSyncError(Exception):
    """Rhubarb ran but its output was unusable, or it failed outright."""


@dataclass(frozen=True)
class MouthCue:
    start: float
    end: float
    shape: str


def find_binary() -> str | None:
    """Path to the ``rhubarb`` binary, or ``None`` when not installed."""
    return shutil.which("rhubarb")


def parse_cues(document: dict) -> list[MouthCue]:
    """Parse Rhubarb ``-f json`` output into validated :class:`MouthCue`s."""
    try:
        raw_cues = document["mouthCues"]
    except (KeyError, TypeError) as exc:
        raise LipSyncError(f"Rhubarb output has no mouthCues: {exc}") from None
    if not isinstance(raw_cues, list) or not raw_cues:
        raise LipSyncError("Rhubarb output has an empty mouthCues list.")
    cues = []
    for entry in raw_cues:
        try:
            start = float(entry["start"])
            end = float(entry["end"])
            shape = entry["value"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LipSyncError(f"Malformed mouth cue {entry!r}: {exc}") from None
        if shape not in VALID_SHAPES or not start <= end:
            raise LipSyncError(f"Malformed mouth cue {entry!r}.")
        cues.append(MouthCue(start=start, end=end, shape=shape))
    return cues


def shape_at(cues: list[MouthCue], moment: float) -> str:
    """Shape active at ``moment`` seconds (``"X"`` when silent/empty)."""
    if not cues:
        return "X"
    index = bisect_right([cue.start for cue in cues], moment) - 1
    cue = cues[max(index, 0)]
    return cue.shape if cue.start <= moment <= cue.end else "X"


def analyze(
    wav_path: str | Path,
    dialog_text: str,
    *,
    recognizer: str = "pocketSphinx",
    timeout: float = 120.0,
) -> list[MouthCue] | None:
    """Bake a mouth-shape timeline for ``wav_path``.

    Returns ``None`` when Rhubarb is not installed; raises
    :class:`LipSyncError` when it fails or its output is malformed.
    """
    binary = find_binary()
    if binary is None:
        return None
    with tempfile.TemporaryDirectory(prefix="voxa-lipsync-") as tmp:
        dialog_file = Path(tmp) / "dialog.txt"
        dialog_file.write_text(dialog_text, encoding="utf-8")
        try:
            result = subprocess.run(
                [binary, "-f", "json", "--quiet", "-r", recognizer, "-d", str(dialog_file), str(wav_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LipSyncError(f"Rhubarb failed to run: {exc}") from None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise LipSyncError(f"Rhubarb failed (exit {result.returncode}): {detail}")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LipSyncError(f"Rhubarb output is not JSON: {exc}") from None
    return parse_cues(document)
