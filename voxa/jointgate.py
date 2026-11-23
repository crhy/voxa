"""Roadmap task 5 (f173): the joint double-talk gate's per-frame feeding, shared by the bench arm
and the LIVE hearing path (issue #72).

``voxa/bench/runner.py`` feeds the bench's ``DoubleTalkGate`` inside its offline pipeline; the
shipping path feeds the SAME gate on the hearing worker (``voxa/window.py::_joint_verdicts``), from
the bounded ring the capture thread stages (``voxa/audio.py::_note_joint_frame``). This module is
the one place that turns a drained list of staged frames into the joint verdicts, so the live path
and the bench cannot drift apart, and it is importable without GTK (the window itself is not).

The three sample streams a frame carries are the cleaned bytes (the model reads those), the raw
microphone and the reference (the gate's coherence reads those at the locked delay), plus the
delay, the residual mask's band margin over its OWN prediction, the running ERLE, and the
alignment/reference flags - exactly the fields ``voxa/audio.py`` measured and stored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# A staged frame: (cleaned_bytes, mic_bytes, ref_array, delay_samples|None, band_margin_db|None,
# erle_db, alignment_healthy, reference_active) - the shape voxa/audio.py stages.
Frame = tuple


def feed_joint_frames(gate: Any, vad: Any, frames: list[Frame]) -> tuple[bool | None, bool | None, float | None]:
    """Run the cleaned-stream Silero and the DoubleTalkGate over one drain; return the verdicts.

    ``(joint_suspect, joint_echo, band_margin)`` follows the bench's evidence rule: the verdict is
    reported when this frame MEASURED or a hold is still latched, and ``None`` when the gate could
    not measure (INDETERMINATE - the caller falls back to the level gate visibly, and "the gate did
    not measure" is never evidence of echo).
    """
    if not frames:
        return None, None, None
    suspect: bool | None = None
    echo: bool | None = None
    margin: float | None = None
    measurable = False
    for cleaned, mic_bytes, ref, delay, band_margin, erle, align_healthy, ref_active in frames:
        vad.feed(cleaned)
        prob = float(vad.last_prob) if vad.probs else None
        mic = np.frombuffer(mic_bytes, dtype="<i2").astype(np.float64) / 32768.0
        dt_ms = (len(mic_bytes) // 2) / 16000.0 * 1000.0
        suspect = bool(gate.observe(
            dt_ms=dt_ms, prob=prob, mic=mic, ref=ref,
            delay_samples=delay, residual_over_prediction_db=band_margin, erle_db=erle,
            reference_active=ref_active, alignment_healthy=align_healthy))
        measurable = bool(gate.conditions.get("measurable"))
        echo = bool(gate.echo_consistent)
        margin = band_margin
    joint_suspect = suspect if (measurable or suspect) else None
    joint_echo = echo if (measurable or echo) else None
    return joint_suspect, joint_echo, margin
