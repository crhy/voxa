"""VoxaTest's closed-loop prompt level (issue #72, task v11e step 3c).

VoxaTest speaks the test user through its OWN playback stream - a sink input tagged
``voxatest`` (see :mod:`voxatest.isolation`). For a live run to measure a *real*
hearing failure, every prompt has to reach Voxa's microphone at a realistic talking
level - the level a person speaking at normal volume about 1 m from the desk actually
produces at this microphone - and no higher. ``tools/voxatest_volume_check.py`` only
does a pre-check that *raises the system sink until the phrase is loud*, which is the
opposite of what a fair test needs: turning the volume up "until she hears it" hides
genuine hearing failures.

This module is the on-the-fly loop the plan asks for:

* it calibrates the tagged stream's gain once so a prompt's *measured arrival at Voxa's
  microphone* equals the recorded reference target (:data:`TARGET_ARRIVAL_DBFS`);
* after every prompt it re-measures the arrival and nudges that same stream back to
  target, logging each prompt's arrival level for the report;
* it never turns the level up because a prompt was missed - an on-target prompt she
  missed is a real failure, so the gain is held at target and the miss is reported;
* it only ever writes VoxaTest's own stream (:func:`_set_stream_percent` uses
  ``set-sink-input-volume`` on the tagged input) - never the system sink, never another
  app's stream.

The pactl conversation and the microphone meter are both injected, so the whole loop is
offline-testable with a fake sink input and a fake arrival meter (no sound).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

from voxa.echo import _pactl
from voxa.reference import parse_sink_inputs, test_inputs

# A person speaking at normal volume about 1 m from the desk: the arrival level the
# loop holds every prompt at, measured at Voxa's microphone (recorded reference).
TARGET_ARRIVAL_DBFS = -24.0
# A prompt this far from the target counts as on target; anything further is nudged.
TARGET_TOLERANCE_DB = 2.0
# The stream gain PulseAudio is ever asked for (percent): bounded so a runaway meter
# can never crank the speakers to the ceiling.
MIN_STREAM_PERCENT = 5
MAX_STREAM_PERCENT = 120
# A nudge this small is not worth writing to the sound server.
STEP_TOLERANCE_PERCENT = 3


def parse_input_volume(text: str) -> int | None:
    """The highest percent from ``pactl get-sink-input-volume`` output (``… / 85% / …``)."""
    percents = [int(p) for p in (p[:-1] for p in text.replace("\n", " ").split() if p.endswith("%") and p[:-1].isdigit())]
    return max(percents) if percents else None


def target_percent(current: int, arrival_dbfs: float,
                   target: float = TARGET_ARRIVAL_DBFS) -> int:
    """The stream volume that would move a measured ``arrival_dbfs`` to ``target``.

    PulseAudio stream volume is a linear amplitude scale, so a dB move is a ratio:
    ``new = current * 10**((target - arrival) / 20)``. The result stays in the range
    the sound server will accept.
    """
    if current <= 0:
        current = MIN_STREAM_PERCENT
    scale = 10.0 ** ((target - arrival_dbfs) / 20.0)
    scaled = current * scale
    return int(max(MIN_STREAM_PERCENT, min(MAX_STREAM_PERCENT, round(scaled))))


def on_target(arrival_dbfs: float, target: float = TARGET_ARRIVAL_DBFS,
              tolerance: float = TARGET_TOLERANCE_DB) -> bool:
    """True when a prompt reached Voxa's mic close enough to the reference target."""
    return abs(arrival_dbfs - target) <= tolerance


def find_test_input(runner: Callable = subprocess.run) -> int | None:
    """The sink-input index of VoxaTest's OWN tagged playback stream, or None."""
    code, listing = _pactl("list", "sink-inputs", runner=runner)
    if code != 0:
        return None
    inputs = test_inputs(parse_sink_inputs(listing))
    return int(inputs[0]["index"]) if inputs else None


def read_stream_percent(index: int, runner: Callable = subprocess.run) -> int | None:
    code, out = _pactl("get-sink-input-volume", str(index), runner=runner)
    return parse_input_volume(out) if code == 0 else None


def _set_stream_percent(index: int, percent: int, runner: Callable = subprocess.run) -> bool:
    code, _ = _pactl("set-sink-input-volume", str(index), f"{percent}%", runner=runner)
    return code == 0


@dataclass
class PromptArrival:
    """One prompt's logged arrival at Voxa's mic, for the report."""

    prompt: str
    arrival_dbfs: float
    on_target: bool
    heard: bool | None
    real_failure: bool
    stream_percent: int

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "arrival_dbfs": round(self.arrival_dbfs, 1),
            "on_target": self.on_target,
            "heard": self.heard,
            "real_failure": self.real_failure,
            "stream_percent": self.stream_percent,
        }


@dataclass
class PromptLevelLoop:
    """Holds VoxaTest's own prompt level at the recorded talking target.

    ``measure(prompt)`` is the injected microphone meter: the dBFS level the prompt
    reached at Voxa's microphone (independent of whether she recognised it). ``heard``
    is whether Voxa reported the prompt - it decides a *real failure* but never moves
    the volume.
    """

    measure: Callable[[str], float]
    runner: Callable = subprocess.run
    target: float = TARGET_ARRIVAL_DBFS
    tolerance: float = TARGET_TOLERANCE_DB
    arrivals: list[PromptArrival] = field(default_factory=list)
    index: int | None = None
    reason: str = ""

    def calibrate(self, calibration_prompt: str) -> bool:
        """Set the tagged stream once so ``calibration_prompt`` arrives at the target."""
        self.index = find_test_input(self.runner)
        if self.index is None:
            self.reason = "VoxaTest's own playback stream is not playing"
            return False
        current = read_stream_percent(self.index, self.runner)
        if current is None:
            self.reason = "stream volume unreadable"
            return False
        arrival = self.measure(calibration_prompt)
        want = target_percent(current, arrival, self.target)
        if abs(want - current) >= STEP_TOLERANCE_PERCENT and not _set_stream_percent(self.index, want, self.runner):
            self.reason = "sound server refused the change"
            return False
        self.reason = "stream set to the recorded talking level"
        return True

    def after_prompt(self, prompt: str, heard: bool) -> PromptArrival:
        """Measure one prompt's arrival, nudge the stream back to target, log it.

        Never turns the level up because ``heard`` is False: a prompt that reached the
        target but was missed is reported as a real failure, gain left untouched.
        """
        if self.index is None:
            self.reason = "loop not calibrated"
        arrival = self.measure(prompt)
        at_target = on_target(arrival, self.target, self.tolerance)
        percent = read_stream_percent(self.index, self.runner) if self.index is not None else 0
        real_failure = (not heard) and at_target
        # Nudge the stream toward target from the MEASURED arrival only - never to
        # chase a miss. On target (heard or not) the gain is left alone.
        if not at_target and self.index is not None:
            current = percent or MIN_STREAM_PERCENT
            want = target_percent(current, arrival, self.target)
            if abs(want - current) >= STEP_TOLERANCE_PERCENT:
                _set_stream_percent(self.index, want, self.runner)
        record = PromptArrival(
            prompt=prompt,
            arrival_dbfs=arrival,
            on_target=at_target,
            heard=heard,
            real_failure=real_failure,
            stream_percent=read_stream_percent(self.index, self.runner) if self.index is not None else 0,
        )
        self.arrivals.append(record)
        return record

    def report(self) -> list[dict]:
        """Each prompt's arrival level, in order, for the live report."""
        return [record.as_dict() for record in self.arrivals]
