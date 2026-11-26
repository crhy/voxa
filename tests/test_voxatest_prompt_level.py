"""Offline tests for the VoxaTest closed-loop prompt level (issue #72, task v11e step 3c).

No sound: the sink input is a fake Pactl stream and the microphone meter is a fake that
reports the arrival level for whatever gain the loop last asked for."""

from __future__ import annotations

import math

import pytest

from voxatest import promptlevel
from voxatest.promptlevel import (
    MAX_STREAM_PERCENT,
    MIN_STREAM_PERCENT,
    TARGET_ARRIVAL_DBFS,
    PromptLevelLoop,
    on_target,
    parse_input_volume,
    target_percent,
)

# The tagged playback stream the fake sound server always reports.
_LISTING = """\
Sink Input #31
	State: RUNNING
	Name: sink-input.frozen
	Application: 123453 (pactl on pid 99)
	Media role: voxatest
	Media name: "VoxaTest test voice"
	Module: 4
	Property list:
		application.name = "voxatest"
		media.name = "VoxaTest test voice"
		media.role = "voxatest"
Sink Input #44
	State: RUNNING
	Application: 7 (some other app)
	Property list:
		application.name = "rhythmbox"
"""


class FakePactl:
    """A sound server with one tagged stream whose arrival at the mic we can set freely.

    ``arrival_at_100`` is the dBFS a prompt reaches at Voxa's microphone when the stream
    sits at 100%; the arrival then moves with the stream's linear gain, exactly like the
    real thing.
    """

    def __init__(self, arrival_at_100: float = -20.0, start_percent: int = 100) -> None:
        self.arrival_at_100 = arrival_at_100
        self.percent = start_percent
        self.sets: list[str] = []
        self.reads = 0

    def __call__(self, argv, *args, **kwargs):
        cmd = " ".join(argv)
        if "list sink-inputs" in cmd:
            return _done(0, _LISTING)
        if "get-sink-input-volume" in cmd:
            self.reads += 1
            return _done(0, f"Sink Input #31.\n\tCurrent Volume: {self.percent}%")
        if "set-sink-input-volume" in cmd:
            self.sets.append(cmd)
            self.percent = int(cmd.split()[-1].rstrip("%"))
            return _done(0, "")
        if "set-sink-volume" in cmd or "set-source-volume" in cmd:
            raise AssertionError(f"loop must never touch the system output or another app: {cmd}")
        return _done(1, "")

    def arrival(self) -> float:
        return self.arrival_at_100 + 20 * math.log10(self.percent / 100.0)


def _done(code: int, out: str):
    import subprocess

    return subprocess.CompletedProcess(["pactl"], code, out, "")


def _loop(pactl: FakePactl, **kw) -> PromptLevelLoop:
    return PromptLevelLoop(measure=lambda prompt: pactl.arrival(), runner=pactl, **kw)


def test_parse_input_volume():
    assert parse_input_volume("Current Volume: 45%") == 45
    assert parse_input_volume("Front Left: 0% Base 60%\n") == 60
    assert parse_input_volume("no data") is None


def test_target_percent_moves_toward_target():
    # -30 dBFS at 100% needs about +6 dB -> ~115% to land on -24 dBFS.
    want = target_percent(100, -30.0)
    assert 110 <= want <= 120
    # a prompt already at target stays put; one too hot comes down
    assert target_percent(100, TARGET_ARRIVAL_DBFS) == 100
    assert target_percent(100, -10.0) < 100


def test_target_percent_is_bounded():
    assert target_percent(100, -90.0) <= MAX_STREAM_PERCENT
    assert target_percent(100, 0.0) >= MIN_STREAM_PERCENT


def test_on_target_window():
    assert on_target(TARGET_ARRIVAL_DBFS)
    assert on_target(TARGET_ARRIVAL_DBFS + promptlevel.TARGET_TOLERANCE_DB)
    assert not on_target(TARGET_ARRIVAL_DBFS + 5)


def test_calibrate_sets_stream_to_target():
    pactl = FakePactl(arrival_at_100=-20.0, start_percent=100)
    loop = _loop(pactl)
    assert loop.calibrate("calibration prompt")
    # only the tagged stream was written, never the system sink
    assert all("set-sink-input-volume" in c for c in pactl.sets)
    assert abs(pactl.percent - target_percent(100, -20.0)) < 3
    assert on_target(pactl.arrival())


def test_find_test_input_is_the_tagged_stream():
    pactl = FakePactl()
    assert promptlevel.find_test_input(pactl) == 31


def test_after_prompt_nudges_off_target_back():
    pactl = FakePactl(arrival_at_100=-20.0, start_percent=40)
    loop = _loop(pactl)
    loop.index = 31  # pretend calibrated, stream left one big step short
    before = pactl.percent
    rec = loop.after_prompt("open the map", heard=True)
    assert not rec.on_target
    assert pactl.percent > before  # pulled back up toward the talking target


def test_ongtarget_prompt_heard_leaves_gain_alone():
    pactl = FakePactl(arrival_at_100=-20.0)
    loop = _loop(pactl)
    loop.calibrate("calibration prompt")
    after_cal = pactl.percent
    rec = loop.after_prompt("open the map", heard=True)
    assert rec.on_target and rec.heard and not rec.real_failure
    assert pactl.percent == after_cal


def test_miss_at_on_target_is_a_real_failure_not_a_louder_cue():
    # THE rule of step 3c: a prompt that reached her mic on target but that she missed
    # is a real hearing failure - the volume must NOT be turned up to make her hear it.
    pactl = FakePactl(arrival_at_100=-20.0)
    loop = _loop(pactl)
    loop.calibrate("calibration prompt")
    gain_after_cal = pactl.percent
    sets_after_cal = len(pactl.sets)
    rec = loop.after_prompt("directions to the library", heard=False)
    assert rec.on_target
    assert rec.real_failure
    assert pactl.percent == gain_after_cal  # never louder until she hears it
    assert len(pactl.sets) == sets_after_cal


def test_repeated_misses_never_crank_the_stream():
    pactl = FakePactl(arrival_at_100=-20.0)
    loop = _loop(pactl)
    loop.calibrate("calibration prompt")
    after_cal = pactl.percent
    for _ in range(5):
        loop.after_prompt("a missed prompt", heard=False)
    assert pactl.percent == after_cal  # held at target, never crept up


def test_report_logs_arrival_level_per_prompt():
    pactl = FakePactl(arrival_at_100=-20.0)
    loop = _loop(pactl)
    loop.calibrate("calibration prompt")
    loop.after_prompt("first prompt", heard=True)
    loop.after_prompt("second prompt", heard=False)
    report = loop.report()
    assert [r["prompt"] for r in report] == ["first prompt", "second prompt"]
    assert all("arrival_dbfs" in r and r["on_target"] for r in report)
    assert report[1]["real_failure"] is True


def test_calibrate_refuses_without_our_stream():
    pactl = FakePactl()
    loop = _loop(pactl)

    def listing_only_other(pactl_obj=pactl):
        return _done(0, "Sink Input #44\n\tProperty list:\n\t\tapplication.name = \"rhythmbox\"\n")

    loop.runner = lambda argv, *a, **k: listing_only_other() if "list sink-inputs" in " ".join(argv) else pactl(argv)
    assert not loop.calibrate("prompt")
    assert "playback stream" in loop.reason


if __name__ == "__main__":
    pytest.main([__file__])
