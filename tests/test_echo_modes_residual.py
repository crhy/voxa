"""f169: the bench arm ``voxa-residual`` - Voxa's NLMS fed the ExactAligner lock (the
f164 voxa-locked wiring) PLUS LockedResidualMask on that locked output, with the mask
reading its reference at the same locked delay. Arm contract: probe, no substitution,
and the clip path never crashes on the confidence fields."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from voxa.bench import modes
from voxa.bench.runner import ClipRunner, arg_parser, load_clip

sys.path.insert(0, str(Path(__file__).parent))
from test_bench_runner import (  # noqa: E402
    StubASR, _broadband, _echo_at, _room, events_of, write_clip,
)


def _dt_clip(root: Path) -> dict:
    """Known bulk delay (100 ms) + 300 ppm drift + a user onset 1.6 s in: the f168 test's
    clip, so the lock's ground truth is exact."""
    def delay(t):  # noqa: ANN001
        return 1600.0 + 300.0 / 1e6 * 16000 * t

    ref_src = _broadband(int(3.0 * 16000), 8)
    mic = np.convolve(_echo_at(ref_src, delay), _room(9))[: len(ref_src)]
    rng = np.random.default_rng(10)
    user = np.zeros(len(ref_src))
    user[int(1.6 * 16000):] = 0.05 * rng.standard_normal(len(ref_src) - int(1.6 * 16000))
    clip_dir = write_clip(root, "f169", "f169-0000-dt-00",
                          (mic + user).astype(np.float32), ref_src.astype(np.float32),
                          {"id": "f169-0000-dt-00", "scenario": "dt_middle", "expect": "keep",
                           "playback": {"text": "The weather is clear today.",
                                        "onset_s": 0.0, "offset_s": 3.0}})
    return load_clip(clip_dir)


def test_the_arm_is_offered_and_always_runs_here():
    probe = modes.availability()["voxa-residual"]
    assert probe["available"] is True
    assert "locked" in probe["reason"]
    assert arg_parser().parse_args(["--echo-mode", "voxa-residual"]).echo_mode == "voxa-residual"


def test_the_mask_reads_the_locked_delay_and_cuts_the_leak(tmp_path: Path) -> None:
    """The arm hands the residual mask the SAME lock the canceller is driven at (the
    reference read at the exact delay), and the locked output's residue comes down."""
    root = tmp_path / "data"
    loaded = _dt_clip(root)
    runner = ClipRunner(asr=StubASR(), echo_mode="voxa-residual", wake_word="voxa")
    summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])
    assert summary["echo_mode"] == "voxa-residual"
    assert runner.degraded is None

    # the lock is proven on this clip: the mask's exact lag matches the clip's known delay
    frames = [event for event in events_of(runner, "frame") if event.get("delay_confident")]
    assert frames, "the lock must prove itself on a known-delay clip"
    assert runner.mask._exact_lag is not None
    # the clip's delay drifts (1600 + 300 ppm); the mask reads the SAME last locked delay
    # the canceller is driven at, and that lands on the clip's known truth (<= 0.31 ms).
    assert runner.mask._exact_lag == frames[-1]["delay_samples"]
    assert abs(runner.mask._exact_lag - (1600.0 + 300.0 / 1e6 * 16000 * frames[-1]["t_s"])) <= 5

    # echo removed: the pre-user span of the cleaned stream holds clearly less energy than
    # the raw mic did (the stage mops up the locked canceller's residue)...
    stream = np.frombuffer(b"".join(runner._clean_stream), dtype="<i2").astype(np.float64)
    mic = loaded["mic"].astype(np.float64)
    n = min(len(stream), len(mic))
    a, b = int(0.3 * 16000), int(1.4 * 16000)  # Voxa plays, the user has not started
    assert np.mean(stream[a:b] ** 2) < 0.5 * np.mean(mic[a:b] ** 2)
    # ...while the user's span survives (never deleted for sharing the clip with echo).
    u = slice(int(2.0 * 16000), int(2.8 * 16000))
    assert np.mean(stream[u] ** 2) > 0.25 * np.mean(mic[u] ** 2)


def test_near_end_only_clip_never_crashes_on_the_confidence_fields(tmp_path: Path) -> None:
    """Silent reference the whole clip: nobody locks, delay_confident stays False, and the
    user is passed through untouched (the mask never deletes near-end evidence)."""
    root = tmp_path / "data"
    rng = np.random.default_rng(4)
    user = (0.05 * rng.standard_normal(2 * 16000)).astype(np.float32)
    clip_dir = write_clip(root, "f169", "f169-0001-nst-00", user, np.zeros_like(user),
                          {"id": "f169-0001-nst-00", "scenario": "nst", "expect": "keep",
                           "playback": None})
    loaded = load_clip(clip_dir)
    runner = ClipRunner(asr=StubASR(), echo_mode="voxa-residual", wake_word="voxa")
    summary = runner.run(loaded["mic"], loaded["ref"], loaded["truth"])
    assert not [e for e in events_of(runner, "frame") if e.get("delay_confident")]
    assert runner.mask._exact_lag is None  # never handed a delay: never claims one
    stream = np.frombuffer(b"".join(runner._clean_stream), dtype="<i2").astype(np.float64)
    mic = loaded["mic"].astype(np.float64)
    n = min(len(stream), len(mic))
    ratio = np.mean(stream[:n] ** 2) / np.mean(mic[:n] ** 2)
    assert 0.5 <= ratio <= 2.5  # passed through, not deleted
