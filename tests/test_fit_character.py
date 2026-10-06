from __future__ import annotations

import math

import pytest

# The fitting tool needs OpenCV and MediaPipe, which the build machines do not have.
fit_character = pytest.importorskip("tools.fit_character")


def test_error_zscores():
    group = {"proportions": {
        "a": {"mean": 1.0, "sd": 0.03},
        "b": {"mean": 2.0, "sd": 0.5},
        "c": {"mean": 0.0, "sd": 0.001},
    }}
    props = {"a": 1.03, "b": 1.0, "c": 0.03}
    # a: z=1 -> 1; b: z=-2 -> 4; c: sd clamped up to 0.03 -> z=1 -> 1
    assert math.isclose(fit_character.error(props, group), 6.0)


def test_paired_axes_never_both_set():
    axes = {"chin-width": 0.3, "chin-height": -0.4, "head-oval": 0.5}
    out = fit_character.targets_from_axes(axes, {})
    assert out["chin-width-incr"] == 0.3
    assert out["chin-height-decr"] == 0.4
    assert "chin-width-decr" not in out
    assert "chin-height-incr" not in out
    assert out["head-oval"] == 0.5


def test_descend_converges_single_axis():
    group = {"proportions": {"x": {"mean": 0.8, "sd": 0.1}}}

    def measure_fn(targets):
        v = targets.get("chin-width-incr", 0.0) - targets.get("chin-width-decr", 0.0)
        return {"x": 0.2 + v}

    axes, fixed = {"chin-width": 0.2}, {}
    got, err, builds = fit_character.descend(
        axes, fixed, group, measure_fn, [0.2, 0.2, 0.1], 10**9)
    assert abs(got["chin-width"] - 0.6) < 1e-9
    assert err < 1e-9
    targets = fit_character.targets_from_axes(got, fixed)
    assert "chin-width-incr" in targets and "chin-width-decr" not in targets


def test_descend_converges_two_axes_signed():
    group = {"proportions": {
        "x": {"mean": 0.6, "sd": 0.1},
        "y": {"mean": -0.2, "sd": 0.1},
    }}

    def measure_fn(targets):
        w = targets.get("chin-width-incr", 0.0) - targets.get("chin-width-decr", 0.0)
        h = targets.get("chin-height-incr", 0.0) - targets.get("chin-height-decr", 0.0)
        return {"x": 0.2 + w, "y": 0.2 + h}

    axes = {"chin-width": 0.0, "chin-height": 0.0}
    got, err, builds = fit_character.descend(
        axes, {}, group, measure_fn, [0.2, 0.2, 0.1], 10**9)
    assert abs(got["chin-width"] - 0.4) < 1e-9
    assert abs(got["chin-height"] + 0.4) < 1e-9
    assert err < 1e-9
    targets = fit_character.targets_from_axes(got, {})
    assert "chin-width-incr" in targets and "chin-width-decr" not in targets
    assert "chin-height-decr" in targets and "chin-height-incr" not in targets
