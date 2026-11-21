"""Tests for the B5 baseline record and the no-regression comparison (voxa/bench/baseline.py).

Hand-built again: report envelopes written as dicts, no WAV, no ASR, no device, no sleeping. The
cases are the ones the ratchet depends on - the bad-side CI bound of a ``>=`` and of a ``<=`` metric,
a metric that is null in the baseline (nothing to hold a later run to), a value just inside and just
outside the baseline spread, a metric that went unmeasurable, and a run whose suite/mode has no
baseline entry at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from voxa.bench import baseline


def row(value, ci, direction, unit="fraction", samples=10, reason="") -> dict:
    """One metric row in the shape ``metrics.compute`` returns."""
    return {"value": value, "ci": list(ci), "samples": samples, "unit": unit,
            "direction": direction, "reason": reason}


def report(echo_mode: str = "voxa", suite: str = "smoke", metrics: dict | None = None,
           run: dict | None = None) -> dict:
    return {"suite": suite, "seed": 0, "echo_mode": echo_mode, "pass": True,
            "metrics": metrics if metrics is not None else {},
            "run": run if run is not None else {"clips": 48, "audio_seconds": 252.1,
                                                "wall_seconds": 37.8, "realtime_factor": 0.15}}


def test_the_bad_side_of_a_ci_is_the_bound_a_later_task_must_not_cross() -> None:
    assert baseline.bad_side(row(0.61, [0.51, 0.69], ">=")) == 0.51  # recall: the low bound
    assert baseline.bad_side(row(8.64, [5.97, 11.78], "<=")) == 11.78  # attenuation: the high bound
    assert baseline.bad_side(row(None, [None, None], ">=", reason="not measurable")) is None
    assert baseline.bad_side(row(1.0, [None, None], "<=")) is None  # no CI: no bound


def test_record_keeps_every_mode_and_suite_with_value_and_ci() -> None:
    reports = [
        report("off", "smoke", {"user_recall_nst": row(0.9455, [0.9, 0.98], ">=")}),
        report("off", "full", {"user_recall_nst": row(0.95, [0.93, 0.97], ">=")}),
        report("voxa", "smoke", {"user_recall_nst": row(0.6091, [0.5091, 0.6909], ">="),
                                 "nearend_atten_dt_db": row(8.64, [5.97, 11.78], "<="),
                                 "align_err_p90_ms": row(None, [None, None], "<=",
                                                         reason="roadmap task 1")}),
    ]
    saved = baseline.record(reports)
    assert sorted(saved["modes"]) == ["off", "voxa"]
    assert sorted(saved["modes"]["off"]) == ["full", "smoke"]
    vox = saved["modes"]["voxa"]["smoke"]
    assert vox["metrics"]["user_recall_nst"]["ci"] == [0.5091, 0.6909]
    assert vox["metrics"]["align_err_p90_ms"]["value"] is None
    assert vox["run"]["clips"] == 48 and vox["run"]["realtime_factor"] == 0.15
    # the ratchet is the default mode's smoke row, on the bad side of each CI
    assert saved["ratchet"] == {"suite": "smoke", "echo_mode": "voxa",
                                "rule": "smoke baseline CI bound on the bad side",
                                "thresholds": {"user_recall_nst": 0.5091,
                                               "nearend_atten_dt_db": 11.78}}


def test_a_value_inside_the_baseline_spread_is_not_a_regression() -> None:
    base = baseline.record([report("voxa", "smoke", {
        "user_recall_nst": row(0.6091, [0.5091, 0.6909], ">="),
        "nearend_atten_dt_db": row(8.64, [5.97, 11.78], "<="),
        "align_err_p90_ms": row(None, [None, None], "<=", reason="roadmap task 1")})])
    fresh = report("voxa", "smoke", {
        "user_recall_nst": row(0.51, [0.45, 0.57], ">="),   # at the bound: holds
        "nearend_atten_dt_db": row(11.78, [9.0, 13.0], "<="),  # at the bound: holds
        "align_err_p90_ms": row(None, [None, None], "<=", reason="roadmap task 1")})
    assert baseline.regressions(fresh, base) == []


def test_a_value_outside_the_spread_or_a_lost_metric_is_a_regression() -> None:
    base = baseline.record([report("voxa", "smoke", {
        "user_recall_nst": row(0.6091, [0.5091, 0.6909], ">="),
        "nearend_atten_dt_db": row(8.64, [5.97, 11.78], "<="),
        "turn_retention_rate": row(0.5, [0.3, 0.7], ">=")})])
    fresh = report("voxa", "smoke", {
        "user_recall_nst": row(0.50, [0.44, 0.56], ">="),   # 0.009 below the baseline CI low
        "nearend_atten_dt_db": row(12.0, [10.0, 14.0], "<="),  # 0.2 dB above the baseline CI high
        "turn_retention_rate": row(None, [None, None], ">=", reason="no accepted span")})
    lines = baseline.regressions(fresh, base)
    assert len(lines) == 3
    assert lines[0].startswith("user_recall_nst: REGRESSION")
    assert lines[1].startswith("nearend_atten_dt_db: REGRESSION")
    assert lines[2].startswith("turn_retention_rate: GAP")


def test_a_run_with_no_baseline_entry_is_reported_as_a_gap_not_a_pass() -> None:
    base = baseline.record([report("voxa", "smoke", {"user_recall_nst": row(0.61, [0.51, 0.69], ">=")})])
    lines = baseline.regressions(report("webrtc", "full"), base)
    assert lines == ["no baseline entry for suite=full echo_mode=webrtc"]


def test_gates_toml_holds_the_recorded_ratchet_and_is_never_looser() -> None:
    """Every baseline ratchet bound still has a line in gates.toml, and no threshold is looser than the
    baseline's own CI bound (tightening one later is fine; loosening one is the forbidden move)."""
    from voxa.bench import gates

    recorded = json.loads(baseline.BASELINE_PATH.read_text(encoding="utf-8"))
    limits = gates.thresholds()
    for name, _bound in recorded["ratchet"]["thresholds"].items():
        assert name in limits, f"{name}: the baseline recorded a ratchet and gates.toml dropped the line"
    for name, (operator, limit) in limits.items():
        row = recorded["modes"]["voxa"]["smoke"]["metrics"].get(name)
        if row is None:
            continue  # a metric the baseline could not sample: the task that adds it owns its number
        bound = baseline.bad_side(row)
        if bound is None:
            continue
        holds = limit >= bound if operator == ">=" else limit <= bound
        assert holds, f"{name}: {operator} {limit} is LOOSER than the baseline CI bound {bound}"


def test_record_and_check_on_files(tmp_path: Path) -> None:
    out = tmp_path / "baseline.json"
    first = tmp_path / "smoke-voxa.json"
    first.write_text(json.dumps(report("voxa", "smoke", {
        "user_recall_nst": row(0.6091, [0.5091, 0.6909], ">=")})))
    assert baseline.main(["record", str(first), "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["modes"]["voxa"]["smoke"]["metrics"]["user_recall_nst"]["value"] == 0.6091
    assert saved["ratchet"]["thresholds"] == {"user_recall_nst": 0.5091}

    good = tmp_path / "fresh.json"
    good.write_text(json.dumps(report("voxa", "smoke", {
        "user_recall_nst": row(0.70, [0.62, 0.78], ">=")})))
    assert baseline.main(["check", str(good), "--out", str(out)]) == 0

    bad = tmp_path / "broken.json"
    bad.write_text(json.dumps(report("voxa", "smoke", {
        "user_recall_nst": row(0.20, [0.1, 0.3], ">=")})))
    assert baseline.main(["check", str(bad), "--out", str(out)]) == 1
