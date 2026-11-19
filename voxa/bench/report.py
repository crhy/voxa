"""B3: the gate report - per-scenario breakdown, 1000-sample bootstrap CIs, report.json, ``check``.

``python -m voxa.bench run`` calls :func:`build_report` and writes it to ``.voxa-bench/report.json``
in the shape the shared brief fixes::

    {"suite": ..., "seed": ..., "echo_mode": ..., "pass": bool,
     "metrics": {"<name>": {"value": x, "ci": [lo, hi], "threshold": t, "pass": b}},
     "by_scenario": {"<scenario>": {"clips": n, "metrics": {...}}},
     "run": {...}, "clips": [...]}

Every metric name the brief lists is present, always with a value or an honest ``null``: a metric this
stage cannot measure carries ``value: null`` and ``pass: true``, never a number it cannot defend.
The CI is the 2.5 % / 97.5 % pair of 1000 bootstrap resamples of the metric's own samples
(``voxa/bench/metrics.py``), so a regression that only the spread of a few clips explains shows up
as a wide CI instead of a false pass.

``python -m voxa.bench check REPORT "user_recall_nst>=0.95" ...`` asserts conditions on a written
report: it prints every metric's value and CI and exits non-zero if any condition fails, or if its
metric is missing or null. The gate numbers live in the task file, not in ``gates.toml``: that file
lists every metric with NO threshold yet (roadmap B5 records the thresholds from the baseline run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import gates, metrics

CONDITION = gates.CONDITION


# --------------------------------------------------------------------------- report

def _thresholds() -> dict[str, tuple[str, float]]:
    """The thresholds ``voxa/bench/gates.toml`` sets for this run (B3 sets none, so it is empty)."""
    return gates.thresholds()


def build_report(suite: str, seed: int, echo_mode: str, results: list[dict],
                 facts: list[dict], audio_seconds: float, wall: float, model: str) -> dict:
    """The report envelope: every metric with value and CI, per scenario, plus the run block."""
    measured = metrics.compute(facts, seed=seed)
    by_scenario: dict[str, dict] = {}
    scenarios = sorted({str(fact["scenario"]) for fact in facts})
    for scenario in scenarios:
        group = [fact for fact in facts if str(fact["scenario"]) == scenario]
        group_metrics = metrics.compute(group, seed=seed)
        by_scenario[scenario] = {
            "clips": len(group),
            "metrics": {name: {"value": row["value"], "ci": row["ci"], "samples": row["samples"]}
                        for name, row in group_metrics.items()},
        }
    thresholds = _thresholds()
    for _name, row in measured.items():
        row["threshold"] = None
        row["pass"] = True
    for name, (operator, limit) in thresholds.items():
        row = measured.get(name)
        if row is None:
            continue
        row["threshold"] = limit
        row["threshold_op"] = operator
        row["pass"] = True if row["value"] is None else bool(gates.compare(row["value"], operator, limit))
    unrecovered = sum(
        1 for fact in facts
        if fact["delay_est"] is not None
        and not any(float(frame["erle_db"]) >= metrics.RECONVERGE_TARGET_DB
                    and float(frame["t_s"]) > float(fact["delay_est"]["t_s"])
                    for frame in fact["frames"]))
    return {
        "suite": suite,
        "seed": seed,
        "echo_mode": echo_mode,
        "pass": all(bool(row.get("pass", True)) for row in measured.values()),
        "metrics": measured,
        "by_scenario": by_scenario,
        "run": {
            "clips": len(results),
            "audio_seconds": round(audio_seconds, 3),
            "wall_seconds": round(wall, 3),
            "realtime_factor": round(wall / audio_seconds, 4) if audio_seconds else None,
            "target_realtime_factor": 0.5,
            "model": model or "base",
            "chunk_ms": metrics.CHUNK_MS,
            "asr_calls": sum(int(item.get("asr_calls", 0)) for item in results),
            "asr_audio_seconds": round(sum(float(item.get("asr_audio_seconds", 0.0)) for item in results), 3),
            "asr_processing_seconds": round(sum(float(item.get("asr_processing_seconds", 0.0))
                                                for item in results), 3),
            "turns": sum(len(item.get("requests", [])) for item in results),
            "suspects": sum(int(item["counts"].get("suspect", 0)) for item in results),
            "barge_ins": sum(int(item["counts"].get("barge_in", 0)) for item in results),
            "own_voice_rejections": sum(int(item["counts"].get("own_voice", 0)) for item in results),
            "ignored_utterances": sum(int(item["counts"].get("ignored", 0)) for item in results),
            "exit_phrases": sum(int(item["counts"].get("exit", 0)) for item in results),
            "delay_estimated": sum(1 for item in results if item.get("delay_estimated_samples")),
            "reconverge_unrecovered": unrecovered,
            "degraded": sorted({item["degraded"] for item in results if item.get("degraded")}),
            "unmeasured": sorted(name for name, row in measured.items() if row["value"] is None),
        },
        "clips": results,
    }


def format_metrics(report: dict) -> list[str]:
    """One aligned line per metric: value, CI, threshold, pass - what ``run`` prints."""
    lines = []
    for name, row in report["metrics"].items():
        value = "null" if row["value"] is None else f"{row['value']:.4f}"
        ci = "null" if row["ci"][0] is None else f"[{row['ci'][0]:.4f}, {row['ci'][1]:.4f}]"
        threshold = row.get("threshold")
        limit = "no threshold yet" if threshold is None else f"{row.get('threshold_op', '>=')} {threshold}"
        flag = "unmeasured" if row["value"] is None else ("ok" if row["pass"] else "FAILED")
        lines.append(f"  {name:<28} {value:>10}  ci {ci:<24}  {limit:<18} {flag}"
                     + (f"  ({row['reason']})" if row.get("reason") else ""))
    return lines


# --------------------------------------------------------------------------- check

def check(argv: list[str] | None = None) -> int:
    """``python -m voxa.bench check REPORT "<metric><op><number>" ...``; non-zero unless all hold."""
    parser = argparse.ArgumentParser(
        prog="python -m voxa.bench check",
        description="assert metric conditions on a report.json (exits 1 on any failure, gap or null)")
    parser.add_argument("report", type=Path, help="the report written by `run`")
    parser.add_argument("conditions", nargs="+",
                        help='conditions like user_recall_nst>=0.95, duck_latency_p50_ms<=300')
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"check: cannot read {args.report}: {exc}", file=sys.stderr)
        return 2
    known = metrics.METRIC_NAMES
    failures = 0
    for condition in args.conditions:
        name, operator, limit = gates.parse_condition(condition)
        if name is None:
            print(f"{condition}: unparsable condition")
            failures += 1
            continue
        row = report.get("metrics", {}).get(name)
        if row is None:
            print(f"{condition}: MISSING {name} is not in the report"
                  + (f" (known metrics: {' '.join(known[:4])} ...)" if name not in known else ""))
            failures += 1
            continue
        value = row.get("value")
        ci = row.get("ci", [None, None])
        shown = "null" if value is None else f"{float(value):.4f}"
        ci_shown = "null" if ci[0] is None else f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
        if value is None:
            print(f"{condition}: NULL {name} = null ci {ci_shown} "
                  f"({row.get('reason', 'not measurable at this stage')})")
            failures += 1
            continue
        holds = gates.compare(float(value), operator, limit)
        print(f"{condition}: {name} = {shown} ci {ci_shown} "
              f"{'OK' if holds else 'FAIL'} (samples {row.get('samples', 0)})")
        if not holds:
            failures += 1
    if failures:
        print(f"check FAILED: {failures} of {len(args.conditions)} conditions do not hold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
