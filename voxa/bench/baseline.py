"""B5: the baseline record every later hearing task is measured against.

``python -m voxa.bench baseline record REPORT...`` merges the reports ``run`` wrote (each carries
its own ``suite`` and ``echo_mode``) into ``voxa/bench/baseline.json``: every metric of every mode
with its value, its bootstrap CI and its sample count, plus the run block (clips, audio and wall
seconds, realtime factor, degraded notes). That file is the frozen "what today looks like" record;
``voxa/bench/gates.toml`` holds the ratchet derived from it, and the roadmap's rule - "no other bench
metric regresses beyond its CI versus ``baseline.json``" - is what :func:`regressions` checks.

The ratchet bound for one metric is the CI bound on its BAD side: for a ``>=`` metric the low
bound, for a ``<=`` metric the high bound. A later task passes as long as its value stays inside
the baseline's spread; a value outside it is a regression, not a rounding difference.

``python -m voxa.bench baseline check REPORT`` compares one fresh report with the baseline entry of
the same suite and echo mode and exits non-zero on any regression or any metric that went unmeasured.
Recording a baseline is a one-off act: re-recording it to make a failing task green is forbidden by
the shared brief, exactly like loosening a threshold in ``gates.toml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).with_name("baseline.json")
RUN_KEYS = ("clips", "audio_seconds", "wall_seconds", "realtime_factor", "turns", "barge_ins",
            "suspects", "asr_calls", "asr_audio_seconds", "asr_processing_seconds",
            "delay_estimated", "reconverge_unrecovered", "degraded", "unmeasured")


def metric_rows(report: dict) -> dict[str, dict]:
    """The defensible part of a report's metrics block: value, CI, samples, unit, direction."""
    out: dict[str, dict] = {}
    for name, row in (report.get("metrics") or {}).items():
        out[name] = {"value": row.get("value"), "ci": row.get("ci", [None, None]),
                     "samples": row.get("samples", 0), "unit": row.get("unit"),
                     "direction": row.get("direction"),
                     "reason": row.get("reason", "")}
    return out


def run_block(report: dict) -> dict:
    """The wall-clock and count facts of one run, so the doc's timings are reproducible."""
    run = report.get("run") or {}
    return {key: run.get(key) for key in RUN_KEYS if key in run}


def bad_side(row: dict) -> float | None:
    """The CI bound on the bad side of ``row``: low bound for ``>=``, high bound for ``<=``."""
    ci = row.get("ci") or [None, None]
    if ci[0] is None or ci[1] is None:
        return None
    direction = row.get("direction") or ">="
    return float(ci[0]) if direction == ">=" else float(ci[1])


def ratchet(metrics_block: dict, suite: str, echo_mode: str) -> dict:
    """The ratchet thresholds the baseline implies for one suite/mode (roadmap B5's rule)."""
    thresholds: dict[str, float] = {}
    for name, row in metric_rows({"metrics": metrics_block}).items():
        bound = bad_side(row)
        if bound is not None:
            thresholds[name] = bound
    return {"suite": suite, "echo_mode": echo_mode,
            "rule": "smoke baseline CI bound on the bad side",
            "thresholds": thresholds}


def record(reports: list[dict], *, default_echo_mode: str = "voxa") -> dict:
    """Merge run reports into the baseline envelope, keyed by echo mode then suite."""
    modes: dict[str, dict] = {}
    for report in reports:
        mode, suite = str(report.get("echo_mode", "?")), str(report.get("suite", "?"))
        modes.setdefault(mode, {})[suite] = {"metrics": metric_rows(report), "run": run_block(report)}
    smoke_voxa = modes.get(default_echo_mode, {}).get("smoke", {}).get("metrics", {})
    return {
        "seed": next((int(r["seed"]) for r in reports if r.get("seed") is not None), 0),
        "default_echo_mode": default_echo_mode,
        "modes": modes,
        "ratchet": ratchet(smoke_voxa, "smoke", default_echo_mode),
    }


def regressions(report: dict, baseline: dict) -> list[str]:
    """Every metric of this run that fell outside the baseline's CI spread for the same mode."""
    mode = str(report.get("echo_mode", "?"))
    suite = str(report.get("suite", "?"))
    entry = (baseline.get("modes", {}).get(mode, {}) or {}).get(suite, {})
    if not entry:
        return [f"no baseline entry for suite={suite} echo_mode={mode}"]
    lines: list[str] = []
    for name, old in entry["metrics"].items():
        bound = bad_side(old)
        if bound is None:
            continue  # the baseline could not measure it either: nothing to hold this run to
        new = (report.get("metrics") or {}).get(name) or {}
        value = new.get("value")
        if value is None:
            lines.append(f"{name}: GAP measured {old['value']} at baseline, null in this run")
            continue
        direction = old.get("direction") or ">="
        worse = float(value) < bound if direction == ">=" else float(value) > bound
        if worse:
            lines.append(f"{name}: REGRESSION {float(value):.4f} beyond baseline "
                         f"{'CI low' if direction == '>=' else 'CI high'} {bound:.4f} "
                         f"(baseline value {old['value']:.4f}, ci {old['ci']})")
    return lines


def main(argv: list[str] | None = None) -> int:
    """``baseline record REPORT...`` writes baseline.json; ``baseline check REPORT`` reads it."""
    parser = argparse.ArgumentParser(prog="python -m voxa.bench baseline")
    parser.add_argument("command", choices=("record", "check"))
    parser.add_argument("reports", nargs="+", type=Path, help="report.json files written by `run`")
    parser.add_argument("--out", type=Path, default=BASELINE_PATH, help="baseline.json path")
    parser.add_argument("--default-echo-mode", default="voxa",
                        help="the mode the ratchet is taken from (voxa is the runner's default)")
    args = parser.parse_args(argv)
    loaded: list[dict] = []
    for path in args.reports:
        try:
            loaded.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"baseline: cannot read {path}: {exc}", file=sys.stderr)
            return 2
    if args.command == "record":
        envelope = record(loaded, default_echo_mode=args.default_echo_mode)
        args.out.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        for mode, suites in sorted(envelope["modes"].items()):
            for suite, entry in sorted(suites.items()):
                valued = sum(1 for row in entry["metrics"].values() if row["value"] is not None)
                print(f"baseline {mode}/{suite}: {valued} valued metrics, "
                      f"{entry['run'].get('clips')} clips, "
                      f"{entry['run'].get('wall_seconds')} s wall")
        print(f"wrote {args.out}: ratchet from "
              f"{envelope['ratchet']['echo_mode']}/{envelope['ratchet']['suite']} "
              f"({len(envelope['ratchet']['thresholds'])} thresholds)")
        return 0
    report = loaded[-1]
    baseline = json.loads(args.out.read_text(encoding="utf-8"))
    lines = regressions(report, baseline)
    for line in lines:
        print(line)
    if lines:
        print(f"baseline: {len(lines)} regression(s) against {args.out}", file=sys.stderr)
        return 1
    print(f"baseline: no regression against {baseline['modes'][report['echo_mode']][report['suite']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
