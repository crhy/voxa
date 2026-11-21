#!/usr/bin/env python3
"""Build the docs/hearing/frontier.md table from the bench reports (Hearing 2, task h01).

The table is generated from the report JSONs, never typed by hand: one flag row per echo mode, one
column per metric the Hearing 2 brief names, each with its value and its 95 % CI exactly as
voxa/bench/metrics.py computed them, plus the mode's own latency and dependency size.

usage: tools/hearing_frontier_table.py [report.json ...]
       (default: .voxa-bench/frontier-*-full.json, then frontier-*-smoke.json)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

# The metrics Hearing 2 named, in the order the frontier table reads.
METRICS = ("erle_fst_median_db", "erle_fst_p10_db", "nearend_atten_dt_db", "echo_leak_word_rate",
           "dt_wer_penalty_pts", "user_recall_nst", "delay_est_err_ms", "cpu_core_share",
           "erle_music_loss_db")

UNITS = {"erle_fst_median_db": "dB", "erle_fst_p10_db": "dB", "nearend_atten_dt_db": "dB",
         "echo_leak_word_rate": "frac", "dt_wer_penalty_pts": "pts",
         "user_recall_nst": "frac", "delay_est_err_ms": "ms", "cpu_core_share": "cores",
         "erle_music_loss_db": "dB"}


def fmt(block: dict | None) -> str:
    """value [lo, hi] exactly as the bench computed it, or the honest marker."""
    if not block or block.get("value") is None:
        return "unmeasured"
    value = block["value"]
    ci = block.get("ci") or [None, None]
    digits = 4 if abs(float(value)) < 1.0 else 2
    text = f"{float(value):.{digits}f}"
    if ci[0] is None or ci[1] is None:
        return text
    return f"{text} [{float(ci[0]):.{digits}f}, {float(ci[1]):.{digits}f}]"


def latency_ms(report: dict) -> tuple[float | None, float | None]:
    """(echo-stage ms per second of audio, echo-stage share of one core). The runner times the
    echo stages separately from the ASR stage, so the arm's own real-time cost is measurable:
    cpu_core_share in the metrics block is the WHOLE pipeline, which the gate records."""
    run = report.get("run", {})
    share = run.get("echo_cpu_core_share")
    audio = run.get("audio_seconds") or 0.0
    echo = run.get("echo_cpu_seconds")
    per_audio_ms = (float(echo) / float(audio) * 1000.0) if echo and audio else None
    return (per_audio_ms, None if share is None else float(share))


def dependency_size(report: dict) -> str:
    """Runtime dependency the arm needs, in MB on disk (what the Flatpak would have to carry)."""
    sizes = report.get("run", {}).get("echo_mode_dependency_mb")
    return "" if sizes is None else json.dumps(sizes, sort_keys=True)


def row(path: Path) -> str:
    report = json.loads(path.read_text())
    mode = report.get("echo_mode", path.stem.split("-")[1])
    cells = [fmt(report.get("metrics", {}).get(m)) for m in METRICS]
    per_audio_ms, core_share = latency_ms(report)
    probe = report.get("run", {}).get("echo_mode_probe", {}).get(mode, {})
    lat = "unmeasured" if per_audio_ms is None else f"{per_audio_ms:.1f}"
    share = "unmeasured" if core_share is None else f"{core_share:.3f}"
    return (f"| {mode} | " + " | ".join(cells) +
            f" | {lat} | {share} | {probe.get('reason', '')} |")


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        paths = sorted(glob.glob(".voxa-bench/frontier-*-full.json"))
        if not paths:
            paths = sorted(glob.glob(".voxa-bench/frontier-*-smoke.json"))
    header = ("| mode | " + " | ".join(f"{m} ({UNITS[m]})" for m in METRICS)
              + " | echo_ms_per_audio_s | echo_core_share | this machine |")
    print(header)
    print("|" + "---|" * (len(METRICS) + 3))
    for path in paths:
        print(row(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
