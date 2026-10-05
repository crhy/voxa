"""Latency reporting for voice commands."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voxa.agent.actionlog import ActionLog

STAGES = (
    "transcribe_ms",
    "route_ms",
    "act_ms",
    "first_token_ms",
    "to_speech_ms",
    "total_ms",
)


def load_records(path: Path | None, last: int | None) -> list[dict[str, Any]]:
    """Read action-log records, keeping only the most recent `last`."""
    return ActionLog(path).read(limit=last)


def latency_report(records: list[dict]) -> dict:
    """Summarise timing records into per-stage stats and the slowest commands."""
    samples: dict[str, list[int]] = {stage: [] for stage in STAGES}
    slowest: list[dict[str, Any]] = []
    count = 0
    for record in records:
        if record.get("route") != "timing":
            continue
        timings = record.get("timings")
        if not timings:
            continue
        count += 1
        for stage in STAGES:
            value = timings.get(stage)
            if isinstance(value, int) and not isinstance(value, bool):
                samples[stage].append(value)
        total = timings.get("total_ms")
        if isinstance(total, int) and not isinstance(total, bool):
            slowest.append({"heard": str(record.get("heard", "")), "total_ms": total})

    stages: dict[str, dict[str, int]] = {}
    for stage, values in samples.items():
        if not values:
            continue
        values.sort()
        n = len(values)
        p95_index = math.ceil(0.95 * n) - 1
        stages[stage] = {
            "median": values[n // 2],
            "p95": values[p95_index],
            "max": values[-1],
            "samples": n,
        }

    slowest.sort(key=lambda entry: entry["total_ms"], reverse=True)
    return {"count": count, "stages": stages, "slowest": slowest[:5]}


def format_latency(report: dict) -> str:
    """Render a latency report as Markdown."""
    if report.get("count", 0) == 0:
        return "No timed commands yet."

    lines = ["| Stage | Median | 95th | Max | Samples |", "|-------|--------|------|-----|---------|"]
    for stage in STAGES:
        stats = report["stages"].get(stage)
        if stats is None:
            continue
        lines.append(
            f"| {stage} | {stats['median']} ms | {stats['p95']} ms | {stats['max']} ms | {stats['samples']} |"
        )

    lines.append("")
    lines.append("## Slowest commands")
    for entry in report["slowest"]:
        lines.append(f"- {entry['heard']}: {entry['total_ms']} ms")

    return "\n".join(lines)
