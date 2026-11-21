"""Show and delete the local turn log (crhy/voxa#116 §6, "easy to view and delete").

"Show my turn log" speaks the summary and names the file; "clear my turn log"
deletes it. Both are read-only: they never send anything anywhere, and they work
whether or not the log is switched on.
"""

from __future__ import annotations

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.turnlog import TurnLog, default_path, summarize


def _spoken_number(value: object) -> str:
    return "not measured yet" if value is None else str(value)


def _spoken_summary(summary: dict) -> str:
    review = summary.get("review", {})
    escalation = summary.get("escalation", {})
    spoken = summary.get("first_spoken_ms", {})
    parts = [
        f"{summary.get('total', 0)} turns logged.",
        f"{escalation.get('escalated', 0)} of {escalation.get('decisions', 0)} went to the big brain"
        f"; escalation accuracy {_spoken_number(escalation.get('accuracy'))}.",
        f"{review.get('reviewed', 0)} answers were reviewed, and the reviewer disagreed with "
        f"{review.get('disagreed', 0)} of them; review hit rate {_spoken_number(review.get('hit_rate'))}.",
        f"Median time to the first spoken word {_spoken_number(spoken.get('median'))} milliseconds."
        f" Little brain {_spoken_number(spoken.get('little'))}, big brain {_spoken_number(spoken.get('big'))}.",
        f"{summary.get('corrections', 0)} corrections and {summary.get('barge_ins', 0)} barge-ins.",
    ]
    return " ".join(parts)


def turn_log_show(args: dict[str, str]) -> ToolResult:
    log = TurnLog()
    records = log.read()
    path = default_path()
    if not records:
        return ToolResult.success(
            f"The turn log is empty. It stays off until you switch it on, and it lives in {path}.",
            detail=str(path),
        )
    summary = summarize(records)
    return ToolResult.success(
        f"{_spoken_summary(summary)} The file is {path}.",
        detail=f"{path}: {summary}",
    )


def turn_log_clear(args: dict[str, str]) -> ToolResult:
    log = TurnLog()
    removed = log.clear()
    path = default_path()
    if not removed:
        return ToolResult.success(f"The turn log is already empty. Nothing to delete in {path}.", detail=str(path))
    return ToolResult.success(f"The turn log is deleted from {path}.", detail=str(path))


def turn_log_tools() -> list[Tool]:
    return [
        Tool(
            name="turn_log_show",
            description="Summarize the local turn log (which brain answered, escalation accuracy, review hits, "
            "time to first spoken word) and say where the file is.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=turn_log_show,
            required=(),
            timeout_seconds=10.0,
        ),
        Tool(
            name="turn_log_clear",
            description="Delete the local turn log file.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=turn_log_clear,
            required=(),
            timeout_seconds=10.0,
        ),
    ]
