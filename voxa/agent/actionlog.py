"""A running log of every command Voxa heard and what it did about it.

Pure Python: no GTK, no host access. The log is a JSONL file, one record per
line, so it can be read, rotated and summarised without any special tooling.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["ActionLog", "ActionRecord", "default_path", "summarize"]


def default_path() -> Path:
    """Where the action log lives: VOXA_ACTION_LOG, else the XDG state dir."""
    override = os.environ.get("VOXA_ACTION_LOG")
    if override:
        return Path(override)
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "voxa" / "actions.jsonl"


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One line of the log: what was heard, how it was routed, what happened."""

    time: str
    heard: str
    route: str
    tool: str = ""
    args: dict[str, str] = field(default_factory=dict)
    ok: bool | None = None
    speech: str = ""
    detail: str = ""
    ms: int = 0
    timings: dict[str, int] | None = None

    def to_json(self) -> str:
        data = {
            "time": self.time,
            "heard": self.heard,
            "route": self.route,
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "speech": self.speech,
            "detail": self.detail,
            "ms": self.ms,
        }
        if self.timings:
            data["timings"] = self.timings
        return json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        )


class ActionLog:
    """Append-only JSONL log. Appending never raises: logging is never a reason
    for a command to fail."""

    def __init__(self, path: Path | None = None, max_bytes: int = 2_000_000) -> None:
        self.path = Path(path) if path is not None else default_path()
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def append(self, record: ActionRecord) -> None:
        line = record.to_json() + "\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if self.path.stat().st_size > self.max_bytes:
                        os.replace(self.path, Path(str(self.path) + ".1"))
                except OSError:
                    pass  # no file yet, or it vanished: nothing to rotate
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError as exc:
                log.warning("could not write action log: %s", exc)

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            return []
        records: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        if limit is not None and limit > 0:
            records = records[-limit:]
        return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts, per-tool averages, recent failures and unrouted questions."""
    by_route: dict[str, int] = {}
    tools: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    heard_counts: dict[str, int] = {}

    for record in records:
        route = str(record.get("route", ""))
        by_route[route] = by_route.get(route, 0) + 1
        if route == "tool":
            name = str(record.get("tool", ""))
            entry = tools.setdefault(name, {"calls": 0, "failed": 0, "ms": 0})
            entry["calls"] += 1
            try:
                entry["ms"] += int(record.get("ms", 0) or 0)
            except (TypeError, ValueError):
                pass
            if not record.get("ok"):
                entry["failed"] += 1
                failures.append(record)
        elif route == "model":
            heard = str(record.get("heard", ""))
            heard_counts[heard] = heard_counts.get(heard, 0) + 1

    by_tool = {
        name: {
            "calls": entry["calls"],
            "failed": entry["failed"],
            "avg_ms": entry["ms"] // entry["calls"] if entry["calls"] else 0,
        }
        for name, entry in tools.items()
    }
    unrouted = [
        {"heard": heard, "count": count}
        for heard, count in sorted(heard_counts.items(), key=lambda item: (-item[1], item[0]))
    ][:50]

    return {
        "total": len(records),
        "by_route": by_route,
        "by_tool": by_tool,
        "failures": failures[::-1][:20],
        "unrouted": unrouted,
    }
