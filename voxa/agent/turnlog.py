"""The local, opt-in turn log from crhy/voxa#116 §6 ("live test data, local only").

Pure Python: no GTK, no host access. The log is a JSONL file, one record per
turn, so it can be read, summarised and deleted without any special tooling.
Nothing is written unless the user opted in (``Settings.turn_log``), the file
stays on this machine, and ``turn_log_clear`` deletes it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["TurnLog", "TurnRecord", "default_path", "summarize"]

LITTLE_BRAIN = "little"
BIG_BRAIN = "big"


def default_path() -> Path:
    """Where the turn log lives: VOXA_TURN_LOG, else the XDG state dir."""
    override = os.environ.get("VOXA_TURN_LOG")
    if override:
        return Path(override)
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "voxa" / "turns.jsonl"


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One line of the log: the fields #116 asks for on every turn."""

    time: str
    kind: str
    brain: str = LITTLE_BRAIN
    first_spoken_ms: int = 0
    escalated: bool | None = None
    escalate_reason: str = ""
    review_verdict: str = ""
    review_note: str = ""
    correction: str = ""
    barge_in: bool = False

    def to_json(self) -> str:
        data = {
            "time": self.time,
            "kind": self.kind,
            "brain": self.brain,
            "first_spoken_ms": int(self.first_spoken_ms or 0),
            "escalated": self.escalated,
            "escalate_reason": self.escalate_reason,
            "review_verdict": self.review_verdict,
            "correction": self.correction,
            "barge_in": bool(self.barge_in),
        }
        if self.review_note:
            data["review_note"] = self.review_note
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class TurnLog:
    """Append-only JSONL log, off until the user opts in.

    Writing never raises: logging is never a reason for a turn to fail, and a
    log that is switched off never creates a file.
    """

    def __init__(self, path: Path | None = None, *, enabled: bool = False, max_bytes: int = 2_000_000) -> None:
        self.path = Path(path) if path is not None else default_path()
        self.enabled = enabled
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def record(self, record: TurnRecord) -> None:
        """Append one turn. Opted out: nothing at all is written."""
        if not self.enabled:
            return
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
                log.warning("could not write turn log: %s", exc)

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

    def clear(self) -> bool:
        """Delete the log (and its rotated copy). Returns whether anything went."""
        removed = False
        with self._lock:
            for path in (self.path, Path(str(self.path) + ".1")):
                try:
                    path.unlink()
                    removed = True
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    log.warning("could not delete turn log: %s", exc)
                    break
        return removed


def _median_ms(values: Iterable[int]) -> int:
    values = sorted(values)
    if not values:
        return 0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) // 2


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Escalation accuracy, review hit rate, median time to first spoken word.

    ``escalation.accuracy`` is measured over the turns the little brain answered
    *and* the big brain reviewed: a reviewed answer the reviewer agreed with is
    a turn that did not need escalating (``confirmed``), one it disagreed with is
    a turn that should have escalated (``missed``). ``review.hit_rate`` is the
    share of reviewed answers where the reviewer found a real error. Both are
    ``None`` until there is data to measure.
    """
    by_kind: dict[str, int] = {}
    by_brain: dict[str, int] = {}
    escalated = 0
    decisions = 0
    reviewed = 0
    disagreed = 0
    confirmed = 0
    missed = 0
    corrections = 0
    barge_ins = 0
    spoken: dict[str, list[int]] = {}

    for record in records:
        kind = str(record.get("kind", ""))
        by_kind[kind] = by_kind.get(kind, 0) + 1
        brain = str(record.get("brain", "")) or LITTLE_BRAIN
        by_brain[brain] = by_brain.get(brain, 0) + 1

        choice = record.get("escalated", None)
        if choice is not None:
            decisions += 1
            if choice:
                escalated += 1

        verdict = str(record.get("review_verdict", "")).strip().lower()
        if verdict:
            reviewed += 1
            if verdict == "disagree":
                disagreed += 1
            if not choice:
                if verdict == "agree":
                    confirmed += 1
                elif verdict == "disagree":
                    missed += 1

        if str(record.get("correction", "")).strip():
            corrections += 1
        if record.get("barge_in"):
            barge_ins += 1

        try:
            ms = int(record.get("first_spoken_ms", 0) or 0)
        except (TypeError, ValueError):
            ms = 0
        if ms > 0:
            spoken.setdefault(brain, []).append(ms)

    measured = confirmed + missed
    accuracy = round(confirmed / measured, 3) if measured else None
    hit_rate = round(disagreed / reviewed, 3) if reviewed else None

    return {
        "total": len(records),
        "by_kind": by_kind,
        "by_brain": by_brain,
        "escalation": {
            "decisions": decisions,
            "escalated": escalated,
            "confirmed": confirmed,
            "missed": missed,
            "accuracy": accuracy,
        },
        "review": {
            "reviewed": reviewed,
            "disagreed": disagreed,
            "hit_rate": hit_rate,
        },
        "first_spoken_ms": {
            "median": _median_ms([ms for values in spoken.values() for ms in values]),
            LITTLE_BRAIN: _median_ms(spoken.get(LITTLE_BRAIN, [])),
            BIG_BRAIN: _median_ms(spoken.get(BIG_BRAIN, [])),
        },
        "corrections": corrections,
        "barge_ins": barge_ins,
    }
