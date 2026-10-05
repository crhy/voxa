"""Stage timing measurements for voice commands."""

from __future__ import annotations

import threading
import time

STAGES = (
    "speech_end",
    "transcribed",
    "routed",
    "acted",
    "first_token",
    "speech_start",
    "done",
)


class Trace:
    """Thread-safe stage timing recorder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._marks: dict[str, float] = {}

    def mark(self, stage: str) -> None:
        """Record a monotonic timestamp for a stage. First mark wins."""
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        with self._lock:
            if stage not in self._marks:
                self._marks[stage] = time.monotonic()

    def since(self, start: str, end: str) -> int | None:
        """Return whole milliseconds between two stages, or None if either missing."""
        with self._lock:
            t0 = self._marks.get(start)
            t1 = self._marks.get(end)
        if t0 is None or t1 is None:
            return None
        return int((t1 - t0) * 1000)

    def to_dict(self) -> dict[str, int]:
        """Return timing measurements as a dict."""
        result = {}
        if "speech_end" in self._marks and "transcribed" in self._marks:
            result["transcribe_ms"] = self.since("speech_end", "transcribed")
        if "transcribed" in self._marks and "routed" in self._marks:
            result["route_ms"] = self.since("transcribed", "routed")
        if "routed" in self._marks and "acted" in self._marks:
            result["act_ms"] = self.since("routed", "acted")
        if "acted" in self._marks and "first_token" in self._marks:
            result["first_token_ms"] = self.since("acted", "first_token")
        if "acted" in self._marks and "speech_start" in self._marks:
            result["to_speech_ms"] = self.since("acted", "speech_start")
        if "speech_end" in self._marks and "done" in self._marks:
            result["total_ms"] = self.since("speech_end", "done")
        return result


def summary(timings: dict[str, int]) -> str:
    """Produce short status-line text from timing dict."""
    if not timings:
        return ""
    total = timings.get("total_ms")
    if total is None:
        return ""
    parts = []
    if "transcribe_ms" in timings:
        parts.append(f"heard {timings['transcribe_ms'] / 1000:.2f}")
    if "act_ms" in timings:
        parts.append(f"acted {timings['act_ms'] / 1000:.2f}")
    if parts:
        return f"{total / 1000:.1f} s ({', '.join(parts)})"
    return f"{total / 1000:.1f} s"
