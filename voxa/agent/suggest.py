"""Rule-based coaching: notice when the user did something the long way.

Pure Python: no GTK, no host access. ``suggest`` looks at the recent action log
and, when it spots a more direct path, returns one friendly ``Suggestion``.
``SuggestionState`` remembers what was already offered and dismissed so the
coaching is never a nag.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from voxa.agent import hearing
from voxa.agent.intents import ROUTED_TOOLS
from voxa.agent.planner import looks_like_command
from voxa.agent.tools.browser import SITES

log = logging.getLogger(__name__)

__all__ = ["Suggestion", "SuggestionState", "suggest", "SHORT_FORMS", "LONG_WAYS"]

# Tools the router can call instantly, and the short form the user can say.
SHORT_FORMS: dict[str, str] = {
    "play_youtube": "play {query}",
    "open_app": "open {name}",
    "close_app": "close {name}",
    "open_site": "open {name}",
    "web_search": "search the web for {query}",
    "search_youtube": "search YouTube for {query}",
    "switch_to": "switch to {name}",
}

# Known long ways round: (first tools, second tools, key, builder). The builder
# gets the two records and returns the sentence, or None when the pair does not
# really match (e.g. the first app is not a browser).
def _direct_browser(first: dict[str, Any], second: dict[str, Any]) -> str | None:
    app = str((first.get("args") or {}).get("name", "")).casefold()
    if app not in {"brave", "browser"}:
        return None
    target = str((second.get("args") or {}).get("name") or (second.get("args") or {}).get("url") or "")
    if not target:
        return None
    return f"You can skip opening the browser: just say “open {target}”."


def _direct_gmail(first: dict[str, Any], second: dict[str, Any]) -> str | None:
    name = str((first.get("args") or {}).get("name", "")).casefold()
    if name != "gmail":
        return None
    return "You can say “compose an email” and I'll go straight there."


def _direct_youtube(first: dict[str, Any], second: dict[str, Any]) -> str | None:
    name = str((first.get("args") or {}).get("name", "")).casefold()
    if name != "youtube":
        return None
    query = str((second.get("args") or {}).get("query", ""))
    if not query:
        return None
    return f"You can just say “play {query}”."


LONG_WAYS: list[tuple[tuple[str, ...], tuple[str, ...], str, Any]] = [
    (
        ("open_app",),
        ("open_site", "open_url", "browse"),
        "direct:open_app>open_site",
        _direct_browser,
    ),
    (
        ("open_site",),
        ("compose_gmail",),
        "direct:open_site>compose_gmail",
        _direct_gmail,
    ),
    (
        ("open_site",),
        ("search_youtube", "play_youtube"),
        "direct:open_site>play_youtube",
        _direct_youtube,
    ),
]


@dataclass(frozen=True, slots=True)
class Suggestion:
    key: str
    text: str
    kind: str
    action: dict[str, Any] | None = None


def _parse_time(value: Any) -> float | None:
    """Epoch seconds for an ISO timestamp, or None when it cannot be read."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _signature(record: dict[str, Any]) -> tuple[Any, ...]:
    args = record.get("args") or {}
    return (str(record.get("tool", "")), tuple(sorted(args.items())))


def _describe(tool: str, args: dict[str, Any]) -> str:
    """Plain words for a step, e.g. "open Gmail" or "play jazz"."""
    if tool == "compose_gmail":
        return "compose an email"
    verb = {
        "open_site": "open",
        "open_app": "open",
        "close_app": "close",
        "play_youtube": "play",
        "search_youtube": "search YouTube for",
        "web_search": "search the web for",
        "switch_to": "switch to",
    }.get(tool, tool.replace("_", " "))
    name = str(args.get("name") or args.get("query") or args.get("url") or "")
    if not name:
        return verb
    return f"{verb} {name[:1].upper()}{name[1:]}"


class SuggestionState:
    """Remembers what was offered and dismissed, in a JSON file."""

    def __init__(self, path: Path | None = None, clock=time.time) -> None:
        self.path = Path(path) if path is not None else None
        self.clock = clock
        self._lock = threading.Lock()
        self.dismissed: set[str] = set()
        self.dismissed_all = False
        self.counts: dict[str, int] = {}
        self.last_offer: float | None = None
        self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if not isinstance(payload, dict):
            return
        dismissed = payload.get("dismissed")
        if isinstance(dismissed, list):
            self.dismissed = {str(item) for item in dismissed}
        self.dismissed_all = bool(payload.get("dismissed_all"))
        counts = payload.get("counts")
        if isinstance(counts, dict):
            self.counts = {str(key): int(value) for key, value in counts.items() if isinstance(value, int)}
        last = payload.get("last_offer")
        self.last_offer = float(last) if isinstance(last, (int, float)) else None

    def _save(self) -> None:
        if self.path is None:
            return
        payload = {
            "dismissed": sorted(self.dismissed),
            "dismissed_all": self.dismissed_all,
            "counts": self.counts,
            "last_offer": self.last_offer,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(payload), encoding="utf-8")
            temp.replace(self.path)
        except OSError as exc:
            log.warning("could not write suggestions state: %s", exc)

    def allowed(self, key: str, min_gap_seconds: float = 600, max_times: int = 2) -> bool:
        if self.dismissed_all or key in self.dismissed:
            return False
        if self.counts.get(key, 0) >= max_times:
            return False
        if self.last_offer is not None and self.clock() - self.last_offer < min_gap_seconds:
            return False
        return True

    def offered(self, key: str) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            self.last_offer = self.clock()
            self._save()

    def dismiss(self, key: str) -> None:
        with self._lock:
            self.dismissed.add(key)
            self._save()

    def dismiss_all(self) -> None:
        with self._lock:
            self.dismissed_all = True
            self._save()


def _shortcut(records: list[dict[str, Any]]) -> Suggestion | None:
    newest = records[-1]
    if str(newest.get("route", "")) != "plan" or newest.get("ok") is not True:
        return None
    heard = str(newest.get("heard", ""))
    # Exactly one step: the record before it is not another step of the same plan.
    previous = records[-2] if len(records) >= 2 else None
    if previous is not None and str(previous.get("route", "")) == "plan" and str(previous.get("heard", "")) == heard:
        return None
    tool = str(newest.get("tool", ""))
    if tool not in ROUTED_TOOLS or tool not in SHORT_FORMS:
        return None
    args = newest.get("args") or {}
    try:
        short = SHORT_FORMS[tool].format(**args)
    except (KeyError, IndexError):
        return None
    return Suggestion(
        key=f"shortcut:{tool}",
        text=f"Next time you can just say “{short}” and I'll do it instantly.",
        kind="shortcut",
    )


def _direct(records: list[dict[str, Any]]) -> Suggestion | None:
    if len(records) < 2:
        return None
    first, second = records[-2], records[-1]
    for route in (first, second):
        if str(route.get("route", "")) != "tool" or route.get("ok") is not True:
            return None
    first_time = _parse_time(first.get("time"))
    second_time = _parse_time(second.get("time"))
    if first_time is None or second_time is None or abs(second_time - first_time) >= 60:
        return None
    for first_tools, second_tools, key, builder in LONG_WAYS:
        if str(first.get("tool", "")) in first_tools and str(second.get("tool", "")) in second_tools:
            text = builder(first, second)
            if text is not None:
                return Suggestion(key=key, text=text, kind="direct")
    # A web search that is really just looking for a known site.
    if str(second.get("tool", "")) == "web_search":
        query = str((second.get("args") or {}).get("query", "")).casefold().strip()
        for site in SITES:
            if query.startswith(site):
                return Suggestion(
                    key=f"direct:web_search:{site}",
                    text=f"I can open that directly: say “open {site}”.",
                    kind="direct",
                )
    return None


def _routine(records: list[dict[str, Any]]) -> Suggestion | None:
    for length in (3, 2):
        occurrences: dict[tuple[Any, ...], list[float]] = {}
        for start in range(0, len(records) - length + 1):
            window = records[start : start + length]
            if any(str(rec.get("route", "")) != "tool" or rec.get("ok") is not True for rec in window):
                continue
            times = [_parse_time(rec.get("time")) for rec in window]
            if any(value is None for value in times):
                continue
            signature = tuple(_signature(rec) for rec in window)
            occurrences.setdefault(signature, []).append(times[0])
        for signature, starts in occurrences.items():
            if len(starts) < 3:
                continue
            starts = sorted(starts)
            if all(b - a <= 120 for a, b in zip(starts, starts[1:], strict=False)):
                steps = [{"tool": tool, "args": dict(args)} for (tool, args) in signature]
                described = [_describe(step["tool"], step["args"]) for step in steps]
                joined = " then ".join(described)
                return Suggestion(
                    key="routine:" + ">".join(step["tool"] for step in steps),
                    text=f"You often do {joined}. Want me to make that one command?",
                    kind="routine",
                    action={"create_routine": steps, "name": joined},
                )
    return None


def _phrasing(records: list[dict[str, Any]]) -> Suggestion | None:
    counts: dict[str, int] = {}
    for record in records:
        if str(record.get("route", "")) != "model":
            continue
        text = hearing.normalize(str(record.get("heard", ""))).strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    for text, count in counts.items():
        if count >= 3 and looks_like_command(text):
            return Suggestion(
                key=f"phrasing:{text}",
                text=f"I keep not understanding “{text}”. Tell me what you'd like it to do and I'll remember it.",
                kind="phrasing",
            )
    return None


def suggest(records: list[dict[str, Any]], state: SuggestionState, enabled: bool = True) -> Suggestion | None:
    """One coaching suggestion for the recent log, or None."""
    if not enabled or not records:
        return None
    newest = records[-1]
    if newest.get("ok") is False:
        return None
    for finder in (_shortcut, _direct, _routine, _phrasing):
        suggestion = finder(records)
        if suggestion is not None and state.allowed(suggestion.key):
            return suggestion
    return None
