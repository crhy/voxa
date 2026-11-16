"""Meeting and call notes: pure Python, standard library only.

One transcribed call becomes one dated block in the Voxa Meeting Notes file: its
transcript, the decisions heard in it, the action items heard in it and the speakers
whose "Name:" prefixes appear. The transcript never leaves the machine: this module
only handles text, and the tool that feeds it reads the recording locally and writes
the block locally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from voxa.agent.dates import MONTH_NAMES, NUMBER_WORDS, WEEKDAY_NAMES, WEEKDAYS, _ordinal

FILE_NAME = "Voxa Meeting Notes.md"

_TITLE_LINE = re.compile(r"^## (\d{4}-\d{2}-\d{2}) \| (.+)$")
_DECISIONS_HEAD = "Decisions:"
_ACTIONS_HEAD = "Action items:"
_SPEAKERS_HEAD = "Speakers:"
_TRANSCRIPT_HEAD = "Transcript:"
_HEADS = (_DECISIONS_HEAD, _ACTIONS_HEAD, _SPEAKERS_HEAD, _TRANSCRIPT_HEAD)

_DECISION = re.compile(r"\b(decid\w*|agree\w*|consensus|we'll|let's)\b", re.IGNORECASE)
_ACTION = re.compile(
    r"\b(i'll|i will|call me back|call (?:him|her|them|you) back|send\w*|pick up|schedule\w*)\b",
    re.IGNORECASE,
)
_SPEAKER_LINE = re.compile(r"^\s*([A-Z][a-z']{1,15})\s*:\s*(.+)$")
_SPEAKER_PREFIX = re.compile(r"^([A-Z][a-z']{1,15}):\s+")
_NUMBER = re.compile(r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)")
_DAYS_AGO = re.compile(rf"(?P<count>{_NUMBER.pattern}) days? (?:ago|back|past)")
_WEEKDAYS_AGO = re.compile(rf"(?P<count>{_NUMBER.pattern}) weekdays? (?:ago|back|past)")
# Words that open a line with a colon without naming a speaker.
_NOT_SPEAKERS = frozenset(
    {
        "transcript",
        "decisions",
        "action",
        "action items",
        "speakers",
        "summary",
        "note",
        "next",
        "then",
        "so",
        "and",
        "but",
        "we",
        "i",
        "they",
    }
)


@dataclass(frozen=True, slots=True)
class Record:
    """One meeting in the archive."""

    day: date
    title: str
    decisions: tuple[str, ...]
    actions: tuple[str, ...]
    speakers: tuple[str, ...]
    transcript: str


def _plain(text: str) -> str:
    """Whisper writes curly apostrophes; the rules below read straight ones."""
    return " ".join(text.replace("’", "'").replace("‘", "'").split())


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"[.!?\n]+", _plain(text)) if part.strip()]


def _sentence_form(sentence: str) -> str:
    body = " ".join(sentence.strip().split())
    if not body:
        return ""
    body = body[0].upper() + body[1:]
    if body[-1] not in ".!?":
        body += "."
    return body


def _without_speaker(sentence: str) -> str:
    """Drop the "Name:" a transcript line starts with; the sentence is what matters."""
    body = sentence.strip()
    match = _SPEAKER_PREFIX.match(body)
    if match is not None and match.group(1).lower() not in _NOT_SPEAKERS:
        return body[match.end() :]
    return body


def _classify(text: str) -> tuple[list[str], list[str]]:
    """Split a transcript's sentences into decisions and action items.

    A sentence that carries something to do counts as an action item even when it
    also says "we'll": one sentence lands in one list, and the actionable one wins.
    """
    found: list[str] = []
    actions: list[str] = []
    for sentence in _sentences(text):
        body = _sentence_form(_without_speaker(sentence))
        if _ACTION.search(body):
            if body not in actions:
                actions.append(body)
        elif _DECISION.search(body) and body not in found:
            found.append(body)
    return found, actions


def decisions(text: str) -> list[str]:
    """The sentences that state a decision: decided, agreed, consensus, "we'll", "let's"."""
    return _classify(text)[0]


def action_items(text: str) -> list[str]:
    """The sentences that assign something to do: "I'll", "I will", "call me back",
    "send", "pick up", "schedule"."""
    return _classify(text)[1]


def speakers(text: str) -> list[str]:
    """The speakers of a transcript, taken from its "Name:" line prefixes."""
    out: list[str] = []
    for line in text.splitlines():
        match = _SPEAKER_LINE.match(line)
        if match is None or match.group(1).lower() in _NOT_SPEAKERS:
            continue
        if match.group(1) not in out:
            out.append(match.group(1))
    return out


def record(title: str, day: date, transcript: str) -> str:
    """One dated block: transcript, decisions, action items and speakers."""
    clean = " ".join(title.strip().split())
    clean = (clean[0].upper() + clean[1:]) if clean else "Call"
    found, actions = _classify(transcript)
    lines = [f"## {day:%Y-%m-%d} | {clean}", _DECISIONS_HEAD]
    lines += [f"- {item}" for item in found]
    lines.append(_ACTIONS_HEAD)
    lines += [f"- {item}" for item in actions]
    lines.append(_SPEAKERS_HEAD)
    lines += [f"- {name}" for name in speakers(transcript)]
    lines.append(_TRANSCRIPT_HEAD)
    lines += [line for line in transcript.strip().splitlines() if line.strip()]
    return "\n".join(lines) + "\n"


def append_record(content: str, block: str) -> str:
    """Add a block to the end of the archive."""
    base = content.strip("\n")
    if not base:
        return block
    return base + "\n\n" + block


def parse(content: str) -> list[Record]:
    """Every block in the archive, oldest first."""
    out: list[Record] = []
    day: date | None = None
    title = ""
    section = ""
    found: list[str] = []
    actions: list[str] = []
    names: list[str] = []
    transcript: list[str] = []

    def flush() -> None:
        if day is None:
            return
        out.append(
            Record(
                day=day,
                title=title,
                decisions=tuple(found),
                actions=tuple(actions),
                speakers=tuple(names),
                transcript="\n".join(transcript).strip(),
            )
        )

    for line in content.splitlines():
        match = _TITLE_LINE.match(line)
        if match:
            flush()
            day, title = date.fromisoformat(match.group(1)), match.group(2)
            section, found, actions, names, transcript = "", [], [], [], []
            continue
        if day is None:
            continue
        stripped = line.strip()
        if stripped in _HEADS:
            section = stripped
            continue
        if section == _TRANSCRIPT_HEAD:
            transcript.append(line)
            continue
        if not stripped.startswith("- "):
            continue
        body = stripped[2:].strip()
        if section == _DECISIONS_HEAD:
            found.append(body)
        elif section == _ACTIONS_HEAD:
            actions.append(body)
        elif section == _SPEAKERS_HEAD:
            names.append(body)
    flush()
    return out


def search(content: str, keyword: str | None = None, day: date | None = None) -> list[Record]:
    """The blocks matching a day, a keyword, or both."""
    want = " ".join(_plain(keyword or "").lower().split())
    out: list[Record] = []
    for item in parse(content):
        if day is not None and item.day != day:
            continue
        if want:
            haystack = " ".join(
                [item.title, *item.decisions, *item.actions, *item.speakers, item.transcript]
            )
            if want not in " ".join(_plain(haystack).lower().split()):
                continue
        out.append(item)
    return out


def _count(word: str) -> int:
    return int(word) if word.isdigit() else NUMBER_WORDS[word]


def relative_day(spoken: str, today: date) -> date | None:
    """A past day said in words; None when it is not a day.

    ``dates.parse_day`` reads the future forms the calendar needs and its tests pin;
    the recall phrases of crhy/voxa#65 ("what did we decide yesterday?") need the past
    forms, so they are read here instead of changing that shared parser.
    """
    s = _plain(spoken).lower()
    if not s:
        return None
    if s in ("today", "this day"):
        return today
    if s in ("yesterday", "the day yesterday", "yesterday's"):
        return today - timedelta(days=1)
    if s in ("the day before yesterday", "day before"):
        return today - timedelta(days=2)
    match = _DAYS_AGO.fullmatch(s)
    if match:
        return today - timedelta(days=_count(match.group("count")))
    match = _WEEKDAYS_AGO.fullmatch(s)
    if match:
        return today - timedelta(weeks=_count(match.group("count")))
    if s in ("last week", "this week", "the week before", "last week's"):
        return today - timedelta(days=7)
    if s in ("last month", "the month before"):
        year, month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        return date(year, month, min(today.day, 28))
    match = re.fullmatch(r"(?:last |this |past )?([a-z]+)", s)
    if match and match.group(1) in WEEKDAYS:
        back = (today.weekday() - WEEKDAYS[match.group(1)]) % 7
        return today - timedelta(days=back or 7)
    return None


def relative_span(spoken: str, today: date) -> tuple[date, date] | None:
    """The window a day phrase covers: one day, or the seven of a named week."""
    day = relative_day(spoken, today)
    if day is None:
        return None
    s = _plain(spoken).lower()
    if s in ("last week", "this week", "the week before", "last week's"):
        end = today if s == "this week" else today - timedelta(days=1)
        return (end - timedelta(days=6), end)
    if re.fullmatch(rf"(?:this |last |past )?(?:the )?{_NUMBER.pattern} ?weeks?(?: (?:ago|back|past))?", s):
        return (day - timedelta(days=6), day)
    return (day, day)


def _day_phrase(day: date, today: date) -> str:
    """"today", "yesterday", "Friday" (within a week) or "October 16th"."""
    delta = (day - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "yesterday"
    if delta == -2:
        return "the day before yesterday"
    if -6 <= delta <= 6:
        return WEEKDAY_NAMES[day.weekday()].lower()
    return f"{MONTH_NAMES[day.month - 1]} {_ordinal(day.day)}".lower()


def _count_word(n: int, singular: str) -> str:
    return f"{n} {singular}" if n == 1 else f"{n} {singular}s"


def spoken_recall(items: list[Record], today: date, limit: int = 3) -> str:
    """What the archive holds for the day or keyword that was asked about."""
    if not items:
        return "I have no meeting notes for that."
    newest = sorted(items, key=lambda item: item.day, reverse=True)[:limit]
    sentences = [f"You have {_count_word(len(items), 'meeting note')} for that: {_named(newest, today)}."]
    for item in newest:
        bits = []
        if item.decisions:
            bits.append(f"{_count_word(len(item.decisions), 'decision')}: " + " ".join(item.decisions))
        if item.actions:
            bits.append(f"{_count_word(len(item.actions), 'action item')}: " + " ".join(item.actions))
        if bits:
            sentences.append(f"{item.title}: " + " ".join(bits))
    return " ".join(sentences)


def _named(items: list[Record], today: date) -> str:
    head = items[:3]
    parts = [f"{item.title} {_day_phrase(item.day, today)}" for item in head]
    if len(items) > len(head):
        parts.append(f"{len(items) - len(head)} more")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def spoken_decisions(items: list[Record], today: date, limit: int = 3) -> str:
    """The decisions of the named meetings, spoken back."""
    found = [decision for item in items for decision in item.decisions][:limit]
    if not found:
        return "I heard no decisions in those notes."
    if len(found) == 1:
        return f"One decision: {found[0]}"
    return f"{len(found)} decisions: " + " ".join(found)


def spoken_actions(items: list[Record], today: date, limit: int = 5) -> str:
    """The action items of the named meetings, spoken back."""
    found = [action for item in items for action in item.actions][:limit]
    if not found:
        return "I heard no action items in those notes."
    if len(found) == 1:
        return f"One action item: {found[0]}"
    return f"{len(found)} action items: " + " ".join(found)
