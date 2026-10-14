from __future__ import annotations

import re
from datetime import datetime

from voxa.agent.calendar_local import (
    _day_word,
    on_day,
    spoken_day,
    spoken_time,
)
from voxa.agent.calendar_local import (
    entries as cal_entries,
)
from voxa.agent.dates import (
    _spoken_date_short,
    next_occurrence,
    spoken_upcoming,
    upcoming,
)
from voxa.agent.dates import (
    entries as date_entries,
)
from voxa.agent.lists import _join
from voxa.agent.lists import parse as list_parse
from voxa.agent.memory_facts import facts as fact_list
from voxa.agent.memory_facts import search

MONTHLY = re.compile(r"\d{1,2}")


def today_summary(calendar: str, dates_text: str, lists_text: str, now) -> str:
    """One spoken brief for today: appointments, things due soon, open to-dos."""
    today = now.date()
    parts: list[str] = []
    items = cal_entries(calendar)
    if on_day(items, today):
        parts.append(spoken_day(items, today, today))
    if upcoming(dates_text, today, 3):
        parts.append(spoken_upcoming(dates_text, today, 3))
    open_items = [text for done, text in list_parse(lists_text).get("To-do", []) if not done]
    if open_items:
        thing = "thing" if len(open_items) == 1 else "things"
        parts.append(f"You have {len(open_items)} {thing} open on your to-do list.")
    if not parts:
        return "Nothing needs your attention today."
    return " ".join(parts)


def forgotten(dates_text: str, lists_text: str, calendar: str, today) -> str:
    """Overdue one-off deadlines and to-do items left open; passed calendar items ignored."""
    parts: list[str] = []
    for date_text, kind, name in date_entries(dates_text):
        if kind != "deadline" or MONTHLY.fullmatch(date_text):
            continue
        day = next_occurrence(date_text, today)
        if day is None or day >= today:
            continue
        delta = (today - day).days
        disp = name if name.lower().startswith("the ") else f"The {name[:1].lower()}{name[1:]}"
        dayword = "day" if delta == 1 else "days"
        parts.append(f"{disp} was due {delta} {dayword} ago.")
    open_items = [text for done, text in list_parse(lists_text).get("To-do", []) if not done]
    if open_items:
        parts.append(f"Still open on your to-do list: {_join(open_items[:3])}.")
    if not parts:
        return "I don't see anything you have forgotten."
    return " ".join(parts)


def suggest_next(dates_text: str, lists_text: str, calendar: str, now) -> str:
    """The next appointment within 2 hours, else an urgent date item, else the first open to-do."""
    soon = None
    for day, time, title in cal_entries(calendar):
        if time is None:
            continue
        start = datetime(day.year, day.month, day.day, time[0], time[1])
        delta = (start - now).total_seconds()
        if 0 <= delta <= 2 * 3600 and (soon is None or start < soon[0]):
            soon = (start, title, time)
    if soon:
        _start, title, time = soon
        return f"Get ready: {title} at {spoken_time(time)}."
    today = now.date()
    candidates = []
    for date_text, _kind, name in date_entries(dates_text):
        day = next_occurrence(date_text, today)
        if day is None:
            continue
        delta = (day - today).days
        if delta < 0 or delta <= 3:
            candidates.append((delta, day, name))
    if candidates:
        candidates.sort(key=lambda c: c[0])
        delta, day, name = candidates[0]
        head = name.lower() if name.lower().startswith("the ") else f"the {name.lower()}"
        if delta < 0:
            return f"I would start with {head}: it is overdue."
        return f"I would start with {head}: it is due {_spoken_date_short(day, today)}."
    open_items = [text for done, text in list_parse(lists_text).get("To-do", []) if not done]
    if open_items:
        return f"Next on your list: {open_items[0]}."
    return "You are all caught up."


def meeting_brief(calendar: str, lists_text: str, facts_text: str, title: str, now) -> str:
    """Brief on a calendar entry: when it is, related facts, and related open to-dos."""
    target = " ".join(title.lower().strip().split())
    for prefix in ("my ", "the "):
        if target.startswith(prefix):
            target = target[len(prefix):]
            break
    for suffix in (" appointment", " meeting"):
        if target.endswith(suffix):
            target = target[: -len(suffix)]
            break
    found = None
    for day, time, entry_title in cal_entries(calendar):
        low = entry_title.lower()
        if low == target or (target and target in low):
            found = (day, time, entry_title)
            break
    if found is None:
        return "I can't find that on your calendar."
    day, time, entry_title = found
    today = now.date()
    word = _day_word(day, today)
    word = word[0].lower() + word[1:]
    if time:
        out = [f"{entry_title} is {word} at {spoken_time(time)}."]
    else:
        out = [f"{entry_title} is {word}."]
    related = search(fact_list(facts_text), entry_title)
    if related:
        out.append(f"You told me: {related[0]}.")
    title_words = {w.strip(".,!?;:'\"") for w in entry_title.lower().split() if w.strip(".,!?;:'\"")}
    open_items = [text for done, text in list_parse(lists_text).get("To-do", []) if not done]
    shared = [text for text in open_items if any(w in text.lower() for w in title_words)]
    if shared:
        out.append(f"Still open: {_join(shared[:3])}.")
    return " ".join(out)
