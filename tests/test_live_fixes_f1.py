from __future__ import annotations

from datetime import date

from voxa.agent.calendar_local import remove
from voxa.agent.dates import parse_day
from voxa.agent.hearing import normalize
from voxa.agent.intents import route


def test_normalize_time() -> None:
    assert normalize("schedule the dentist for Friday at 3 p.m").endswith("3 pm")
    assert route("schedule the dentist for Friday at 3 p.m").tool == "calendar_add"
    assert route("what time is it in Tokyo").tool == "world_time"


def test_remove_fuzzy() -> None:
    content = "- 2026-10-12 12:00 | Lunch with Sam\n- 2026-10-13 --:-- | Dentist\n"
    new, removed, _, _ = remove(content, "lunch with sound")
    assert removed == "Lunch with Sam"
    assert "Lunch with Sam" not in new


def test_remove_strip() -> None:
    content = "- 2026-10-12 12:00 | Lunch with Sam\n- 2026-10-13 --:-- | Dentist\n"
    new, removed, _, _ = remove(content, "my lunch with sound appointment")
    assert removed == "Lunch with Sam"
    assert "Lunch with Sam" not in new


def test_parse_day_ordinals() -> None:
    today = date(2026, 10, 10)
    assert parse_day("May fifth", today) == (None, 5, 5)
    assert parse_day("the fifth of May", today) == (None, 5, 5)
    assert parse_day("May twenty-first", today) == (None, 5, 21)
    assert parse_day("the twenty first of May", today) == (None, 5, 21)
    assert parse_day("the first", today) == (2026, 11, 1)
    assert parse_day("the 15th", today) == (2026, 10, 15)
