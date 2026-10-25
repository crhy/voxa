"""Capability 96 (prepare meeting briefings): the brief says the day in words, and
cancel works on a calendar that holds exactly one dentist line."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from voxa.agent.headsup import meeting_brief
from voxa.agent.hearing import normalize
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools.calendar_local import _calendar_add_handler, _calendar_cancel_handler

SAMPLES = Path(__file__).resolve().parent.parent / "voxatest" / "data" / "samples"


def _freeze_clock(monkeypatch) -> None:
    class _DateShim:
        today = staticmethod(lambda: dt.date(2026, 10, 10))

    def _datetime_shim(*args):
        return dt.datetime(*args)

    _datetime_shim.now = lambda: dt.datetime(2026, 10, 10, 9, 0)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.date", _DateShim)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.datetime", _datetime_shim)


def test_cap96_heard_phrases_route() -> None:
    heard = [
        "Brief me on my dentist appointment",
        "Hey, tell me about my dentist appointment",
        "can you brief me on my dentist appointment please",
    ]
    for text in heard:
        assert route(normalize(text)) == ToolCall("meeting_brief", {"title": "dentist"})
    assert route(normalize("Add the dentist to my calendar for Friday at 3 pm")) == ToolCall(
        "calendar_add", {"title": "the dentist", "when": "Friday at 3 pm"}
    )
    assert route(normalize("Cancel my dentist appointment")) == ToolCall(
        "calendar_cancel", {"title": "dentist"}
    )


def test_meeting_brief_names_the_day_in_words() -> None:
    now = dt.datetime(2026, 10, 10, 9, 0)
    result = meeting_brief("- 2026-10-11 15:00 | Dentist\n", "", "", "dentist", now)
    assert result == "Dentist is tomorrow at 3 PM."


def test_cap96_chain_add_then_cancel(monkeypatch, tmp_path) -> None:
    from voxa.agent.reminders import ReminderStore

    _freeze_clock(monkeypatch)
    monkeypatch.setattr("voxa.agent.tools.calendar_local.ReminderStore", lambda: ReminderStore(tmp_path / "r.json"))
    store = {"content": SAMPLES.joinpath("calendar-clean-sample.md").read_text(encoding="utf-8")}
    monkeypatch.setattr("voxa.agent.tools.calendar_local._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.calendar_local._write",
        lambda content: store.__setitem__("content", content),
    )

    added = _calendar_add_handler({"title": "the dentist", "when": "tomorrow at 3 pm"})
    assert added.ok and "dentist" in added.speech.casefold()
    assert "dentist" in store["content"].casefold()

    cancelled = _calendar_cancel_handler({"title": "dentist"})
    assert cancelled.ok and cancelled.speech == "I took The dentist off your calendar."
    assert "entist" not in store["content"]
    assert "Gym" in store["content"]


def test_cap96_cases_reset_the_calendar_and_widen_the_day_word() -> None:
    data = json.loads(Path("voxatest/data/test_cases_cap.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in data["cases"] if case["capability"] == 96}
    assert set(cases) == {"cap96-01", "cap96-02", "cap96-02-p1", "cap96-02-p2", "cap96-03"}
    for case in cases.values():
        steps = case.get("setup", [])
        if case["id"] in ("cap96-01", "cap96-03"):
            assert steps and steps[0] == {"remove": "~/Documents/Voxa Calendar.md"}
            assert case["vm_only"] is True
            sample = Path("voxatest/data") / str(steps[1]["copy"])
            assert steps[1]["to"] == "~/Documents/Voxa Calendar.md"
            assert sample.is_file()
    for case_id in ("cap96-02", "cap96-02-p1", "cap96-02-p2"):
        assert cases[case_id]["expected_keywords"] == ["friday|tomorrow|today"]
    assert "dentist" not in SAMPLES.joinpath("calendar-clean-sample.md").read_text(encoding="utf-8").casefold()
    assert "dentist" in SAMPLES.joinpath("calendar-dentist-sample.md").read_text(encoding="utf-8").casefold()
