"""Scheduled routines: the parse table, the due table, persistence, and a run that keeps its order.

crhy/voxa#67's acceptance line: "a phrase-triggered routine and a scheduled one both run their
steps in order in tests."
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from voxa.agent.routines import (
    Routine,
    RoutineStore,
    list_reply,
    parse_schedule,
    parse_unschedule,
    schedule_reply,
    schedule_text,
    unschedule_reply,
)

# A Friday and the next day (a Saturday) of the same week, for the tables below.
FRIDAY = datetime(2026, 11, 13, 17, 0)
SATURDAY = datetime(2026, 11, 14, 17, 0)

SCHEDULE_ROWS: list[tuple[str, tuple[str, int | None, tuple[int, int] | None] | None]] = [
    # The issue's own example line, and the phrasings the task names.
    ("every friday at five, send me my weekly spending", ("send me my weekly spending", 4, (5, 0))),
    ("Voxa, every friday at five, send me my weekly spending", ("send me my weekly spending", 4, (5, 0))),
    ("schedule the movie time routine for friday at seven", ("movie time", 4, (7, 0))),
    ("put the morning routine on thursday at 4pm", ("morning", 3, (16, 0))),
    # HH:MM, a bare hour, spoken hours, and a weekday with no time at all.
    ("schedule the movie time routine for friday at 17:30", ("movie time", 4, (17, 30))),
    ("schedule the movie time routine for friday", ("movie time", 4, None)),
    ("every friday, send me my weekly spending", ("send me my weekly spending", 4, None)),
    ("put my shutdown routine on sunday at midnight", ("shutdown", 6, (0, 0))),
    ("every monday at noon, open the curtains", ("open the curtains", 0, (12, 0))),
    ("reschedule the news routine every saturday at 7", ("news", 5, (7, 0))),
    # Not scheduling phrasings.
    ("open browser", None),
    ("play music", None),
    ("list my routines", None),
    ("delete the movie time routine", None),
    ("when I say coffee time, open browser and play music", None),
    ("set a timer for five minutes", None),
    ("remind me at three to call bob", None),
    ("schedule the movie time routine for tomorrow", None),
    ("schedule the movie time routine for friday at teatime", None),
]


@pytest.mark.parametrize("text,expected", SCHEDULE_ROWS)
def test_parse_schedule_table(text: str, expected: tuple[str, int | None, tuple[int, int] | None] | None) -> None:
    assert parse_schedule(text) == expected


UNSCHEDULE_ROWS: list[tuple[str, str | None]] = [
    ("unschedule the movie time routine", "movie time"),
    ("stop the movie time routine running every friday", "movie time"),
    ("unschedule morning", "morning"),
    ("cancel the bedtime routine running every saturday at 11", "bedtime"),
    ("open browser", None),
    ("delete the movie time routine", None),
    ("stop the coffee time routine", None),
]


@pytest.mark.parametrize("text,expected", UNSCHEDULE_ROWS)
def test_parse_unschedule_table(text: str, expected: str | None) -> None:
    assert parse_unschedule(text) == expected


def _store(path: Path) -> RoutineStore:
    return RoutineStore(path)


DUE_ROWS: list[tuple[Routine, datetime, bool]] = [
    (Routine("movie time", ("movie time",), ("open browser",), weekday=4, at=(17, 0)), FRIDAY, True),
    (Routine("movie time", ("movie time",), ("open browser",), weekday=4, at=(17, 0)), SATURDAY, False),
    (Routine("movie time", ("movie time",), ("open browser",)), FRIDAY, False),
    (Routine("movie time", ("movie time",), ("open browser",), weekday=4), FRIDAY, True),
    (Routine("movie time", ("movie time",), ("open browser",), weekday=4, at=(17, 0)), FRIDAY.replace(minute=1), True),
    (Routine("movie time", ("movie time",), ("open browser",), weekday=4, at=(17, 0)), FRIDAY.replace(minute=2), False),
    (Routine("movie time", ("movie time",), ("open browser",), weekday=4, at=(17, 0)), FRIDAY.replace(minute=59, hour=16), False),
]


@pytest.mark.parametrize("routine,now,want_due", DUE_ROWS)
def test_due_table(routine: Routine, now: datetime, want_due: bool, tmp_path: Path) -> None:
    store = _store(tmp_path / "routines.json")
    store.add(routine)
    assert [r.name for r in store.due(now)] == ([routine.name] if want_due else [])


def test_due_keeps_the_routine_for_next_week(tmp_path: Path) -> None:
    """Unlike a reminder, a scheduled routine is not consumed: it runs again next week."""
    store = _store(tmp_path / "routines.json")
    store.add(Routine("movie time", ("movie time",), ("open browser",), weekday=4, at=(17, 0)))
    assert len(store.due(FRIDAY)) == 1
    assert len(store.due(FRIDAY.replace(second=5))) == 1
    assert len(store.all()) == 1


def test_due_orders_by_time_then_name(tmp_path: Path) -> None:
    store = _store(tmp_path / "routines.json")
    store.add(Routine("late one", ("late one",), ("open browser",), weekday=4, at=(17, 0)))
    store.add(Routine("early one", ("early one",), ("open browser",), weekday=4, at=(16, 59)))
    store.add(Routine("any time", ("any time",), ("open browser",), weekday=4))
    # 16:59 and 17:00 both say now at 17:00; a routine with no time is due all day and sorts first.
    assert [r.name for r in store.due(FRIDAY)] == ["any time", "early one", "late one"]


def test_schedule_and_unschedule_persist(tmp_path: Path) -> None:
    path = tmp_path / "routines.json"
    store = _store(path)
    store.add(Routine("movie time", ("movie time",), ("open browser", "play the trailer")))
    assert store.set_schedule("movie time", 4, (17, 0)) is True
    assert store.set_schedule("nope", 4, (17, 0)) is False
    reloaded = _store(path)
    assert reloaded.all() == [
        Routine("movie time", ("movie time",), ("open browser", "play the trailer"), weekday=4, at=(17, 0))
    ]
    assert reloaded.clear_schedule("movie time") is True
    assert reloaded.clear_schedule("movie time") is False
    assert _store(path).all() == [Routine("movie time", ("movie time",), ("open browser", "play the trailer"))]


def test_old_routines_json_still_loads(tmp_path: Path) -> None:
    """A file written by the earlier Voxa has no schedule keys, and junk is ignored."""
    path = tmp_path / "routines.json"
    path.write_text(
        '[{"name": "coffee time", "phrases": ["coffee time"], "steps": ["open browser"]}, '
        '{"name": "bad", "phrases": [], "steps": [], "weekday": 9, "at": [99, 99]}]',
        encoding="utf-8",
    )
    store = _store(path)
    assert store.all() == [
        Routine("coffee time", ("coffee time",), ("open browser",)),
        Routine("bad", (), ()),
    ]
    assert store.due(FRIDAY) == []


def test_find_matches_the_spoken_routine(tmp_path: Path) -> None:
    store = _store(tmp_path / "routines.json")
    store.add(Routine("weekly spending", ("weekly spending",), ("open browser",)))
    store.add(Routine("movie time", ("movie time",), ("open browser",)))
    assert store.find("weekly spending").name == "weekly spending"
    assert store.find("send me my weekly spending").name == "weekly spending"
    assert store.find("the movie time routine").name == "movie time"
    assert store.find("open browser") is None


def test_replies_name_the_schedule() -> None:
    assert schedule_reply("movie time", 4, (17, 0)) == "Scheduled the movie time routine for every friday at 17:00."
    assert schedule_reply("movie time", 4, None) == "Scheduled the movie time routine for every friday."
    assert unschedule_reply("movie time", True) == "Stopped the movie time routine from running on its schedule."
    assert unschedule_reply("movie time", False) == "The movie time routine has no schedule."
    assert schedule_text(Routine("movie time", (), (), weekday=4, at=(17, 0))) == "every friday 17:00"
    assert schedule_text(Routine("movie time", (), (), weekday=4)) == "every friday"
    assert schedule_text(Routine("movie time", (), ())) == ""


def test_list_reply_shows_the_schedule(tmp_path: Path) -> None:
    store = _store(tmp_path / "routines.json")
    store.add(Routine("movie time", ("movie time",), ("open browser",), weekday=4, at=(17, 0)))
    store.add(Routine("weekly spending", ("weekly spending",), ("open browser",)))
    assert list_reply(store.all()) == "Routines: movie time \u2014 every friday 17:00, weekly spending"
    # Plain names still work, as test_routines_spoken.py expects.
    assert list_reply(["good morning", "wind down"]) == "Routines: good morning, wind down"


class StepRecorder:
    """A stand-in for the window: it speaks the routine's name once, then runs each step
    through the real router, one at a time, in order."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.steps: list[str] = []
        self.tools: list[str] = []

    def run(self, routine: Routine) -> None:
        from voxa.agent.intents import route

        self.spoken.append(routine.name)
        queue = list(routine.steps)
        while queue:
            step = queue.pop(0)
            self.steps.append(step)
            call = route(step)
            if call is not None:
                self.tools.append(call.tool)


def test_phrase_and_scheduled_runs_both_keep_the_order(tmp_path: Path) -> None:
    """The acceptance line of crhy/voxa#67: both triggers run their steps in order."""
    store = _store(tmp_path / "routines.json")
    store.add(
        Routine(
            name="weekly spending",
            phrases=("weekly spending",),
            steps=("open browser", "play the news", "type hello"),
            weekday=4,
            at=(17, 0),
        )
    )
    store.add(Routine(name="coffee time", phrases=("coffee time",), steps=("open browser", "type hello")))

    by_phrase = StepRecorder()
    matched = store.match("run coffee time")
    assert matched is not None
    by_phrase.run(matched)
    assert by_phrase.steps == ["open browser", "type hello"]

    by_clock = StepRecorder()
    for routine in store.due(FRIDAY):
        by_clock.run(routine)
    assert by_clock.spoken == ["weekly spending"]
    assert by_clock.steps == ["open browser", "play the news", "type hello"]
    assert by_clock.tools == ["open_app", "play_music", "type_text"]


def test_window_runs_due_routines_from_its_clock() -> None:
    """The window's 5-second tick feeds due routines to the same path a spoken trigger uses."""
    source = Path("voxa/window.py").read_text(encoding="utf-8")
    tick = source.split("def _reminders_tick")[1].split("def _run_due_routines")[0]
    assert "self._run_due_routines(now)" in tick
    body = source.split("def _run_due_routines")[1].split("def _start_progress")[0]
    assert "self.routine_store.due(now)" in body
    assert "self._start_routine(routine.steps)" in body
    assert "self._routine_fired" in body
    # A routine fired by the clock must not go back through match().
    assert "match(" not in body
