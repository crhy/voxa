"""Routines: parse_create table, match table, store persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxa.agent.routines import Routine, RoutineStore, parse_create, parse_delete, parse_list

CREATE_ROWS: list[tuple[str, tuple[str, list[str]] | None]] = [
    ("when I say coffee time, open browser and play music", ("coffee time", ["open browser", "play music"])),
    ("when I say good morning, open the curtains and play music", ("good morning", ["open the curtains", "play music"])),
    ("when I say bedtime, close the lights then play jazz", ("bedtime", ["close the lights", "play jazz"])),
    ("when I say movie night, open netflix and then play the trailer for dune",
     ("movie night", ["open netflix", "play the trailer for dune"])),
    ("create a routine called morning that open browser and play music", ("morning", ["open browser", "play music"])),
    ("create a routine called shutdown that close everything and turn off the lights",
     ("shutdown", ["close everything", "turn off the lights"])),
    ("when I say hello, type hello and send it", ("hello", ["type hello", "send it"])),
    ("when I say quiet, mute the volume", ("quiet", ["mute the volume"])),
    ("when I say party, play music and open the lights and play videos",
     ("party", ["play music", "open the lights", "play videos"])),
    ("when I say news, play the news then open browser", ("news", ["play the news", "open browser"])),
    ("when I say relax, play some jazz music", ("relax", ["play some jazz music"])),
    ("when I say work, open browser, play music", ("work", ["open browser", "play music"])),
    ("what is the weather", None),
    ("play music", None),
    ("open browser", None),
]


@pytest.mark.parametrize("text,expected", CREATE_ROWS)
def test_parse_create_table(text: str, expected: tuple[str, list[str]] | None) -> None:
    assert parse_create(text) == expected


def test_parse_create_splits_only_on_command_verbs() -> None:
    # "and videos" is not a command, so it stays glued to the previous step.
    assert parse_create("when I say go, play music and videos") == ("go", ["play music and videos"])


LIST_ROWS = ["list my routines", "show routines", "list the routines", "list routines"]


@pytest.mark.parametrize("text", LIST_ROWS)
def test_parse_list_true(text: str) -> None:
    assert parse_list(text) is True


def test_parse_list_false_for_other() -> None:
    assert parse_list("open browser") is False


def test_parse_delete_name() -> None:
    assert parse_delete("delete the coffee time routine") == "coffee time"
    assert parse_delete("remove morning routine") == "morning"
    assert parse_delete("open browser") is None


def test_match_by_name_and_phrase() -> None:
    store = RoutineStore(Path("/nonexistent/routines.json"))
    store.add(Routine(name="coffee time", phrases=("coffee time",), steps=("open browser", "play music")))
    assert store.match("coffee time") is not None
    assert store.match("run coffee time") is not None
    assert store.match("start coffee time") is not None
    assert store.match("do coffee time") is not None
    assert store.match("it's coffee time") is not None
    assert store.match("time for coffee time") is not None
    assert store.match("coffee") is None
    assert store.match("open browser") is None


def test_store_persistence(tmp_path: Path) -> None:
    path = tmp_path / "routines.json"
    store = RoutineStore(path)
    store.add(Routine(name="morning", phrases=("morning",), steps=("open browser", "play music")))
    assert store.all() == [Routine(name="morning", phrases=("morning",), steps=("open browser", "play music"))]
    reloaded = RoutineStore(path)
    assert reloaded.all() == store.all()
    assert reloaded.match("morning") is not None
    assert reloaded.remove("morning") is True
    assert reloaded.remove("nope") is False
    assert reloaded.all() == []
