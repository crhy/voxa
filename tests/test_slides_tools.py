"""presentation_make routes spoken deck requests and writes a real .odp."""

from __future__ import annotations

import zipfile
from pathlib import Path

from voxa import documents
from voxa.agent.intents import ToolCall, route
from voxa.agent.slides import parse_slides, write_odp
from voxa.agent.tools import default_registry
from voxa.agent.tools.slides import _presentation_make_handler


def test_presentation_phrases_route() -> None:
    assert route("make a presentation about solar power") == ToolCall(
        "presentation_make", {"topic": "solar power", "count": ""}
    )
    assert route("create 5 slides about the roman empire") == ToolCall(
        "presentation_make", {"topic": "the roman empire", "count": "5"}
    )
    assert route("make me a slide deck about photosynthesis") == ToolCall(
        "presentation_make", {"topic": "photosynthesis", "count": ""}
    )
    assert route("give me three slides on water cycles") == ToolCall(
        "presentation_make", {"topic": "water cycles", "count": "three"}
    )
    assert route("build a deck about the civil war") == ToolCall(
        "presentation_make", {"topic": "the civil war", "count": ""}
    )


def test_presentation_does_not_steal_other_intents() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})
    assert route("what is the capital of France") is None
    assert route("add milk to my shopping list") == ToolCall(
        "list_add", {"item": "milk", "list": "shopping"}
    )
    assert route("make a plan for my trip to Rome") != ToolCall("presentation_make", {})


def test_parse_slides_clean() -> None:
    text = "1. Solar Power\n- Clean energy\n- Free fuel\n2. Storage\n- Batteries\n- The grid"
    assert parse_slides(text) == [
        ("Solar Power", ["Clean energy", "Free fuel"]),
        ("Storage", ["Batteries", "The grid"]),
    ]


def test_parse_slides_messy_falls_back_to_lines() -> None:
    text = "Solar Power\nWind Power\nTidal Power"
    assert parse_slides(text) == [
        ("Solar Power", []),
        ("Wind Power", []),
        ("Tidal Power", []),
    ]


def test_parse_slides_empty() -> None:
    assert parse_slides("") == []
    assert parse_slides("   \n\n  \n") == []


def test_write_odp_structure(tmp_path: Path) -> None:
    path = tmp_path / "deck.odp"
    write_odp(path, "Solar Power", [("Solar Power", ["Clean", "Free"]), ("Storage", ["Batteries"])])
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert names[0] == "mimetype"
        assert archive.read("mimetype") == b"application/vnd.oasis.opendocument.presentation"
        assert archive.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
        assert "META-INF/manifest.xml" in names
        content = archive.read("content.xml").decode("utf-8")
        assert content.count("<draw:page ") == 2
        assert "Solar Power" in content
        assert "Batteries" in content


def test_presentation_make_handler_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "voxa.agent.tools.slides.ask_model",
        lambda messages: (
            "1. Solar Power\n- Clean energy\n- Free fuel\n"
            "2. Storage\n- Batteries\n- The grid\n"
            "3. Savings\n- Lower bills\n- Fewer emissions"
        ),
    )
    opened: list[Path] = []
    monkeypatch.setattr(documents, "open_in_libreoffice", lambda path: opened.append(path))
    result = _presentation_make_handler({"topic": "solar power", "count": "3"})
    assert result.ok
    assert result.speech == "I made 3 slides about solar power and opened them in LibreOffice."
    assert len(opened) == 1
    saved = opened[0]
    assert saved.parent == tmp_path / "Voxa Drafts"
    assert saved.name == "solar-power slides.odp"
    assert saved.exists()
    with zipfile.ZipFile(saved) as archive:
        assert archive.read("mimetype") == b"application/vnd.oasis.opendocument.presentation"


def test_presentation_make_handler_no_brain(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.slides.ask_model", None)
    result = _presentation_make_handler({"topic": "solar power", "count": ""})
    assert not result.ok
    assert result.speech == "I can't reach my brain right now."


def test_presentation_make_handler_empty_topic() -> None:
    result = _presentation_make_handler({"topic": "   ", "count": ""})
    assert not result.ok


def test_registry_has_presentation_make() -> None:
    assert "presentation_make" in default_registry().names()
