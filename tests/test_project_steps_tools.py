"""A big goal breaks into steps that are saved as a named list."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.projectsteps import _project_steps_handler


def test_project_steps_phrases_route() -> None:
    assert route("break down planting a vegetable garden into steps") == ToolCall(
        "project_steps", {"goal": "planting a vegetable garden"}
    )
    assert route("help me plan moving house") == ToolCall("project_steps", {"goal": "moving house"})
    assert route("what are the steps to learn guitar") == ToolCall("project_steps", {"goal": "learn guitar"})
    assert route("make a plan for moving") == ToolCall("project_steps", {"goal": "moving"})


def test_project_steps_does_not_steal_other_intents() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})
    assert route("plan a trip to paris") != ToolCall("project_steps", {"goal": "a trip to paris"})
    assert route("what's the plan") != ToolCall("project_steps", {"goal": ""})


def test_project_steps_handler_end_to_end(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.projectsteps._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.projectsteps._write",
        lambda content: store.__setitem__("content", content),
    )
    monkeypatch.setattr(
        "voxa.agent.tools.projectsteps.ask_model",
        lambda messages: (
            "1. Buy supplies\n2. Tape the edges\n3. Roll the walls\n"
            "4. Cut in the corners\n5. Remove the tape"
        ),
    )

    result = _project_steps_handler({"goal": "paint the kitchen"})
    assert result.ok
    assert result.speech == (
        "I broke paint the kitchen into 5 steps and saved them as the Paint The Kitchen list. "
        "First: Buy supplies."
    )
    assert "## Paint The Kitchen" in store["content"]
    assert "- [ ] Buy supplies" in store["content"]
    assert "- [ ] Remove the tape" in store["content"]


def test_project_steps_handler_without_brain(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.projectsteps.ask_model", None)
    result = _project_steps_handler({"goal": "learn guitar"})
    assert not result.ok


def test_project_steps_handler_rejects_empty_goal() -> None:
    result = _project_steps_handler({"goal": "  "})
    assert not result.ok


def test_project_steps_handler_rejects_unusable_reply(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.projectsteps.ask_model", lambda messages: "I am not sure.")
    result = _project_steps_handler({"goal": "learn guitar"})
    assert not result.ok


def test_registry_has_project_steps() -> None:
    assert "project_steps" in default_registry().names()
