"""Capability 89: project status survives the "git"->"get" mishearing, a missing project
name, and a fresh machine that has no code project until the case seeds one."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools.devstatus import _project_status_handler
from voxatest.setup import run_setup


def test_heard_phrases_route_to_project_status() -> None:
    assert route("what is the get status of the project") == ToolCall("project_status", {"name": ""})
    assert route("what is the status of the Voxa repo") == ToolCall("project_status", {"name": "Voxa"})
    assert route("how is the Voxa project doing") == ToolCall("project_status", {"name": "Voxa"})
    assert route("what is the git status of the Voxa project") == ToolCall("project_status", {"name": "Voxa project"})


def test_bare_status_phrase_routes_to_project_status() -> None:
    assert route("what's the status") == ToolCall("project_status", {"name": ""})


def test_other_status_questions_are_not_stolen() -> None:
    assert route("what is the status of the meeting") is None or route(
        "what is the status of the meeting"
    ).tool != "project_status"
    assert route("what is the capital of France") is None
    assert route("how is my computer doing") == ToolCall("system_health", {})


def test_handler_names_the_project_when_none_is_spoken(monkeypatch) -> None:
    def fake(command: list[str]) -> str:
        if command[:2] == ["sh", "-c"]:
            return "/home/me/voxa/.git\n" if "ls -d" in command[2] else "/home/me\n"
        if command[0] == "git":
            return "## main...origin/main [ahead 3, behind 2]\nM voxa/agent/lists.py\n"
        return ""

    monkeypatch.setattr(
        "voxa.agent.tools.devstatus._run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, fake(command), ""),
    )
    result = _project_status_handler({"name": ""})
    assert result.ok
    assert result.speech.startswith("Voxa is on main")


def test_handler_says_no_project_when_there_is_none(monkeypatch) -> None:
    monkeypatch.setattr(
        "voxa.agent.tools.devstatus._run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, "", ""),
    )
    result = _project_status_handler({"name": ""})
    assert not result.ok
    assert result.speech == "I can't find a project on this computer."


def test_run_setup_seeds_a_git_repo() -> None:
    calls: list[list[str]] = []
    error = run_setup(
        [{"gitrepo": "~/voxa"}],
        data_dir=Path("."),
        host=lambda cmd, *, stdin=None: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, b"", b""),
    )
    assert error == ""
    assert len(calls) == 1
    script = calls[0][-1]
    assert "git init" in script
    assert "$HOME/voxa" in script
    assert "commit" in script


def test_run_setup_recursive_remove() -> None:
    calls: list[list[str]] = []
    error = run_setup(
        [{"remove": "~/voxa", "recursive": True}],
        data_dir=Path("."),
        host=lambda cmd, *, stdin=None: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, b"", b""),
    )
    assert error == ""
    assert "rm -f -r" in calls[0][-1]
    assert "$HOME/voxa" in calls[0][-1]


def test_shipped_cap89_cases_seed_a_project() -> None:
    data = json.loads(Path("voxatest/data/test_cases_cap.json").read_text())
    cases = [c for c in data["cases"] if c["id"].startswith("cap89")]
    assert {c["id"] for c in cases} == {"cap89-01", "cap89-01-p1", "cap89-01-p2"}
    for case in cases:
        assert case["expected_tool"] == "project_status"
        assert {"gitrepo": "~/voxa"} in case["setup"]
        assert {"remove": "~/voxa", "recursive": True} in case["setup"]
        assert case["vm_only"] is True
