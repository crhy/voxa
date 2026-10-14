"""Voice commands for project and model status route and work end to end."""

from __future__ import annotations

import subprocess

from voxa.agent.devstatus import parse_git_status, parse_task_line, project_dir, spoken_git, spoken_model
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.devstatus import _model_status_handler, _project_status_handler


def test_project_dir_matches_spoken_name() -> None:
    candidates = ["/Voxa", "/home/me/voxa", "/home/me/other"]
    assert project_dir("the Voxa project", "/home/me", candidates) == "/home/me/voxa"
    assert project_dir("voxa", "/home/me", candidates) == "/home/me/voxa"
    assert project_dir("other", "/home/me", candidates) == "/home/me/other"
    assert project_dir("missing", "/home/me", candidates) is None


def test_parse_git_status_reads_branch_counts() -> None:
    text = "## main...origin/main [ahead 3, behind 2]\nM voxa/agent/lists.py\n"
    assert parse_git_status(text) == ("main", 1, 3, 2)
    assert parse_git_status("## main\n") == ("main", 0, 0, 0)


def test_spoken_git_phrases() -> None:
    assert (
        spoken_git("Voxa", "main", 12, 3, 0, "Fix the radio list")
        == "Voxa is on main with 12 changed files, 3 commits not pushed. The last commit was: Fix the radio list."
    )
    assert spoken_git("Voxa", "main", 0, 0, 0, "") == "Voxa is on main and clean."
    assert (
        spoken_git("Voxa", "main", 1, 0, 2, "")
        == "Voxa is on main with 1 changed file and 2 behind."
    )


def test_parse_task_line_classifies() -> None:
    assert parse_task_line("22:53:39 c56-recurring OK (131 paths changed)") == ("passed", "c56-recurring")
    assert parse_task_line("22:53:39 c56-recurring did not pass") == ("failed", "c56-recurring")
    assert parse_task_line("START Voxa agent c89-dev -> work") == ("working", "c89-dev")
    assert parse_task_line("") is None


def test_spoken_model_phrases() -> None:
    health = '{"status":"ok","model":"qwen3.8-flash-next-coder-iq1_m"}'
    assert (
        spoken_model(health, "22:53:39 c56-recurring OK (131 paths changed)")
        == "The local model is running: qwen3.8-flash-next-coder-iq1_m. Its last task, c56-recurring, passed."
    )
    assert spoken_model(None, None) == "The local model is not running."
    assert (
        spoken_model(None, "START Voxa agent c89-dev -> work")
        == "The local model is not running. It is working on c89-dev."
    )


def test_project_and_model_phrases_route() -> None:
    assert route("what is the git status of the Voxa project") == ToolCall("project_status", {"name": "Voxa project"})
    assert route("what's the status of the voxa repo") == ToolCall("project_status", {"name": "voxa"})
    assert route("how is the voxa project doing") == ToolCall("project_status", {"name": "voxa"})
    assert route("is the local model running") == ToolCall("model_status", {})
    assert route("what is strata doing") == ToolCall("model_status", {})
    assert route("local model status") == ToolCall("model_status", {})


def test_guard_phrases_stay_put() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})
    assert route("how is my computer doing") == ToolCall("system_health", {})


def test_project_handler_end_to_end(monkeypatch) -> None:
    def fake(command: list[str]) -> str:
        if command[:2] == ["sh", "-c"]:
            return "/Voxa/.git\n" if "ls -d" in command[2] else "/home/me\n"
        if command[0] == "git":
            return "## main...origin/main [ahead 3, behind 2]\nM voxa/agent/lists.py\n" if "status" in command else "Fix the radio list\n"
        return ""

    monkeypatch.setattr(
        "voxa.agent.tools.devstatus._run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, fake(command), ""),
    )
    result = _project_status_handler({"name": "Voxa"})
    assert result.ok
    assert result.speech == (
        "Voxa is on main with 1 changed file, 3 commits not pushed and 2 behind. "
        "The last commit was: Fix the radio list."
    )


def test_project_handler_reports_missing_project(monkeypatch) -> None:
    def fake(command: list[str]) -> str:
        if command[:2] == ["sh", "-c"]:
            return "" if "ls -d" in command[2] else "/home/me\n"
        return ""

    monkeypatch.setattr(
        "voxa.agent.tools.devstatus._run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, fake(command), ""),
    )
    result = _project_status_handler({"name": "nope"})
    assert not result.ok
    assert result.speech == "I can't find a project called nope."


def test_model_handler_end_to_end(monkeypatch) -> None:
    def fake(command: list[str]) -> str:
        if command[0] == "curl":
            return '{"status":"ok","model":"qwen3.8-flash-next-coder-iq1_m"}'
        if command[0] == "tail":
            return "06:08:48 c92-headsup OK (144 paths changed)"
        return ""

    monkeypatch.setattr(
        "voxa.agent.tools.devstatus._run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, fake(command), ""),
    )
    result = _model_status_handler({})
    assert result.ok
    assert result.speech == (
        "The local model is running: qwen3.8-flash-next-coder-iq1_m. "
        "Its last task, c92-headsup, passed."
    )


def test_registry_has_devstatus_tools() -> None:
    names = default_registry().names()
    assert "project_status" in names
    assert "model_status" in names
