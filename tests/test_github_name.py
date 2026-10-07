from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools.github import _set_github_owner_handler, clean_owner, github_tools


def test_clean_owner_table():
    assert clean_owner("c r h y") == "crhy"
    assert clean_owner("@Crhy") == "crhy"
    assert clean_owner("my dash name") == "my-name"
    assert clean_owner("  ") == ""
    assert clean_owner("-bad") == ""


def test_set_github_owner_saves_through_hook(monkeypatch):
    saved = []
    monkeypatch.setattr("voxa.agent.tools.github.set_owner", saved.append)
    result = _set_github_owner_handler({"name": "c r h y"})
    assert result.ok
    assert result.speech == "Got it. Your GitHub name is crhy."
    assert saved == ["crhy"]


def test_set_github_owner_without_hook(monkeypatch):
    monkeypatch.setattr("voxa.agent.tools.github.set_owner", None)
    result = _set_github_owner_handler({"name": "crhy"})
    assert not result.ok
    assert result.speech == "I cannot save that here."


def test_set_github_owner_empty_name():
    result = _set_github_owner_handler({"name": "  "})
    assert not result.ok
    assert result.speech == "I did not catch the GitHub name."


def test_router_matches():
    assert route("my github name is c r h y") == ToolCall("set_github_owner", {"name": "c r h y"})
    assert route("my GitHub username is crhy") == ToolCall("set_github_owner", {"name": "crhy"})
    assert route("my get hub account is crhy") == ToolCall("set_github_owner", {"name": "crhy"})


def test_router_open_github_not_owner():
    call = route("open github")
    assert call is None or call.tool != "set_github_owner"


def test_github_tools_registered():
    names = [tool.name for tool in github_tools()]
    assert names == ["set_github_owner"]
