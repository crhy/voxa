from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

Adw.init()

import types  # noqa: E402

from voxa import window as window_module  # noqa: E402
from voxa.ui.state import AssistantModel, AssistantState  # noqa: E402

MainWindow = window_module.MainWindow


class ConversationRecorder:
    def __init__(self) -> None:
        self.held = 0
        self.released = 0

    def hold_prompt(self) -> None:
        self.held += 1

    def release_prompt(self) -> None:
        self.released += 1


def make_ns() -> types.SimpleNamespace:
    model = AssistantModel()
    model.set_state(AssistantState.READY)
    ns = types.SimpleNamespace()
    ns._issue_flow = None
    ns.assistant_model = model
    ns.conversation = ConversationRecorder()
    ns.finished = []
    ns.logged = []

    def _on_tool_finished(result):
        ns.finished.append(result)
        return False

    def _log_action(heard, route, **kwargs):
        ns.logged.append((heard, route, kwargs))

    ns._on_tool_finished = _on_tool_finished
    ns._log_action = _log_action
    return ns


def test_issue_flow_full(monkeypatch):
    spawns = []

    def spawn_rec(command):
        spawns.append(command)

    monkeypatch.setattr(window_module.host, "spawn", spawn_rec)

    ns = make_ns()
    assert ns._issue_flow is None
    assert ns.assistant_model.state is AssistantState.READY

    MainWindow._begin_issue_flow(ns, "voxa", "crhy/voxa")
    assert ns._issue_flow is not None
    assert ns.assistant_model.state is AssistantState.DICTATING
    assert ns.assistant_model.detail == "Issue title"
    assert ns.conversation.held == 1

    MainWindow._feed_issue_flow(ns, "Head is too small")
    assert ns.assistant_model.state is AssistantState.DICTATING
    assert ns.assistant_model.detail == "Issue description"

    MainWindow._feed_issue_flow(ns, "The header is too small")
    assert ns.assistant_model.state is AssistantState.DICTATING
    assert ns.assistant_model.detail == "Issue description"

    MainWindow._feed_issue_flow(ns, "stop dictation")
    assert ns._issue_flow is None
    assert ns.assistant_model.state is AssistantState.READY
    assert ns.conversation.released == 1

    assert len(spawns) == 1
    assert spawns[0][0] == "xdg-open"
    assert spawns[0][1].startswith("https://github.com/crhy/voxa/issues/new?title=Head%20is%20too%20small")


def test_issue_flow_no_repo(monkeypatch):
    spawns = []
    monkeypatch.setattr(window_module.host, "spawn", lambda command: spawns.append(command))

    ns = make_ns()
    MainWindow._begin_issue_flow(ns, "nope", None)
    assert ns._issue_flow is None
    assert ns.assistant_model.state is AssistantState.READY
    assert ns.conversation.held == 0
    assert spawns == []
    assert len(ns.finished) == 1
    assert not ns.finished[0].ok


def test_issue_flow_cancel_at_title(monkeypatch):
    spawns = []
    monkeypatch.setattr(window_module.host, "spawn", lambda command: spawns.append(command))

    ns = make_ns()
    MainWindow._begin_issue_flow(ns, "voxa", "crhy/voxa")
    MainWindow._feed_issue_flow(ns, "cancel")
    assert ns._issue_flow is None
    assert ns.assistant_model.state is AssistantState.READY
    assert ns.conversation.released == 1
    assert spawns == []
