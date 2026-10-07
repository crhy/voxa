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
from voxa.config import Settings  # noqa: E402
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


class ConfigStoreRecorder:
    def __init__(self) -> None:
        self.saved = []

    def save(self, settings) -> None:
        self.saved.append(settings)


class ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, **_extra):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self) -> None:
        self.target(*self.args, **self.kwargs)


def make_ns() -> types.SimpleNamespace:
    model = AssistantModel()
    model.set_state(AssistantState.READY)
    ns = types.SimpleNamespace()
    ns.settings = Settings()
    ns.config_store = ConfigStoreRecorder()
    ns.assistant_model = model
    ns.conversation = ConversationRecorder()
    ns._mail_flow = None
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


def test_mail_flow_full(monkeypatch):
    composed = []
    monkeypatch.setattr(window_module.mail, "compose", lambda to, subject, body: composed.append((to, subject, body)))
    monkeypatch.setattr(window_module.threading, "Thread", ImmediateThread)

    ns = make_ns()
    assert ns._mail_flow is None
    assert ns.assistant_model.state is AssistantState.READY

    MainWindow._start_mail_flow(ns, "my mom", "send an email to my mom")
    assert ns._mail_flow is not None
    assert ns.assistant_model.state is AssistantState.DICTATING
    assert ns.assistant_model.detail == "Email address"
    assert ns.conversation.held == 1
    assert ns.finished[0].ok
    assert "email address" in ns.finished[0].speech

    MainWindow._feed_mail_flow(ns, "mom at example dot com")
    assert ns.assistant_model.state is AssistantState.DICTATING
    assert ns.assistant_model.detail == "Email subject"
    assert ns.settings.contacts == {"mom": "mom@example.com"}
    assert len(ns.config_store.saved) == 1

    MainWindow._feed_mail_flow(ns, "Sunday lunch")
    assert ns.assistant_model.state is AssistantState.DICTATING
    assert ns.assistant_model.detail == "Email body"

    MainWindow._feed_mail_flow(ns, "are you free on sunday")
    assert ns.assistant_model.state is AssistantState.DICTATING
    assert ns.assistant_model.detail == "Email body"

    MainWindow._feed_mail_flow(ns, "stop dictation")
    assert composed == [("mom@example.com", "Sunday lunch", "Are you free on sunday.")]
    assert ns._mail_flow is None
    assert ns.assistant_model.state is AssistantState.READY
    assert ns.conversation.released == 1


def test_mail_flow_cancel_at_subject(monkeypatch):
    composed = []
    monkeypatch.setattr(window_module.mail, "compose", lambda to, subject, body: composed.append((to, subject, body)))
    monkeypatch.setattr(window_module.threading, "Thread", ImmediateThread)

    ns = make_ns()
    MainWindow._start_mail_flow(ns, "my mom", "send an email to my mom")
    MainWindow._feed_mail_flow(ns, "mom at example dot com")
    MainWindow._feed_mail_flow(ns, "cancel")
    assert composed == []
    assert ns._mail_flow is None
    assert ns.assistant_model.state is AssistantState.READY
    assert ns.conversation.released == 1


def test_contacts_default_empty_and_not_shared():
    assert Settings().contacts == {}
    a = Settings()
    b = Settings()
    a.contacts["mom"] = "mom@example.com"
    assert b.contacts == {}
