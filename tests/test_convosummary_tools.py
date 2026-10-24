"""Summarising the current conversation routes and reads back decisions and action items."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import convosummary as cs
from voxa.agent.tools import default_registry


def test_conversation_phrases_route() -> None:
    assert route("summarize our conversation") == ToolCall("conversation_summary", {})
    assert route("what did we decide") == ToolCall("conversation_summary", {})
    assert route("what are my action items from this conversation") == ToolCall("conversation_summary", {})


def test_summary_rules_do_not_steal_ordinary_questions() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})


def test_registry_has_conversation_summary() -> None:
    assert "conversation_summary" in default_registry().names()


def test_short_conversation_says_not_much(monkeypatch) -> None:
    monkeypatch.setattr(cs, "get_history", lambda: [{"role": "user", "content": "what is the capital of Italy"}])
    result = cs._conversation_summary_handler({})
    assert result.ok and result.speech == "We haven't talked about much yet."


def test_summary_reads_back_decisions(monkeypatch) -> None:
    history = [
        {"role": "user", "content": "should we use xvfb for tests"},
        {"role": "assistant", "content": "yes, use xvfb"},
        {"role": "user", "content": "summarize our conversation"},
    ]
    monkeypatch.setattr(cs, "get_history", lambda: history)
    monkeypatch.setattr(
        cs,
        "ask_model",
        lambda messages: "Decision: use xvfb for tests\nTo do: add the summary tool",
    )
    result = cs._conversation_summary_handler({})
    assert result.ok
    assert result.speech == "We decided Use xvfb for tests. To do: Add the summary tool."


def test_summary_ignores_extra_chatter(monkeypatch) -> None:
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    monkeypatch.setattr(cs, "get_history", lambda: history)
    monkeypatch.setattr(
        cs,
        "ask_model",
        lambda messages: (
            "Sure, here is a summary:\n"
            "- Decision: ship it\n"
            "1. To do: write tests\n"
            "random chatter here\n"
            "Decision: ship it"
        ),
    )
    result = cs._conversation_summary_handler({})
    assert result.ok
    assert result.speech == "We decided Ship it. To do: Write tests."


def test_summary_empty_brain_reply_fails(monkeypatch) -> None:
    history = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    monkeypatch.setattr(cs, "get_history", lambda: history)
    monkeypatch.setattr(cs, "ask_model", lambda messages: "")
    result = cs._conversation_summary_handler({})
    assert not result.ok


def test_summary_without_brain_fails(monkeypatch) -> None:
    history = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    monkeypatch.setattr(cs, "get_history", lambda: history)
    monkeypatch.setattr(cs, "ask_model", None)
    result = cs._conversation_summary_handler({})
    assert not result.ok and result.speech == "I can't reach my brain right now."
