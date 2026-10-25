"""Echo a remembered fact back in the second person."""

from __future__ import annotations

from voxa.agent.memory_facts import echo_fact
from voxa.agent.tools.memory_facts import _remember_fact_handler


def test_echo_second_person_pronouns() -> None:
    assert echo_fact("my cat is named Mittens") == "your cat is named Mittens"
    assert echo_fact("I am allergic to peanuts") == "you're allergic to peanuts"
    assert echo_fact("I'm scared of thunder") == "you're scared of thunder"
    assert echo_fact("I like jazz music") == "you like jazz music"
    assert echo_fact("she reminded me to study") == "she reminded you to study"
    assert echo_fact("a friend of mine runs daily") == "a friend of yours runs daily"


def test_echo_keeps_capitalisation_and_other_words() -> None:
    assert echo_fact("My brother Sam cooks") == "Your brother Sam cooks"
    assert echo_fact("Mittens is mine") == "Mittens is yours"


def test_echo_does_not_split_words() -> None:
    # "me" inside a larger word must stay untouched.
    assert echo_fact("the name is Ahmed") == "the name is Ahmed"
    assert echo_fact("form is important") == "form is important"


def test_echo_caps_long_fact_under_30_words() -> None:
    long_fact = " ".join(f"w{i}" for i in range(60))
    echoed = echo_fact(long_fact)
    assert echoed.endswith("\u2026")
    assert len(echoed.split()) <= 26
    reply = f"I'll remember that {echoed}."
    assert len(reply.split()) < 30


def test_remember_handler_replies_with_fact(monkeypatch) -> None:
    store = {"content": ""}
    monkeypatch.setattr("voxa.agent.tools.memory_facts._read", lambda: store["content"])
    monkeypatch.setattr(
        "voxa.agent.tools.memory_facts._write",
        lambda content: store.__setitem__("content", content),
    )
    result = _remember_fact_handler({"fact": "my cat is named Mittens"})
    assert result.ok
    assert result.speech == "I'll remember that your cat is named Mittens."
    assert "Mittens" in result.speech


def test_remember_handler_empty_fact_unchanged(monkeypatch) -> None:
    monkeypatch.setattr("voxa.agent.tools.memory_facts._read", lambda: "")
    monkeypatch.setattr("voxa.agent.tools.memory_facts._write", lambda content: None)
    result = _remember_fact_handler({"fact": "   "})
    assert not result.ok and result.speech == "What should I remember?"
