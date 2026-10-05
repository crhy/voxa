from __future__ import annotations

from voxa.endpoint import is_complete, is_followup

COMPLETE = [
    "voxa open gmail",
    "close brutal chess",
    "play some jazz",
    "turn it up",
    "start dictating",
    "cancel",
    "open libra office right",
    "open gmail",
    "go back",
    "read this page",
    "mute",
    "louder",
    "google cats",
    "search the web for weather",
    "type hello there",
    "close the window",
    "switch to gimp",
    "next",
    "never mind",
    "go offline",
    "stop",
    "take dictation",
]

NOT_COMPLETE = [
    "",
    "voxa",
    "turn off the",
    "play some",
    "open",
    "what is the",
    "what time is it",
    "open gmail and then",
    "tell me a joke",
    "close brutal chess and open gmail",
    "how are you",
    "the",
    "a",
    "some",
    "please",
    "then",
    "um",
    "uh",
    "turn up the",
    "play music on",
    "voxa and then",
    "what is the weather",
    "hello there",
    "hey",
    "ok",
]


def test_complete_utterances() -> None:
    for text in COMPLETE:
        assert is_complete(text), text


def test_incomplete_utterances() -> None:
    for text in NOT_COMPLETE:
        assert not is_complete(text), text


def test_rows_meet_the_minimum_table_size() -> None:
    assert len(COMPLETE) + len(NOT_COMPLETE) >= 30


def test_custom_wake_word_alone_is_not_complete() -> None:
    assert not is_complete("computer", wake_word="computer")
    assert not is_complete("voxa", wake_word="voxa")
    assert is_complete("voxa open gmail", wake_word="voxa")


FOLLOWUP_TRUE = [
    "and open gmail",
    "also the window",
    "then close it",
    "now play music",
    "next",
    "again",
    "what about the weather",
    "how about jazz",
    "no not that one",
    "actually open gmail",
    "instead close it",
    "make it louder",
    "the other one",
    "another window",
    "more coffee",
    "less sugar",
    "louder please",
    "quieter now",
    "yes please",
    "yeah ok",
    "okay then",
    "what time is it",
    "where is my file",
    "can you open gmail",
    "is it raining",
    "will it rain",
    "do you know",
    "does this work",
    "are we good",
    "could you help",
    "how are you",
    "who is that",
    "when does it open",
    "why not",
    "tell me a joke?",
]

FOLLOWUP_FALSE = [
    "nice weather today",
    "the cat sat",
    "hello there",
    "see you later",
    "the",
    "a",
    "please",
    "um uh",
    "banana",
    "purple dinosaur",
    "I like coffee",
    "walk the dog",
    "green apple pie",
    "table lamp",
    "desk shade",
    "curtain rod",
    "garden gnome",
    "wooden spoon",
    "marble statue",
    "quiet library",
]


def test_followup_true_rows() -> None:
    for text in FOLLOWUP_TRUE:
        assert is_followup(text), text


def test_followup_false_rows() -> None:
    for text in FOLLOWUP_FALSE:
        assert not is_followup(text), text


def test_followup_table_meets_minimum_size() -> None:
    assert len(FOLLOWUP_TRUE) + len(FOLLOWUP_FALSE) >= 20
