"""rhY's log, 13 and 15 October 2026: Whisper wrote "Rye", "Ry" and "Raya" Thornton, and the
spelling he gave ("Rhy is spelled R-H-Y") was ignored or searched for literally."""
from __future__ import annotations

import json

import pytest

from voxa import spellings
from voxa.vocabulary import build_hint


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_standalone_spelling_fixes_and_reruns_the_previous_request():
    lesson = spellings.teach("Rhy is spelled R-H-Y", "tell me about the country singer Rye Thornton")
    assert lesson == ("previous", "tell me about the country singer Rhy Thornton")
    assert spellings.load() == {"Rhy Thornton": ["rye thornton"]}


def test_inline_spelling_is_applied_not_searched_for():
    lesson = spellings.teach("Look up the country singer, Ry Thornton spelled R-H-Y", None)
    assert lesson == ("inline", "Look up the country singer, Rhy Thornton")


def test_learned_name_is_fixed_in_later_requests_and_new_mishearings():
    spellings.teach("Rhy is spelled R-H-Y", "tell me about the musician Rye Thornton")
    assert spellings.apply("play rye thornton on youtube") == "play Rhy Thornton on youtube"
    assert spellings.apply("tell me about the musician Raya Thornton") == "tell me about the musician Rhy Thornton"


@pytest.mark.parametrize("text", ["bake some rye bread", "Billy Bob Thornton movies", "Mary Thornton"])
def test_other_words_and_people_are_left_alone(text):
    spellings.teach("Rhy is spelled R-H-Y", "tell me about Rye Thornton")
    assert spellings.apply(text) == text


def test_hand_edited_styling_is_kept(config):
    (config / "voxa").mkdir()
    (config / "voxa/spellings.json").write_text(json.dumps({"rhY Thornton": ["rye thornton"]}))
    assert spellings.apply("songs by Rye Thornton") == "songs by rhY Thornton"
    assert spellings.quote("the singer rhY Thornton") == 'the singer "rhY Thornton"'


def test_nothing_to_learn():
    assert spellings.teach("open pluma", "anything") is None
    assert spellings.teach("Rhy is spelled R-H-Y", None) is None
    assert spellings.teach("Zed is spelled Z-E-D", "open pluma") is None


def test_taught_names_lead_the_whisper_hint():
    hint = build_hint(["Pluma"], "voxa", people=["rhY Thornton"])
    assert hint.startswith("Voxa. Names: rhY Thornton. Commands:")
    assert build_hint(["Pluma"], "voxa").startswith("Voxa. Commands:")
