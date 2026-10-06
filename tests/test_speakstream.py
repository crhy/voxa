"""Speak-while-writing: sentences are released as soon as they are complete."""

from __future__ import annotations

from voxa.speakstream import SentenceFeeder, is_thinking_model, looks_like_reasoning


def test_first_sentence_is_released_as_soon_as_it_is_complete() -> None:
    feeder = SentenceFeeder()
    assert feeder.feed("Spaced Linux is a ") == []
    assert feeder.feed("Devuan-based distribution. It uses") == ["Spaced Linux is a Devuan-based distribution."]
    assert feeder.released_any


def test_a_tiny_opening_fragment_waits_for_more() -> None:
    feeder = SentenceFeeder()
    assert feeder.feed("Sure. ") == []  # too short to be worth a separate utterance
    assert feeder.feed("Here is what I found about it. More") == ["Sure. Here is what I found about it."]


def test_later_pieces_are_longer_and_nothing_is_lost_or_repeated() -> None:
    feeder = SentenceFeeder()
    text = "The first sentence is here. Short one. Another short. " + "This later sentence is long enough to be released on its own merits. Tail"
    out: list[str] = []
    for index in range(0, len(text), 7):
        out += feeder.feed(text[index:index + 7])
    out += feeder.finish()
    assert " ".join(out) == text
    assert out[0] == "The first sentence is here."


def test_finish_uses_the_cleaned_answer_and_keeps_what_was_already_spoken() -> None:
    feeder = SentenceFeeder()
    spoken = feeder.feed("Paris is the capital of France. It is")
    rest = feeder.finish("Paris is the capital of France. It is on the Seine.")
    assert spoken == ["Paris is the capital of France."]
    assert rest == ["It is on the Seine."]


def test_finish_with_nothing_left_returns_nothing() -> None:
    feeder = SentenceFeeder()
    feeder.feed("One complete sentence here. ")
    assert feeder.finish() == []


def test_decimal_points_and_abbreviations_inside_a_sentence_do_not_split_it_early() -> None:
    feeder = SentenceFeeder()
    assert feeder.feed("Version 3.14 was") == []  # "3.14" has no space after the dot


def test_thinking_models_and_scratchpads_are_recognised() -> None:
    assert is_thinking_model("hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S")
    assert is_thinking_model("deepseek-r1:14b")
    assert not is_thinking_model("qwen2.5:0.5b") and not is_thinking_model("phi4:14b")
    assert looks_like_reasoning("<think>hmm") and looks_like_reasoning("so the answer</think>") and not looks_like_reasoning("Hello.")
