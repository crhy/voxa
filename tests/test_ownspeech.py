from __future__ import annotations

from voxa.conversation import ConversationController
from voxa.ownspeech import OwnSpeech, normalise

SENTENCE = "The weather in Portland is sunny with a high of seventy two degrees"


def test_normalise():
    assert normalise("The weather in Portland is sunny with a high of seventy two degrees") == [
        "the", "weather", "in", "portland", "is", "sunny", "with", "a", "high", "of", "seventy", "two", "degrees",
    ]
    assert normalise("Hello, world!") == ["hello", "world"]
    assert normalise("72 degrees") == ["72", "degrees"]
    assert normalise("") == []
    assert normalise("...!?") == []


def test_echo_of_full_sentence():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.active()
    assert os.is_echo("the weather in portland is sunny")


def test_strip_keeps_command_words():
    os = OwnSpeech()
    os.started(SENTENCE)
    remaining, share = os.strip("weather in Portland is sunny Voxa stop", protect=["voxa"])  # noqa: B005
    assert remaining == "voxa stop"
    assert share > 0.0
    assert not os.is_echo("weather in Portland is sunny Voxa stop", protect=["voxa"])


def test_unrelated_untouched():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.strip("what time is it") == ("what time is it", 0.0)  # noqa: B005
    assert not os.is_echo("what time is it")


def test_fuzzy_echo():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.is_echo("the whether in portland is sunni")


def test_chance_words_untouched():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.strip("is it") == ("is it", 0.0)  # noqa: B005
    assert not os.is_echo("is it")


def test_expires_after_keep_seconds():
    fake = [0.0]
    os = OwnSpeech(keep_seconds=6.0, clock=lambda: fake[0])
    os.started(SENTENCE)
    os.finished()
    fake[0] = 7.0
    assert not os.active()
    assert os.strip("the weather in portland is sunny") == ("the weather in portland is sunny", 0.0)  # noqa: B005
    assert not os.is_echo("the weather in portland is sunny")


def test_three_sentences_in_a_row():
    os = OwnSpeech()
    os.started("The weather in Portland is sunny")
    os.started("It is seventy two degrees today")
    os.started("Have a nice afternoon")
    assert os.is_echo("the weather in portland is sunny")
    assert os.is_echo("it is seventy two degrees")
    assert os.is_echo("have a nice afternoon")


def test_empty_and_punctuation():
    os = OwnSpeech()
    os.started(SENTENCE)
    assert os.strip("") == ("", 0.0)
    assert not os.is_echo("")
    assert os.strip("...!?") == ("...!?", 0.0)  # noqa: B005
    assert not os.is_echo("...!?")


def test_controller_path_drops_echo_and_strips_mixed():
    class _Whisper:
        def transcribe(self, *args, **kwargs):
            return ""

    class _Service:
        def model_name(self):
            return "tiny"

    prompts: list[str] = []
    controller = ConversationController(
        wake_whisper=_Whisper(),
        prompt_whisper=_Whisper(),
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=400,
        max_segment_seconds=8.0,
        on_woken=lambda: None,
        on_prompt=lambda text: prompts.append(text),
        on_status=lambda text: None,
        on_error=lambda text: None,
        on_transcribe_timing=lambda *args: None,
    )
    os = OwnSpeech()
    os.started(SENTENCE)
    controller.own_speech = os

    # A pure echo is dropped: the filter returns None, so no prompt callback runs.
    assert controller._filter_own_speech("the weather in portland is sunny") is None
    assert prompts == []

    # A mixed utterance survives with only her own word runs removed.
    mixed = controller._filter_own_speech("the weather in portland is sunny what time is it")
    assert mixed is not None
    assert "weather" not in mixed
    assert "what time is it" in mixed
