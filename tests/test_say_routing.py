import voxa.agent.tools.typing as typing_mod
from voxa.agent.intents import route


def test_would_you_say_questions_not_typed():
    assert route("Would you say that operating system is better than windows") is None
    assert route("Would you say a space clinics is better than windows") is None


def test_say_that_again_repeats_last():
    for text in ("Can you say that again", "say that again", "repeat that", "what did you say"):
        call = route(text)
        assert call is not None and call.tool == "repeat_last"


def test_type_still_types():
    call = route("type hello world")
    assert call is not None and call.tool == "type_text"
    assert call.args == {"text": "hello world"}


def test_say_hello_speaks():
    call = route("say hello to everyone")
    assert call is not None and call.tool == "say_text"
    assert call.args == {"text": "hello to everyone"}


def test_how_do_you_say_goes_to_model():
    assert route("how do you say thank you in Japanese") is None


def test_repeat_last_without_previous_reply():
    typing_mod.LAST_REPLY = ""
    result = typing_mod._repeat_last_handler({})
    assert result.ok
    assert result.speech == "I have not said anything yet."


def test_repeat_last_with_previous_reply():
    typing_mod.LAST_REPLY = "Hello there."
    result = typing_mod._repeat_last_handler({})
    assert result.ok
    assert result.speech == "Hello there."
