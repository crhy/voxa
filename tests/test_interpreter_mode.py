"""Interpreter mode: one session that alternates two languages (issue #66).

The pair, the alternation, the spoken commands and which language the microphone
listens in are all pure logic, so they run offline. The live two-person audio
loop is not: see the notes.
"""

from __future__ import annotations

from voxa.agent import interpreter
from voxa.agent.intents import route
from voxa.agent.policy import RiskLevel
from voxa.agent.tools import default_registry
from voxa.agent.tools import interpreter as tools
from voxa.agent.tools import translate as translate_tool
from voxa.conversation import ConversationController


def test_pair_resolves_spoken_names() -> None:
    assert interpreter.pair("english", "spanish") == ("English", "Spanish")
    assert interpreter.pair("español", "japanese") == ("Spanish", "Japanese")
    assert interpreter.pair("DEUTSCH.", "russian") == ("German", "Russian")
    assert interpreter.pair("mandarin", "spanish") is None  # no voice for it
    assert interpreter.pair("english", "english") is None  # one language is not interpreting
    assert interpreter.pair_in("between french and japanese") == ("French", "Japanese")
    assert interpreter.pair_in("english and spanish") == ("English", "Spanish")
    assert interpreter.pair_in("no languages here") is None


def test_other_side_and_answer_language() -> None:
    sides = ("English", "Spanish")
    assert interpreter.other(sides, "English") == "Spanish"
    assert interpreter.other(sides, "Spanish") == "English"
    assert interpreter.other(sides, "German") is None
    assert interpreter.other(None, "English") is None
    assert interpreter.target_language(sides, "Spanish") == "English"
    assert interpreter.target_language(sides, None) is None


def test_prompt_keeps_the_strict_translation_wording() -> None:
    prompt = interpreter.prompt_for("Japanese")
    assert "Japanese" in prompt
    assert "ONLY the translation" in prompt
    assert prompt == translate_tool.PROMPT.format(language="Japanese")


def test_session_alternates_across_three_turns() -> None:
    session = interpreter.Session()
    assert session.start("english", "spanish")
    assert session.sides() == ("English", "Spanish")
    assert session.last_side == "English"
    turns: list[tuple[str, str]] = []
    for _ in range(3):
        heard = session.last_side
        turns.append((heard, interpreter.target_language(session.sides(), heard)))
        session.switch()
    assert turns == [("English", "Spanish"), ("Spanish", "English"), ("English", "Spanish")]
    session.stop()
    assert not session.active
    assert session.a is None and session.b is None and session.last_side is None
    assert session.sides() is None
    session.switch()
    assert session.last_side is None


def test_start_stop_and_status_tools() -> None:
    session = tools.SESSION
    session.stop()
    started = tools.interpreter_start({"language_a": "english", "language_b": "spanish"})
    assert started.ok and "English" in started.speech and "Spanish" in started.speech
    assert session.active and session.last_side == "English"
    status = tools.interpreter_status({})
    assert status.ok
    assert "Interpreting between English and Spanish" in status.speech
    assert "listening for English" in status.speech and "answering in Spanish" in status.speech
    assert tools.interpreter_switch({}).ok and session.last_side == "Spanish"
    status = tools.interpreter_status({})
    assert "listening for Spanish" in status.speech and "answering in English" in status.speech
    assert tools.interpreter_stop({}).ok and not session.active
    assert not tools.interpreter_stop({}).ok
    assert not tools.interpreter_status({}).ok
    assert not tools.interpreter_switch({}).ok


def test_unknown_language_names_the_languages_she_has() -> None:
    result = tools.interpreter_start({"language_a": "klingon", "language_b": "spanish"})
    assert not result.ok
    for name in ("Arabic", "English", "Spanish", "Russian"):
        assert name in result.speech
    assert not tools.SESSION.active
    same = tools.interpreter_start({"language_a": "english", "language_b": "english"})
    assert not same.ok and "two different languages" in same.speech
    assert not tools.SESSION.active
    assert not tools.interpreter_start({"language_a": "", "language_b": ""}).ok


def test_interpreter_tools_are_registered_read_only() -> None:
    registry = default_registry()
    names = set(registry.names())
    assert {"interpreter_start", "interpreter_stop", "interpreter_status"} <= names
    assert "interpreter_switch" in names
    for name in ("interpreter_start", "interpreter_stop", "interpreter_status", "interpreter_switch"):
        assert registry.get(name).risk == RiskLevel.READ_ONLY


def test_router_phrasings() -> None:
    def routed(text: str) -> str:
        call = route(text)
        return call.tool if call is not None else ""

    start = route("interpreter mode english and spanish")
    assert start.tool == "interpreter_start" and start.args == {"language_a": "English", "language_b": "Spanish"}
    assert route("start interpreting between french and japanese").args == {
        "language_a": "French",
        "language_b": "Japanese",
    }
    assert route("stop interpreting").tool == "interpreter_stop"
    assert route("switch languages").tool == "interpreter_switch"
    assert route("what languages am I interpreting").tool == "interpreter_status"
    assert route("interpreter mode").tool == "interpreter_start"
    # Nothing already spoken changes meaning.
    assert route("how do I say where is the station in Japanese").tool == "say_in_language"
    assert routed("switch to spotify") != "interpreter_switch"
    assert routed("change the language to spanish") != "interpreter_switch"
    assert routed("what languages do you speak") != "interpreter_status"
    assert routed("interpreting the difference between Java and JavaScript") != "interpreter_start"


class Recorder:
    """A Whisper stand-in that records which language it was asked for."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def transcribe(self, segment: bytes, language: str, hint: str = "") -> str:
        self.asked.append(language)
        return "hello"


def _controller(wake: Recorder, prompt: Recorder) -> ConversationController:
    return ConversationController(
        wake_whisper=wake,
        prompt_whisper=prompt,
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=100,
        voice_detector=False,
        max_segment_seconds=0.3,
        on_woken=lambda: None,
        on_prompt=lambda _text: None,
        on_status=lambda _text: None,
        on_error=lambda _text: None,
    )


def test_each_side_is_transcribed_in_its_own_language() -> None:
    assert interpreter.transcription_code("Japanese") == "ja"
    assert interpreter.transcription_code("English") == "en"
    assert interpreter.transcription_code(None) is None


def test_only_the_real_model_follows_the_side_language() -> None:
    wake, prompt = Recorder(), Recorder()
    controller = _controller(wake, prompt)
    controller.prompt_language = "es"
    assert controller._transcribe(b"\x00\x10" * 8, prompt) == "hello"
    assert controller._transcribe(b"\x00\x10" * 8, wake, "Voxa.") == "hello"
    # The utterance is heard in Spanish; the wake word is still chased in English.
    assert prompt.asked == ["es"] and wake.asked == ["en"]
    controller.prompt_language = None
    assert controller._transcribe(b"\x00\x10" * 8, prompt) == "hello"
    assert prompt.asked == ["es", "en"]
