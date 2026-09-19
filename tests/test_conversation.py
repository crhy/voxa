from __future__ import annotations

from voxa.conversation import ConversationController, ConversationHistory, detect_exit_phrase, strip_wake_word


def test_strip_wake_word_returns_none_when_absent() -> None:
    assert strip_wake_word("what time is it", "computer") is None


def test_strip_wake_word_returns_empty_remainder_for_bare_wake_word() -> None:
    assert strip_wake_word("Computer", "computer") == ""


def test_strip_wake_word_returns_trailing_command_in_same_utterance() -> None:
    assert (
        strip_wake_word("Computer, what's the weather today", "computer")
        == "what's the weather today"
    )


def test_strip_wake_word_is_case_insensitive_and_matches_multi_word_phrase() -> None:
    assert strip_wake_word("Hey Voice Assistant tell me a joke", "hey voice assistant") == "tell me a joke"


def test_strip_wake_word_with_empty_configured_word_never_matches() -> None:
    assert strip_wake_word("computer", "") is None


def test_detect_exit_phrase_matches_cancel_phrases() -> None:
    for utterance in ("Never mind.", "CANCEL", "Forget it!", "nevermind"):
        assert detect_exit_phrase(utterance) == "cancel", utterance


def test_detect_exit_phrase_matches_goodbye_phrases() -> None:
    for utterance in ("Goodbye", "Bye", "Goodbye for now", "See you!", "That's all", "thats all", "Done"):
        assert detect_exit_phrase(utterance) == "goodbye", utterance


def test_detect_exit_phrase_ignores_plain_questions() -> None:
    for utterance in (
        "never mind that detail, what's the weather?",
        "good",
        "I'm done with my homework",
        "can you stop the weather updating?",
        "bye, and tell me a joke later",
    ):
        assert detect_exit_phrase(utterance) is None, utterance


def test_detect_exit_phrase_requires_exact_utterance() -> None:
    assert detect_exit_phrase("please cancel that") is None
    assert detect_exit_phrase("goodbye everyone") is None


def test_system_prompt_survives_many_turns() -> None:
    history = ConversationHistory("You are Voxa.", max_turns=24)
    for i in range(50):
        history.add_user(f"question {i}")
        history.add_assistant(f"answer {i}")

    messages = history.messages()
    assert messages[0] == {"role": "system", "content": "You are Voxa."}
    assert messages[-1] == {"role": "assistant", "content": "answer 49"}


def test_turn_limit_evicts_only_oldest_turns() -> None:
    history = ConversationHistory("You are Voxa.", max_turns=4)
    for i in range(10):
        history.add_user(f"question {i}")
        history.add_assistant(f"answer {i}")

    messages = history.messages()
    assert len(messages) == 5
    assert messages[1] == {"role": "user", "content": "question 8"}


def test_drop_last_and_clear_only_touch_turns() -> None:
    history = ConversationHistory("You are Voxa.", max_turns=4)
    history.add_user("question")
    history.add_assistant("answer")
    history.drop_last()
    assert history.messages() == [
        {"role": "system", "content": "You are Voxa."},
        {"role": "user", "content": "question"},
    ]
    history.drop_last()
    history.drop_last()
    assert history.messages() == [{"role": "system", "content": "You are Voxa."}]
    history.add_user("question")
    history.clear()
    assert history.messages() == [{"role": "system", "content": "You are Voxa."}]


class StoppingTranscriber:
    """Sets the controller's stop event mid-"transcription"."""

    def __init__(self, controller: ConversationController) -> None:
        self.controller = controller

    def transcribe(self, segment: bytes, language: str) -> str:
        self.controller.stop()
        return "voxa, what time is it"


def test_stop_during_transcription_drops_prompt_callback() -> None:
    loud = (b"\x00\x10" * 1000, 1000.0)
    quiet = (b"\x00\x00" * 1000, 0.0)

    prompts: list[str] = []
    woken: list[bool] = []
    controller = ConversationController(
        wake_whisper=object(),
        prompt_whisper=object(),
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=100,
        max_segment_seconds=5.0,
        on_woken=lambda: woken.append(True),
        on_prompt=prompts.append,
        on_status=lambda _status: None,
        on_error=lambda _error: None,
    )
    controller.wake_whisper = StoppingTranscriber(controller)

    controller.start()
    for _ in range(10):
        controller.feed(*loud)
    for _ in range(5):
        controller.feed(*quiet)
    if controller.thread is not None:
        controller.thread.join(timeout=3.0)

    assert prompts == []
    assert woken == []


def test_strip_wake_word_tolerates_close_misspelling_at_start() -> None:
    assert strip_wake_word("Vox, what time is it", "voxa") == "what time is it"
    assert strip_wake_word("Voxer what time is it", "voxa") == "what time is it"


def test_strip_wake_word_fuzzy_does_not_fire_on_ordinary_speech() -> None:
    assert strip_wake_word("what time is it", "voxa") is None
    assert strip_wake_word("Mark, sir, what time is it?", "voxa") is None
    assert strip_wake_word("the voice is loud", "voxa") is None


def test_detect_exit_phrase_stop_and_go_offline() -> None:
    for utterance in ("Stop.", "Stop talking", "stop it", "Stop dictation"):
        assert detect_exit_phrase(utterance) == "stop", utterance
    for utterance in ("Go offline", "go offline.", "Turn off", "Goodbye"):
        assert detect_exit_phrase(utterance) == "goodbye", utterance
    assert detect_exit_phrase("stop the presses and tell me the news") is None
