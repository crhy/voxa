from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from voxa.conversation import (
    ConversationController,
    detect_exit_phrase,
    strip_wake_word,
)


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
    for utterance in ("Never mind.", "CANCEL", "Stop talking", "stop it", "Forget it!", "nevermind"):
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


# ConversationController drives the same pause segmentation as dictation, so
# controller tests feed real chunks and let the 0.25 s queue.get timeout in
# segment_stream advance time (mirroring tests/test_dictation.py).
LOUD_CHUNK = (b"\x00\x10" * 1000, 1000.0)  # 0.0625 s of "speech", above threshold


class FakeWhisper:
    """A stand-in for WhisperService with canned transcripts and call counts."""

    def __init__(self, canned=()) -> None:
        self.canned = list(canned)
        self.calls = 0
        self.error: Exception | None = None

    def transcribe(self, pcm: bytes, language: str = "en") -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.canned:
            return self.canned.pop(0)
        return ""


class _Events:
    """Collects the conversation callbacks so tests can assert on them."""

    def __init__(self) -> None:
        self.woken = 0
        self.prompts: list[str] = []
        self.statuses: list[str] = []
        self.errors: list[str] = []
        self.exits: list[str] = []

    def on_woken(self) -> None:
        self.woken += 1

    def on_prompt(self, text: str) -> None:
        self.prompts.append(text)

    def on_status(self, text: str) -> None:
        self.statuses.append(text)

    def on_error(self, text: str) -> None:
        self.errors.append(text)

    def on_exit(self, kind: str) -> None:
        self.exits.append(kind)


def make_controller(
    wake: FakeWhisper,
    prompt: FakeWhisper,
    wake_word: str = "voxa",
    prompt_timeout: float | None = None,
) -> tuple[ConversationController, _Events]:
    events = _Events()
    controller = ConversationController(
        wake_whisper=wake,
        prompt_whisper=prompt,
        language="en",
        wake_word=wake_word,
        threshold=450,
        silence_ms=100,
        max_segment_seconds=6.0,
        on_woken=events.on_woken,
        on_prompt=events.on_prompt,
        on_status=events.on_status,
        on_error=events.on_error,
        on_exit=events.on_exit,
    )
    if prompt_timeout is not None:
        controller.PROMPT_TIMEOUT_SECONDS = prompt_timeout
    return controller, events


@pytest.fixture()
def run_controller() -> "Callable[ConversationController, ConversationController]":
    def start(controller: ConversationController) -> ConversationController:
        controller.start()
        started.append(controller)
        return controller

    started: list[ConversationController] = []
    yield start
    for controller in started:
        controller.stop()
        if controller.thread is not None:
            controller.thread.join(timeout=5)


def feed_speech(controller: ConversationController, chunks: int = 6) -> None:
    # Six loud chunks = 0.375 s, past the 0.3 s minimum content for one
    # complete segment; the next 0.25 s queue timeout closes it.
    for _ in range(chunks):
        controller.feed(*LOUD_CHUNK)


def wait_for(condition, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_wake_word_remainder_wakes_and_prompts_in_same_utterance(run_controller):
    wake = FakeWhisper(["Voxa, what's the weather today"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: len(events.prompts) == 1)
    assert events.woken == 1
    assert events.prompts == ["what's the weather today"]
    assert prompt.calls == 0  # the remainder never needs the real model


def test_bare_wake_word_then_prompt_on_next_utterance(run_controller):
    wake = FakeWhisper(["Voxa"])
    prompt = FakeWhisper(["what's the weather tomorrow"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: "Listening for your request…" in events.statuses)
    assert controller.waiting_for_prompt is True
    feed_speech(controller)
    assert wait_for(lambda: events.prompts == ["what's the weather tomorrow"])
    assert events.woken == 1
    assert wake.calls == 1
    assert prompt.calls == 1
    assert controller.waiting_for_prompt is False


def test_armed_prompt_bypasses_wake_word(run_controller):
    wake = FakeWhisper()
    prompt = FakeWhisper(["", "the weather tomorrow"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.arm_prompt()
    feed_speech(controller)
    assert wait_for(lambda: prompt.calls == 1)
    assert events.prompts == []
    assert events.woken == 0
    assert controller.waiting_for_prompt is True  # empty transcript keeps waiting
    feed_speech(controller)
    assert wait_for(lambda: events.prompts == ["the weather tomorrow"])
    assert prompt.calls == 2
    assert wake.calls == 0
    assert controller.waiting_for_prompt is False


def test_muted_controller_drops_audio_until_unmuted(run_controller):
    wake = FakeWhisper(["Voxa"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.mute()
    feed_speech(controller)
    time.sleep(1.0)
    assert events.woken == 0 and events.prompts == [] and events.statuses == []
    controller.unmute()
    feed_speech(controller)
    assert wait_for(lambda: "Listening for your request…" in events.statuses)


def test_cancel_ends_turn_but_keeps_listening(run_controller):
    wake = FakeWhisper(["cancel", "Voxa"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: events.exits == ["cancel"])
    assert events.prompts == []
    assert controller.waiting_for_prompt is False
    assert controller.thread is not None and controller.thread.is_alive()
    feed_speech(controller)
    assert wait_for(lambda: events.woken == 1)


def test_exit_phrase_is_checked_even_in_prompt_phase(run_controller):
    wake = FakeWhisper()
    prompt = FakeWhisper(["stop talking"])
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    controller.arm_prompt()
    feed_speech(controller)
    assert wait_for(lambda: events.exits == ["cancel"])
    assert events.prompts == []
    assert controller.waiting_for_prompt is False
    assert controller.thread is not None and controller.thread.is_alive()


def test_goodbye_stops_the_worker(run_controller):
    wake = FakeWhisper(["Goodbye"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: events.exits == ["goodbye"])
    assert controller.thread is not None
    controller.thread.join(timeout=5)
    assert not controller.thread.is_alive()
    feed_speech(controller)
    time.sleep(0.5)
    assert events.prompts == [] and events.woken == 0


def test_prompt_idle_timeout_reports_and_resets(run_controller):
    wake = FakeWhisper(["Voxa"])
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt, prompt_timeout=1.0)
    run_controller(controller)
    feed_speech(controller)
    assert wait_for(lambda: "Listening for your request…" in events.statuses)
    assert wait_for(
        lambda: "Didn't catch that — say “voxa” again." in events.statuses,
        timeout=5.0,
    )
    assert controller.waiting_for_prompt is False


def test_transcription_error_is_reported_and_worker_survives(run_controller):
    wake = FakeWhisper()
    prompt = FakeWhisper()
    controller, events = make_controller(wake, prompt)
    run_controller(controller)
    wake.error = RuntimeError("model crash")
    feed_speech(controller)
    assert wait_for(lambda: events.errors == ["model crash"])
    assert events.woken == 0
    wake.error = None
    wake.canned.append("Voxa")
    feed_speech(controller)
    assert wait_for(lambda: events.woken == 1)
    assert controller.thread is not None and controller.thread.is_alive()
