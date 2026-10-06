"""Tests for the pause-voice helpers: pure functions plus one decision-order check."""

from __future__ import annotations

from voxa.pausewords import is_pause_request, resume_request


def test_pause_phrases_without_media():
    for text in (
        "pause listening",
        "Pause Listening!",
        "stop listening",
        "take a break",
        "go to sleep",
        "voxa pause",
        "pause voxa",
        "pause",
        "voxa, take a break",
        "vox a pause",
    ):
        assert is_pause_request(text, media_playing=False) is True


def test_pause_phrases_with_media():
    for text in ("pause listening", "stop listening", "take a break", "go to sleep", "voxa pause", "pause voxa"):
        assert is_pause_request(text, media_playing=True) is True
    # "pause" alone is always for Voxa herself; the music is paused only when it is named.
    assert is_pause_request("pause", media_playing=True) is True
    assert is_pause_request("voxa pause", media_playing=True) is True


def test_pause_rejects_unrelated():
    for text in ("what time is it", "pause the music", "play some jazz", "I need a break", "sleep mode"):
        assert is_pause_request(text, media_playing=False) is False
        assert is_pause_request(text, media_playing=True) is False


def test_resume_bare_wake_word():
    assert resume_request("voxa", "voxa") == (True, "")
    assert resume_request("Voxa!", "voxa") == (True, "")
    assert resume_request("vox a", "voxa") == (True, "")
    assert resume_request("vaxa", "voxa") == (True, "")
    assert resume_request("voxo", "voxa") == (True, "")
    assert resume_request("boxa", "voxa") == (True, "")
    assert resume_request("computer", "computer") == (True, "")


def test_resume_with_request():
    assert resume_request("voxa what time is it?", "voxa") == (True, "what time is it")
    assert resume_request("vaxa, open the browser", "voxa") == (True, "open the browser")
    assert resume_request("vox a pause listening", "voxa") == (True, "pause listening")


def test_resume_rejects_other_sentences():
    assert resume_request("what time is it", "voxa") == (False, "what time is it")
    assert resume_request("elevate the volume", "voxa") == (False, "elevate the volume")
    assert resume_request("voxa", "computer") == (False, "voxa")


def test_decision_order():
    class Assistant:
        paused = False

        def pause(self):
            self.paused = True

        def resume(self):
            self.paused = False

    def decide(text: str, assistant: Assistant, media_playing: bool) -> str:
        if assistant.paused:
            resumed, rest = resume_request(text, "voxa")
            if not resumed:
                return "dropped"
            assistant.resume()
            return "resumed" if not rest else f"resumed-and-routed {rest}"
        if is_pause_request(text, media_playing=media_playing):
            assistant.pause()
            return "paused"
        return "routed"

    assistant = Assistant()
    assert decide("pause listening", assistant, False) == "paused"
    assert assistant.paused
    assert decide("what time is it", assistant, False) == "dropped"
    assert assistant.paused
    assert decide("pause", assistant, True) == "dropped"
    assert decide("vaxa", assistant, True) == "resumed"
    assert not assistant.paused
    assert decide("pause", assistant, True) == "paused"
    assistant.resume()
    assert decide("pause the music", assistant, True) == "routed"
    assert decide("voxa take a break", assistant, False) == "paused"
    assert assistant.paused
