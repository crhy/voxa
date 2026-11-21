"""Issue 136: while PAUSED only the wake word may reach her - no false positives.

These are the decisions voxa/window.py applies on every action path (the wake-word
resume, the prompt gate, the exit phrases, speech and timed reminders); the VoxaTest
paused suite (voxatest/paused.py) measures the same functions through the real hearing
pipeline, so a rule that passes here and fails there cannot happen.
"""

from __future__ import annotations

from voxa.conversation import detect_exit_phrase, strip_wake_word
from voxa.pausepolicy import exit_action, may_speak, reminder_action, wake_allowed


def test_fuzzy_near_misses_do_not_end_a_pause():
    # The wake-word matcher (voxa/conversation.py) accepts these within its 0.65
    # threshold; that is exactly what woke a paused Voxa. They must not wake her now.
    for heard in ("voxel of the day", "Roxa called", "vox uh what time is it",
                  "locks a open pluma"):
        # The old trigger: the fuzzy matcher mistook these for "voxa".
        assert strip_wake_word(heard, "voxa") is not None
        allowed, _rest, why = wake_allowed(heard, "voxa")
        assert not allowed, heard
        assert why == "no wake word"
    for heard in ("boxer training", "Vox Media reported"):
        allowed, _rest, _why = wake_allowed(heard, "voxa")
        assert not allowed, heard


def test_words_the_ear_spelled_as_voxa_mid_sentence_do_not_end_a_pause():
    # Whisper base writes the exact name for ordinary speech; what marks a real call is
    # the address: the wake word ends its own breath ("Voxa." / "Voxa,"), and these do not.
    for heard in ("Voxa is a genus of head louse.", "Voxa what happened next?",
                  "Voxa training takes a lot of patience.", "Voxa's cash food for the winter."):
        allowed, _rest, why = wake_allowed(heard, "voxa")
        assert not allowed, heard
        assert why == "no address"


def test_the_wake_word_as_an_address_does():
    for heard in ("Voxa.", "Voxa!", "Voxa", "vox a", "Voxa, what time is it?",
                  "vaxa, open the browser", "boxa, stop the timer"):
        allowed, _rest, why = wake_allowed(heard, "voxa")
        assert allowed, heard
        assert why == "wake word"


def test_the_wake_word_and_its_accepted_spellings_do():
    for heard in ("voxa", "Voxa!", "vox a", "vaxa", "voxo", "boxa"):
        allowed, rest, why = wake_allowed(heard, "voxa")
        assert allowed, heard
        assert why == "wake word"


def test_her_own_reply_replayed_as_echo_does_not_wake_her():
    # voxa/ownspeech.py protects the wake word from stripping, so this echo of her
    # last reply keeps its "Voxa": the overlap share is what rejects it.
    heard = "Voxa, the capital of France is Paris"
    allowed, _rest, why = wake_allowed(heard, "voxa", own_share=0.8)
    assert not allowed
    assert why == "echo"
    # A genuine request that merely shares a word is not an echo.
    assert wake_allowed(heard, "voxa", own_share=0.2)[0]


def test_bare_exit_phrases_are_background_speech_while_paused():
    for phrase in ("cancel", "never mind", "forget it", "done", "see you", "stop",
                   "offline", "go to sleep", "turn off"):
        assert detect_exit_phrase(phrase, "voxa") is not None  # the old trigger
        assert exit_action(phrase, "voxa") is None              # paused: ignored
    # Ordinary sentences that merely contain one of them:
    assert exit_action("never mind I found it", "voxa") is None
    assert exit_action("cancel that order please", "voxa") is None


def test_an_exit_phrase_addressed_to_voxa_still_act():
    assert exit_action("Voxa, go offline", "voxa") == "goodbye"
    assert exit_action("vaxa, cancel", "voxa") == "cancel"


def test_silence_and_toasts_while_paused():
    assert may_speak(True) is False
    assert may_speak(False) is True
    assert reminder_action(True) == "toast"
    assert reminder_action(False) == "speak"


def test_decision_order_pause_hear_resume_route():
    class Assistant:
        paused = False

        def pause(self):
            self.paused = True

        def resume(self):
            self.paused = False

    assistant = Assistant()
    assistant.pause()
    for heard in ("voxel of the day", "cancel", "never mind",
                  "Voxa, the capital of France is Paris"):
        echo = 0.8 if "capital" in heard else 0.0
        allowed, _rest, _why = wake_allowed(heard, "voxa", own_share=echo)
        assert not allowed, heard
        assert exit_action(heard, "voxa") is None, heard
        assert assistant.paused
    allowed, rest, _why = wake_allowed("Voxa, what time is it", "voxa")
    assert allowed and rest == "what time is it"
    assistant.resume()
    assert not assistant.paused
