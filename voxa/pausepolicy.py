"""What may reach Voxa while she is PAUSED (pure Python, no GTK).

Paused means "switched on but ignoring everything except the wake word" (issue 136).
Every action path that runs while paused - the wake-word resume, the prompt gate, the
exit phrases, spoken replies and timed reminders - decides here, in one place, so the
product (voxa/window.py) and the VoxaTest paused suite (voxatest/paused.py) measure the
SAME decision instead of two copies of it.

The two rules that keep the false positives out:

* only the wake word itself ends a pause. The wake-word matcher in
  voxa/conversation.py (:func:`voxa.conversation.strip_wake_word`) accepts anything
  within 0.65 of "voxa" so a command spoken in one breath survives Whisper ("voxel",
  "Roxa", ...); that threshold is why background speech woke her while paused. Here the
  strict spellings of :func:`voxa.pausewords.resume_request` (the wake word plus the
  misspellings the app already accepts) are the gate.
* an utterance that is (mostly) her own last reply is an echo, not a wake word.
  voxa/ownspeech.py protects the wake word from its own stripping, so an echo of a
  reply that named her ("Voxa, the capital of France is Paris") keeps its "Voxa" and
  woke her - the share over her remembered sentences rejects it instead.
"""

from __future__ import annotations

from .pausewords import resume_request, wake_aliases

#: An utterance whose words are at least this largely her own last reply is an echo.
#: The same threshold voxa/ownspeech.py uses to call a whole utteration her own voice.
ECHO_SHARE = 0.6

#: The break that makes the wake word an ADDRESS: its own breath, a comma, or the end.
_ADDRESS_BREAK = (",", ".", "!", "?", ";", ":", "-", "\u2014")


def addressed_wake(heard: str, wake_word: str) -> bool:
    """Did the wake word arrive AS the address, or did the ear merely spell a word like it?

    Whisper base turns "boxer training" into "Voxa training", "foxes cache" into "Voxa's
    cache" and "vox uh" into "Voxa what". Those transcripts hold the exact name, so the
    strict spelling alone cannot stop them. What does stop them is the shape of a real
    call: "Voxa." on its own breath, or "Voxa," before the request - the wake word ends
    its breath and the sentence follows. That is the same shape the conversation pipeline
    is built around (the wake utterance arms the prompt).
    """
    text = heard.strip()
    lowered = text.casefold()
    for alias in wake_aliases(wake_word):
        if lowered.startswith(alias):
            after = lowered[len(alias):len(alias) + 1]
            return after == "" or after in _ADDRESS_BREAK
    return False


def wake_allowed(heard: str, wake_word: str, own_share: float = 0.0) -> tuple[bool, str, str]:
    """(allowed, rest, why) for one utterance heard while paused.

    ``heard`` is the raw utterance (the wake word NOT stripped), ``own_share`` is
    voxa.ownspeech's text-overlap feature for it: how much of the utterance was
    something Voxa herself was saying.
    """
    if own_share >= ECHO_SHARE:
        # An echo of her own reply: even its protected "Voxa" is not a wake word.
        return (False, "", "echo")
    resumed, rest = resume_request(heard, wake_word)
    if not resumed:
        return (False, "", "no wake word")
    if not addressed_wake(heard, wake_word):
        # The ear spelled the name out of ordinary speech ("Voxa training takes..."):
        # a paused Voxa wakes for the address, not for the spelling.
        return (False, "", "no address")
    return (True, rest, "wake word")


def exit_action(heard: str, wake_word: str) -> str | None:
    """The exit phrase that may act while paused, or None to ignore it.

    "cancel", "never mind", "done" and the rest of voxa/conversation.py's exit phrases
    are ordinary words in background speech; while paused they act only when the
    sentence also names Voxa ("Voxa, go offline").
    """
    resumed, rest = resume_request(heard, wake_word)
    if not resumed or not rest:
        return None
    from .conversation import detect_exit_phrase

    return detect_exit_phrase(rest)


def may_speak(is_paused: bool) -> bool:
    """May she speak at all right now? While paused: no. A toast is enough."""
    return not is_paused


def reminder_action(is_paused: bool) -> str:
    """What a due reminder/timer/focus does: a toast while paused, speech when active.

    Timed actions must not wake her or take the floor while paused; the toast keeps the
    information (it is shown either way) without making a sound.
    """
    return "toast" if is_paused else "speak"
