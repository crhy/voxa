"""Whisper hallucinations on noise never interrupt Voxa (issue #72, task v11d item 2).

The single INTERRUPT that survived the v11c no-speech guard was ``heard='Thanks for watching.'`` -
a classic Whisper hallucination on the echo path's residual, her own closing line invented out of
silence. The guard drops such transcripts (and high no_speech_prob brackets) UNLESS the cleaned
signal really carries speech, so a genuine user turn is still heard.
"""

from voxa.attribution import SourceEvidence
from voxa.bargein import is_whisper_hallucination
from voxa.bargectl import HOLD, INTERRUPT, BargeInController

DT_MS = 100.0
HERS = "The weather is clear today and the sun is out."


def _drive(candidate, evidence):
    control = BargeInController()
    control.reply_started(HERS)
    for _ in range(2 * int(400.0 / DT_MS)):
        control.observe(0.05, speaking=True, muted=True, dt_ms=DT_MS)
    outcome = None
    for _ in range(60):
        outcome = control.observe(0.9, speaking=True, muted=True, dt_ms=DT_MS,
                                  candidate=candidate, evidence=evidence)
    return outcome


def _loud_mic_no_real_speech() -> SourceEvidence:
    """Reference silent, microphone loud, but NO speech in the cleaned signal: attribution says USER."""
    return SourceEvidence(reference_active=False, reference_power=0.0, mic_level=0.9, vad_prob=None)


def _loud_mic_real_speech() -> SourceEvidence:
    return SourceEvidence(reference_active=False, reference_power=0.0, mic_level=0.9, vad_prob=0.9)


# --- pure guard ---------------------------------------------------------------------------

def test_known_silence_phrases_are_hallucinations_without_speech() -> None:
    for phrase in ("Thanks for watching.", "Thanks for watching", "subscribe",
                   "like and subscribe", "you", "music", "."):
        assert is_whisper_hallucination(phrase, phrase, vad_prob=None), phrase


def test_high_no_speech_prob_is_a_hallucination_without_speech() -> None:
    assert is_whisper_hallucination("something strange", "something strange",
                                    no_speech_prob=0.85, vad_prob=0.2)


def test_real_speech_is_never_a_hallucination() -> None:
    # The same phrase with real VAD speech in the cleaned signal is kept.
    assert not is_whisper_hallucination("Thanks for watching.", "Thanks for watching.",
                                        no_speech_prob=0.9, vad_prob=0.8)


def test_a_real_user_turn_is_not_a_hallucination() -> None:
    assert not is_whisper_hallucination("open the library", "open the library", vad_prob=None)


# --- bargectl integration -----------------------------------------------------------------

def test_hallucinated_phrase_holds_instead_of_interrupting() -> None:
    outcome = _drive(lambda: ("Thanks for watching.", "Thanks for watching."),
                     _loud_mic_no_real_speech())
    assert outcome.action == HOLD, outcome
    assert outcome.reason == "hallucination"


def test_real_speech_phrase_still_interrupts() -> None:
    # Same words, but the cleaned signal has real speech: the guard must not swallow it.
    outcome = _drive(lambda: ("Thanks for watching.", "Thanks for watching."),
                     _loud_mic_real_speech())
    assert outcome.action == INTERRUPT, outcome


def test_a_real_interrupt_survives_the_guard() -> None:
    outcome = _drive(lambda: ("open the library", "open the library"), _loud_mic_no_real_speech())
    assert outcome.action == INTERRUPT, outcome
