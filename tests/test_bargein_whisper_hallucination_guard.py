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


def test_an_unproven_delay_makes_vad_alone_keep_a_known_phrase_issue_72_v11d():
    """hiroshi live failure (22-char v11d): "Thanks for watching." on her Japanese echo
    INTERRUPTED her because Silero fired at 0.67 ON THE RESIDUE the canceller left - at a
    delay the exact aligner never confirmed. VAD alone only vetoes the guard when the echo
    delay is a proved lock; parked, it must clear Whisper's known phrases too."""
    assert is_whisper_hallucination("Thanks for watching.", "Thanks for watching.",
                                    vad_prob=0.67, delay_proven=False)
    # A proved delay keeps the old veto: real speech in the cleaned signal wins.
    assert not is_whisper_hallucination("Thanks for watching.", "Thanks for watching.",
                                        vad_prob=0.67, delay_proven=True)
    # Unproven delay and real speech: an UNKNOWN, confident transcript still interrupts -
    # the guard reads phrases and bars, not "the user is wrong".
    assert not is_whisper_hallucination("stop the timer now please",
                                        "stop the timer now please",
                                        vad_prob=0.9, avg_logprob=-0.3,
                                        compression_ratio=1.1, delay_proven=False)


def test_a_real_user_turn_is_not_a_hallucination() -> None:
    assert not is_whisper_hallucination("open the library", "open the library", vad_prob=None)


def test_low_confidence_transcript_is_a_hallucination_without_speech() -> None:
    # Claude's 18:35 audit: the phrase list alone let a fresh invention through ("The link is
    # in the description." was only added after the fact). Whisper's OWN segment confidence is
    # the bar: a bad avg_logprob or a repeating compression ratio on a no-speech moment is the
    # echo-path residual being talked over, whatever the words happened to be.
    assert is_whisper_hallucination("the link is in the description.", "the link is in the description.",
                                    vad_prob=None, avg_logprob=-1.6)
    assert is_whisper_hallucination("some fresh invention here", "some fresh invention here",
                                    vad_prob=None, compression_ratio=3.1)


def test_confidence_bar_needs_a_real_speech_turn() -> None:
    # The bar only bites where the cleaned signal shows no speech: a genuine user turn that
    # happens to transcribe poorly stays an interruption.
    assert not is_whisper_hallucination("open the library", "open the library",
                                        vad_prob=0.9, avg_logprob=-1.6, compression_ratio=3.1)


def test_good_confidence_is_no_bar_at_all() -> None:
    assert not is_whisper_hallucination("some fresh invention here", "some fresh invention here",
                                        vad_prob=None, avg_logprob=-0.3, compression_ratio=1.4)


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


def test_bad_whisper_confidence_holds_an_unknown_phrase() -> None:
    # The same transcript the attribution evidence carries: when the candidate pass itself
    # reports a failing segment (avg_logprob below the bar), a phrase not on the list is still
    # the residual being talked over - task v11d (Claude's 18:35 note: the guard was fed no
    # confidence at all).
    failing = _loud_mic_no_real_speech()
    failing.avg_logprob = -1.8
    outcome = _drive(lambda: ("some fresh invention here", "some fresh invention here"), failing)
    assert outcome.action == HOLD, outcome
    assert outcome.reason == "hallucination"


def test_an_empty_span_the_guard_holds_still_records_its_decision() -> None:
    """Task v11d step 4 (live 22-char gate, aoife/charlotte/greta): the reference check
    counts attribution lines during a reply ("alignment healthy: H of D", decisions must be
    > 0). Empty spans - the echo's pauses, transcribed as nothing - were being dropped by
    the hallucination guard BEFORE attribution, so whole replies decided nothing. The guard
    now runs after the attribution trace: the span is still held, but it was decided."""
    events: list[str] = []
    control = BargeInController(trace=lambda event, **kw: events.append(event))
    control.reply_started(HERS)
    for _ in range(2 * int(400.0 / DT_MS)):
        control.observe(0.05, speaking=True, muted=True, dt_ms=DT_MS)
    outcome = None
    for _ in range(60):
        outcome = control.observe(0.9, speaking=True, muted=True, dt_ms=DT_MS,
                                  candidate=lambda: ("", ""), evidence=_loud_mic_no_real_speech())
    assert outcome.action == HOLD, outcome
    assert "words" in events and "hallucination" in events, events
    assert events.index("words") < events.index("hallucination"), events
