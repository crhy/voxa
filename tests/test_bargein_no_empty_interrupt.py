"""A barge-in must never fire on an empty, wake-word-only or filler transcript (issue #130).

The traced self-interruption was a level rise with ``heard=''`` right after ``gate-open``: a
reference that had gone silent made the attribution read a bare level rise as the user, and
Voxa cancelled her own reply on a "transcript" with no words. The transcript is the only thing
that decides what happens to her words - no words, no interrupt.
"""

from voxa.attribution import SourceEvidence
from voxa.bargectl import HOLD, INTERRUPT, BargeInController

DT_MS = 100.0
HERS = "The weather is clear today and the sun is out."


def _drive(candidate, evidence):
    """Settle the baseline on her voice, then drive a loud run of frames through observe."""
    control = BargeInController()
    control.reply_started(HERS)
    for _ in range(2 * int(400.0 / DT_MS)):  # onset window + settle: the baseline is trusted
        control.observe(0.05, speaking=True, muted=True, dt_ms=DT_MS)
    outcome = None
    for _ in range(60):  # someone/something loud over her reply, past the listening window
        outcome = control.observe(0.9, speaking=True, muted=True, dt_ms=DT_MS,
                                  candidate=candidate, evidence=evidence)
    return outcome


def _silent_reference() -> SourceEvidence:
    """The exact failing state: reference silent, microphone loud - the attribution says USER."""
    return SourceEvidence(reference_active=False, reference_power=0.0, mic_level=0.9, vad_prob=0.9)


def test_empty_transcript_never_interrupts() -> None:
    assert _drive(lambda: ("", ""), _silent_reference()).action != INTERRUPT


def test_empty_transcript_is_a_hold_not_an_interrupt() -> None:
    assert _drive(lambda: ("", ""), _silent_reference()).action == HOLD


def test_wake_word_only_never_interrupts() -> None:
    assert _drive(lambda: ("voxa", "voxa"), _silent_reference()).action != INTERRUPT


def test_filler_only_never_interrupts() -> None:
    for heard in ("uh", "hmm", "uh uh", "[noise]", "(inaudible)", "um um um"):
        assert _drive(lambda heard=heard: (heard, heard), _silent_reference()).action != INTERRUPT, heard


def test_a_real_word_still_interrupts() -> None:
    """The guard must not swallow a genuine interruption: "open the library" always stops her."""
    assert _drive(lambda: ("open the library", "open the library"), _silent_reference()).action == INTERRUPT


def test_a_bare_stop_word_still_interrupts() -> None:
    assert _drive(lambda: ("stop", "stop"), _silent_reference()).action == INTERRUPT
