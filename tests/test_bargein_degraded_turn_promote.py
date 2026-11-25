"""Issue #72, task v11 step 3: an abstain is not permanent.

When the REFERENCE is unusable (the live tap reads silent while she speaks, or the alignment is
unhealthy) attribution can only ever hold the span back, so in the rc5 live run NOTHING said over
Voxa could become ``user`` and there were 0 INTERRUPTs. While she speaks and the reference is
unusable, the controller must decide by the turn-wide text veto: heard CONTENT words that are not
in her whole turn are the user, so they INTERRUPT; her own words (an echo misheard by Whisper)
still never interrupt.
"""

import pytest

from voxa.attribution import SourceEvidence
from voxa.bargectl import (DEGRADED_FOREIGN_MIN, BargeInController, HOLD, INTERRUPT)


def _drive(ctl, heard, raw, *, reference_active, alignment_healthy=1.0, clock=None):
    """Teach the gate her quiet reply, then run the loud candidate the words arrived in.

    Mirrors tools/self_hearing_replay.py: the reference is (un)usable exactly as the live tap read
    it, alignment healthy or not, and the mic towers over her level (the line exists because the
    gate opened). No VAD number is invented. The clock advances with the audio, exactly as the
    frame clock hands milliseconds to ``observe``.
    """
    # Settle the level gate on her own quiet voice first.
    for _ in range(30):
        if clock is not None:
            clock["t"] += 0.1
        ctl.observe(0.05, speaking=True, muted=True, dt_ms=100.0)
    ev = SourceEvidence(reference_active=reference_active, speaking=True,
                        alignment_healthy=alignment_healthy, mic_level=0.9, level_ratio=4.0,
                        echo_explained_db=(0.0 if not alignment_healthy else None))

    def evidence():
        return ev

    actions = []
    for _ in range(24):
        if clock is not None:
            clock["t"] += 0.1
        outcome = ctl.observe(0.9, speaking=True, muted=True, dt_ms=100.0,
                              candidate=lambda: (heard, raw), evidence=evidence())
        actions.append(outcome.action)
    return actions


@pytest.fixture
def controller():
    clock = {"t": 1000.0}

    def now() -> float:
        return clock["t"]

    ctl = BargeInController(clock=now)
    ctl.note_turn("The capital of France is Paris.")
    ctl.reply_started("The capital of France is Paris.")
    return ctl, clock


def test_foreign_words_over_silent_tap_interrupt(controller):
    ctl, clock = controller
    # The exact rc5 failure: a real question over her reply, the live tap silent.
    actions = _drive(ctl,
                     "How do I make banana bread from scratch?",
                     "How do I make banana bread from scratch?",
                     reference_active=False, clock=clock)
    assert INTERRUPT in actions, f"expected an INTERRUPT, got {sorted(set(actions))}"
    assert HOLD not in actions or actions.count(INTERRUPT) >= 1


def test_wake_plus_two_foreign_interrupt(controller):
    ctl, clock = controller
    # "Voxa, make coffee": wake word plus two words that are not hers.
    actions = _drive(ctl, "Voxa, make coffee", "Voxa, make coffee",
                     reference_active=False, clock=clock)
    assert INTERRUPT in actions


def test_her_own_echo_never_interrupts(controller):
    ctl, clock = controller
    # A Whisper mishearing of her own reply (foreign words under the bar) must not interrupt.
    actions = _drive(ctl,
                     "the capital of France is Paries", "the capital of France is Paries",
                     reference_active=False, clock=clock)
    assert INTERRUPT not in actions
    assert HOLD in actions or "own-voice" in actions


def test_unhealthy_alignment_also_promotes(controller):
    ctl, clock = controller
    actions = _drive(ctl,
                     "Tell me a joke about a programmer",
                     "Tell me a joke about a programmer",
                     reference_active=True, alignment_healthy=False, clock=clock)
    assert INTERRUPT in actions


def test_threshold_is_three_words():
    assert DEGRADED_FOREIGN_MIN == 3
