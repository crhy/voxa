"""f172 (roadmap task 5): the joint double-talk gate inside the barge-in decision.

What the barge controller now owes the joint gate, and nothing else:

* the gate's MIRROR verdict closes a candidate the fallback level gate already opened - the
  span it is collecting is her echo, and closing keeps the audio in the capture ring
  (evidence, never deletion);
* the fast onset yield is reference-aware: when the gate MEASURED the frame and did not raise
  the suspect, the microphone's tower over the speaker path is not enough to take the floor -
  the level gate and the words carry on, and the suspect may still raise the yield later;
* a candidate that heard NO words is never "user" on the level tower alone: there are no words
  to be unlike her sentence, so the tower abstains and the decision is at most uncertain.
"""

from __future__ import annotations

from voxa.attribution import USER, SourceEvidence, attribute
from voxa.bargectl import IGNORED, YIELD, BargeInController

# The speaker path's level, and the microphone's level while it carries ONLY that echo: the
# first frame anchors the onset detector's coupling at 0 dB, after which the microphone's dB
# can be compared with the ceiling the speaker path implies.
_REFERENCE_DB = -30.0
_ECHOED_MIC_DB = -30.0
_TOWERING_MIC_DB = -10.0  # 20 dB over the speaker path's ceiling: above UNEXPLAINED_DB


def _ev(**fields) -> SourceEvidence:
    base = dict(reference_active=True, alignment_healthy=True,
                echo_explained_db=-1.0, vad_prob=0.2)
    base.update(fields)
    return SourceEvidence(**base)


def _anchor(ctl: BargeInController, evidence: SourceEvidence) -> None:
    """One frame where the microphone carries exactly the echo: the coupling anchor.

    Without it the onset detector has no common point between the two dB scales, and "the
    microphone towers over the reference" is a number nobody measured.
    """
    ctl.observe(0.05, speaking=True, muted=True, evidence=evidence,
                mic_db=_ECHOED_MIC_DB, reference_db=_REFERENCE_DB,
                delay_ms=200.0, dt_ms=20.0)


def _drive_onset(ctl: BargeInController, evidence: SourceEvidence,
                 frames: int = 20) -> list:
    """Anchor, then hold the microphone over the speaker path's ceiling for a full evidence run.

    20 frames x 20 ms of audio: ONSET_EVIDENCE_MS is 320 ms, so the onset confirms mid-run -
    every frame from there on carries the onset claim, and the run's outcomes are the evidence.
    """
    outcomes = []
    for _ in range(frames):
        outcomes.append(ctl.observe(0.05, speaking=True, muted=True, evidence=evidence,
                                    mic_db=_TOWERING_MIC_DB, reference_db=_REFERENCE_DB,
                                    delay_ms=200.0, dt_ms=20.0))
    return outcomes


def test_the_mirror_verdict_closes_an_opened_candidate() -> None:
    """The level gate opened on the tower; the joint gate then MEASURED echo-consistency:
    the candidate is dropped, the stats say so, and the capture ring keeps the audio."""
    ctl = BargeInController()
    ctl.reply_started("The weather is clear today.")
    for _ in range(40):  # 800 ms of her own level: the gate hears only after its 400 ms
        # onset window, then learns for 400 ms more before it may open at all
        ctl.observe(0.05, speaking=True, muted=True, evidence=_ev(),
                    mic_db=_ECHOED_MIC_DB, reference_db=_REFERENCE_DB,
                    delay_ms=200.0, dt_ms=20.0)
    opened = None
    for _ in range(6):  # the tower: 80 ms of a level over her own, inside the 160 ms window
        opened = ctl.observe(0.5, speaking=True, muted=True, evidence=_ev(),
                             mic_db=_ECHOED_MIC_DB, reference_db=_REFERENCE_DB,
                             delay_ms=200.0, dt_ms=20.0)
    assert opened.action == IGNORED and opened.reason == "listening"
    assert ctl.pending_since > 0.0, "the fallback level gate never opened"
    closed = ctl.observe(0.5, speaking=True, muted=True,
                         evidence=_ev(joint_suspect=False, joint_echo=True),
                         mic_db=_ECHOED_MIC_DB, reference_db=_REFERENCE_DB,
                         delay_ms=200.0, dt_ms=20.0)
    assert closed.action == IGNORED and closed.reason == "gate-closed"
    assert ctl.pending_since == 0.0
    assert ctl.stats["joint_closed"] >= 1
    assert ctl.stats["joint_closed_candidate"] == 1


def test_the_onset_yield_waits_for_the_joint_gate_when_it_has_measured() -> None:
    """The onset rule's ceiling is exceeded for a full evidence run, but the joint gate
    measured the frame and found NO second source: the floor stays, visibly blocked."""
    ctl = BargeInController()
    ctl.reply_started("The weather is clear today.")
    _anchor(ctl, _ev(joint_suspect=False))
    outcomes = _drive_onset(ctl, _ev(joint_suspect=False))
    assert all(o.action != YIELD for o in outcomes), \
        "the onset yielded while the joint gate said not-suspect"
    assert ctl.stats.get("onset_blocked", 0) >= 1
    assert ctl.pending_since == 0.0, "the blocked onset still opened a candidate"


def test_the_onset_yield_fires_when_the_joint_gate_agrees() -> None:
    """The same onset, with the joint gate's suspect raised: the fast yield is unchanged."""
    ctl = BargeInController()
    ctl.reply_started("The weather is clear today.")
    _anchor(ctl, _ev(joint_suspect=True))
    outcomes = _drive_onset(ctl, _ev(joint_suspect=True))
    assert any(o.action == YIELD for o in outcomes)


def test_the_onset_yield_is_unchanged_when_the_gate_cannot_measure() -> None:
    """joint_suspect None (the gate never measured this frame): the degraded path keeps the
    fast yield - "the gate did not measure" is never evidence of echo."""
    ctl = BargeInController()
    ctl.reply_started("The weather is clear today.")
    _anchor(ctl, _ev())
    outcomes = _drive_onset(ctl, _ev())
    assert any(o.action == YIELD for o in outcomes)


def test_an_empty_candidate_is_never_user_on_the_level_tower_alone() -> None:
    """f172: the tower used to half-count against UNKNOWN_DISSIMILARITY with no words at all,
    which with the unexplained residual reached "user" and interrupted her on a candidate
    that heard nothing."""
    ev = SourceEvidence(reference_active=True, alignment_healthy=True,
                        echo_explained_db=-20.0, vad_prob=0.2,
                        level_ratio=3.0, word_count=0, text_letters_measurable=False)
    assert attribute(ev).source != USER
    # A real one-word answer keeps the tower's half-weight: it is not an empty span.
    ev.word_count = 1
    assert attribute(ev).source == USER


def test_the_echo_mirror_never_drops_a_candidate_the_suspect_opened() -> None:
    """f187: a person interrupting over her echo holds coherent frames between her words -
    the gate's mirror latched and DROPPED the candidate the suspect itself had opened, and
    the frozen last-word clip missed its barge-in. Once this reply raised a suspect, the
    mirror verdict may not drop the candidate again (the mirror still vetoes the fallback
    level gate on replies that never raised one)."""
    ctl = BargeInController()
    ctl.reply_started("The weather is clear today.")
    for _ in range(40):  # her own level: teach the gate, no openings
        ctl.observe(0.05, speaking=True, muted=True, evidence=_ev(),
                    mic_db=_ECHOED_MIC_DB, reference_db=_REFERENCE_DB,
                    delay_ms=200.0, dt_ms=20.0)
    opened = ctl.observe(0.3, speaking=True, muted=True,
                         evidence=_ev(joint_suspect=True, joint_echo=False),
                         mic_db=_ECHOED_MIC_DB, reference_db=_REFERENCE_DB,
                         delay_ms=200.0, dt_ms=20.0)
    assert opened.gate_opened and ctl.pending_since > 0.0, "the suspect never opened"
    outcome = ctl.observe(0.3, speaking=True, muted=True,
                          evidence=_ev(joint_suspect=False, joint_echo=True),
                          mic_db=_ECHOED_MIC_DB, reference_db=_REFERENCE_DB,
                          delay_ms=200.0, dt_ms=20.0)
    assert outcome.action == IGNORED and outcome.reason == "listening"
    assert ctl.pending_since > 0.0, "the mirror dropped a candidate the suspect had opened"
    assert ctl.stats.get("joint_closed_candidate", 0) == 0
