"""Audio-based attribution (task A2, handoff 6.3).

The PAIRS are the point: the SAME words, heard twice - once with Voxa speaking alone (the echo
path explains the microphone and leaves its usual residual), once with a person talking over
her (the microphone holds energy the delayed reference cannot explain). A rule that decides
both the same way is not an attribution rule, and the bench's `pair_accuracy` metric says so.
"""

import pytest

from voxa.attribution import (ECHO, MIN_REFERENCE_POWER, UNCERTAIN, USER, EchoPathBaseline,
                              SourceDecision, SourceEvidence, attribute)
from voxa.bargein import source_decision
from voxa.ownspeech import OwnSpeech


def _alone(**overrides) -> SourceEvidence:
    """Voxa speaking alone: the delayed reference explains the microphone, nothing is left over."""
    ev = SourceEvidence(reference_active=True, reference_power=4e-3, echo_explained_db=-0.4,
                        residual_db=-34.0, residual_margin_db=-0.5, vad_prob=0.10,
                        alignment_healthy=True, level_ratio=1.1)
    for key, value in overrides.items():
        setattr(ev, key, value)
    return ev


def _over_her(**overrides) -> SourceEvidence:
    """A person over her reply: the echo path cannot explain part of the microphone."""
    ev = SourceEvidence(reference_active=True, reference_power=4e-3, echo_explained_db=-14.0,
                        residual_db=-21.0, residual_margin_db=9.0, vad_prob=0.93,
                        alignment_healthy=True, level_ratio=2.6)
    for key, value in overrides.items():
        setattr(ev, key, value)
    return ev


# ------------------------------------------------------------------ the pairs


@pytest.mark.parametrize("heard", ["yes", "blue", "stop", "that one", "wait"])
def test_a_brief_answer_is_hers_alone_and_the_speaker_over_her(heard):
    speaking = "The weather is clear today."
    alone = source_decision(heard, speaking, evidence=_alone())
    over = source_decision(heard, speaking, evidence=_over_her())
    assert alone.source != USER, (heard, alone.reasons)
    assert over.source == USER, (heard, over.reasons)


def test_a_stop_word_said_by_herself_never_stops_her_and_said_by_a_person_always_does():
    # The task's step 4, in both directions, with the SAME words and the SAME sentence.
    speaking = "Please stop the car now."
    assert source_decision("stop", speaking, evidence=_alone()).source != USER
    assert source_decision("stop", speaking, evidence=_over_her()).source == USER


def test_the_floor_the_echo_path_leaves_is_a_fully_explained_microphone_not_a_tower():
    """Issue #72, task v11d (Claude's 18:35 audit, live barge.log): while she spoke ALONE, the
    microphone at -120 dB (nothing left beyond her delayed reference) read as "microphone holds
    -120.0 dB beyond the reference" and added its +0.5 to the USER - her own garbled echo
    interrupted her. A floor-level microphone is fully explained by the reference: the STRONGEST
    echo evidence, never a tower, and the same words over a microphone that really holds +dB
    beyond the reference still say USER."""
    speaking = "The first sentence is long enough that Voxa will still be speaking on the chessboard."
    heard = "The first sentence in the long enough that butter will still be speaking"
    floor = _alone(echo_explained_db=-120.0, residual_db=-120.0, residual_margin_db=18.1,
                   speaking=True, vad_prob=0.2)
    decision = source_decision(heard, speaking, evidence=floor)
    assert decision.source != USER, decision.reasons
    # The same words over a microphone that stands well over the reference stay a second source.
    tower = _over_her(echo_explained_db=-14.0, residual_margin_db=9.0, vad_prob=0.9, speaking=True)
    assert source_decision(heard, speaking, evidence=tower).source == USER


def test_the_words_are_a_feature_and_cannot_veto_the_audio():
    # The user repeats a whole clause of her reply while she is reading it: the words look
    # exactly like her sentence, and the audio still says a second source is present.
    speaking = "Reminding you at three to call Bob."
    heard = "call Bob at three"
    assert source_decision(heard, speaking, evidence=_over_her()).source == USER
    # And the reverse: words that look nothing like her sentence, with audio that says echo.
    assert source_decision("quantum flux capacitor", speaking, evidence=_alone()).source != USER


def test_an_exact_repetition_of_her_own_sentence_is_the_speaker_when_the_audio_says_so():
    # The hardest pair in the set (bench clip smoke-0033/0032-pair): the user says the WHOLE
    # sentence, word for word, while she is reading it. The words are identical on both sides,
    # so only the audio can tell them apart - and this is the case a text rule gets wrong by
    # construction.
    sentence = "The library opens at nine in the morning."
    alone = source_decision(sentence, sentence, evidence=_alone())
    assert alone.source != USER, alone.reasons
    over = source_decision(sentence, sentence, evidence=_over_her())
    assert over.source == USER, over.reasons


def test_the_number_of_words_decides_nothing():
    # Same audio, same sentence: a one-word answer and a whole sentence are both the user's.
    speaking = "The weather is clear today."
    for heard in ("yes", "open the calculator and read the note aloud please"):
        assert source_decision(heard, speaking, evidence=_over_her()).source == USER, heard


# ------------------------------------------------------------------ degraded and unmeasurable


def test_evidence_that_was_never_measured_is_uncertain_not_a_guess():
    decision = attribute(SourceEvidence(reference_active=True))
    assert decision.source == UNCERTAIN
    assert decision.degraded is True
    assert any("no VAD" in reason for reason in decision.reasons)


def test_an_echo_path_that_was_never_fitted_qualifies_nothing():
    # Handoff 6.8: an unqualified delay means the audio is unqualified - a visible degraded state,
    # not a decision, and never a command taken from playback contamination.
    decision = attribute(_alone(alignment_healthy=False))
    assert decision.source == UNCERTAIN
    assert decision.degraded is True
    assert any("alignment unhealthy" in reason for reason in decision.reasons)


def test_a_silent_reference_leaves_nothing_to_cancel():
    # Nothing is coming out of the speakers (near-end-only speech, or Voxa silent): any speech in
    # the microphone is the user's, and no text match may say otherwise.
    decision = attribute(SourceEvidence(reference_active=False, vad_prob=0.8,
                                         text_overlap=1.0, text_sound=1.0))
    assert decision.source == USER


def test_speaking_with_a_silent_tap_never_credits_the_microphone_to_the_user():
    # The self-interruption of issue #72: Voxa is mid-reply, the reference tap measures silence
    # (the missing reference of roadmap task 1), and the microphone carries her own voice. The
    # reference is active by definition, so attribution may NOT answer ``user`` here - that is
    # what made her answer herself. It abstains (degraded) and lets the words/joint gate decide.
    decision = attribute(SourceEvidence(reference_active=False, speaking=True, vad_prob=0.8,
                                         mic_level=0.5, text_overlap=1.0, text_sound=1.0))
    assert decision.source != USER, decision.reasons
    assert decision.degraded is True
    assert any("by definition while speaking" in reason for reason in decision.reasons)


def test_one_crude_signal_alone_does_not_make_a_person():
    # The microphone towers over her level, and that is all the evidence says: the echo path is
    # fitted (so its streams count) but leaves nothing of its own behind.
    decision = attribute(_alone(echo_explained_db=None, residual_margin_db=3.0,
                                vad_prob=None, level_ratio=3.0))
    assert decision.source == UNCERTAIN
    assert any("only one stream" in reason for reason in decision.reasons)


def test_a_path_with_no_echo_model_reports_itself_degraded():
    # No reference sink on this machine: the words carry the decision, and the decision says so
    # (roadmap task 1 hands the reference tap back; until then this is a visible degraded state).
    decision = source_decision("quarterly report", "Please continue with the quarterly report")
    assert decision.source == ECHO
    assert decision.degraded is True
    assert any("no echo model" in reason for reason in decision.reasons)


# ------------------------------------------------------------------ the echo path's own baseline


def test_the_echo_path_baseline_compares_like_with_like():
    # A residual only counts against what THIS path usually leaves behind: a path that always
    # leaves -20 dB is not evidence of a person, and a step above its own level is.
    baseline = EchoPathBaseline()
    assert baseline.note(-20.0) is None
    for _ in range(4):
        baseline.note(-20.0)
    assert baseline.note(-20.0) == pytest.approx(0.0)
    assert baseline.note(-11.0) > 6.0
    assert attribute(_alone(residual_margin_db=1.0, echo_explained_db=None,
                            vad_prob=None)).source != USER


def test_the_reference_threshold_is_the_one_the_echo_path_uses():
    from voxa.anc import MIN_REFERENCE_POWER as ANC_MIN_REFERENCE_POWER

    assert MIN_REFERENCE_POWER == ANC_MIN_REFERENCE_POWER


def test_attribution_touches_no_audio_and_no_gtk():
    # Handoff 6.8: the GTK thread may READ these numbers, it must never run a model or touch a
    # device. attribution.py is pure arithmetic, and this test keeps it that way.
    source = open(__file__.replace("tests/test_attribution.py", "voxa/attribution.py")).read()
    for forbidden in ("import gi", "gi.require", "numpy", "subprocess", "whisper", "GStreamer"):
        assert forbidden not in source, forbidden


# ------------------------------------------------------------------ the filter and the controller


def open_the_gate(control, clock, heard: str) -> float:
    """Teach the gate her own level, then hand it the loud run that opens it (as the live app
    and the bench do: the baseline must have heard her voice before a loud sound means anything)."""
    clock["now"] = 0.5
    for _ in range(40):
        clock["now"] += 0.05
        control.observe(0.15, speaking=True, muted=True, wake_word="voxa",
                        candidate=lambda: (heard, heard), dt_ms=50.0)
    for _ in range(40):
        clock["now"] += 0.05
        outcome = control.observe(0.6, speaking=True, muted=True, wake_word="voxa",
                                  candidate=lambda: (heard, heard), dt_ms=50.0)
        if outcome.reason == "listening" and outcome.gate_opened:
            return clock["now"]
    raise AssertionError("the level gate never opened")


def judge(control, clock, heard: str, level: float = 0.6, **kwargs):
    """Keep feeding AUDIO until the collected words are judged.

    The listening window is a duration of sound (``BARGE_IN_LISTEN_MS``), not a number of
    callbacks and not wall-clock time, so the helper advances the simulated clock by the same
    50 ms each sample says it stands for.
    """
    for _ in range(40):
        clock["now"] += 0.05
        outcome = control.observe(level, speaking=True, muted=True, wake_word="voxa",
                                  candidate=lambda: (heard, heard), dt_ms=50.0, **kwargs)
        if outcome.action != "ignored":
            return outcome
    return None


def test_the_own_speech_filter_keeps_a_user_who_repeats_her_words():
    speech = OwnSpeech(clock=lambda: 0.0)
    speech.started("The weather is clear today.")
    kept, share = speech.strip("the weather is clear", evidence=None, confirmed_user=False)
    # No audio evidence on this path: the degraded case, which still filters her echo.
    assert kept != "the weather is clear", "the degraded path still strips echo"
    assert share > 0.5, "the overlap is reported whether or not anything was removed"

    user = SourceDecision(USER, 0.9, ("residual +9.0 dB",))
    assert speech.strip("the weather is clear", evidence=user)[0] == "the weather is clear"
    assert speech.is_echo("the weather is clear today", evidence=user) is False
    # A span already confirmed as the user's is never rewritten, whatever the words look like.
    assert speech.strip("the weather is clear", confirmed_user=True)[0] == "the weather is clear"
    # Her own echo, with audio that says echo, is still filtered.
    echo = SourceDecision(ECHO, 0.9, ("microphone explained by reference -0.4 dB",))
    assert speech.strip("the weather is clear", evidence=echo)[0] == ""


def test_the_controller_holds_when_the_audio_cannot_decide():
    from voxa.bargectl import HOLD, BargeInController

    clock = {"now": 0.0}
    control = BargeInController(clock=lambda: clock["now"])
    control.reply_started("The weather is clear today.")
    undecided = SourceEvidence(reference_active=True, alignment_healthy=False)
    open_the_gate(control, clock, "yes")
    held = judge(control, clock, "yes", evidence=undecided)
    assert held.action == HOLD, held.reason
    assert held.source == UNCERTAIN
    assert control.pending_since == 0.0, "a held candidate does not leave the gate stuck open"


def test_the_controller_stops_her_only_on_audio_that_supports_a_second_source():
    from voxa.bargectl import INTERRUPT, OWN_VOICE, BargeInController

    for evidence, action in ((_alone(), OWN_VOICE), (_over_her(), INTERRUPT)):
        clock = {"now": 0.0}
        control = BargeInController(clock=lambda: clock["now"])
        control.reply_started("Please stop the car now.")
        open_the_gate(control, clock, "stop")
        decided = judge(control, clock, "stop", evidence=evidence)
        assert decided.action == action, (evidence.echo_explained_db, decided.reason)


def test_the_floor_of_a_silent_microphone_never_decides_user_issue_72_v11c():
    """Issue #72, task v11c step 5: between sentences of her own TTS the microphone sits at
    the floor this echo path always leaves (-120 dB sentinel), the baseline is ringed low,
    and the residual tower fires anyway. With no words and no speech in the cleaned signal
    there is NO second source in those samples - the span abstains, it is never "user"."""
    gap = _alone(speaking=True, word_count=0, vad_prob=None,
                 echo_explained_db=-120.0, residual_margin_db=18.1)
    decision = attribute(gap)
    assert decision.source == UNCERTAIN, decision.reason
    assert not decision.degraded          # honest silence, not a broken path
    assert any("floor" in reason for reason in decision.reasons), decision.reason

    # The SAME reading while a person is actually talking over the gap keeps deciding user:
    # speech in the cleaned signal, or words, and the floor guard stays out of the way.
    talking = attribute(_alone(speaking=True, word_count=2, vad_prob=0.91,
                               echo_explained_db=-120.0, residual_margin_db=18.1))
    assert talking.source == USER, talking.reason


def test_no_speech_on_the_cleaned_signal_never_decides_user_issue_72_v11c():
    """Live barge.log (12:22): INTERRUPTs decided with the words heard EMPTY on reasons
    "no VAD on the cleaned signal" / "cleaned signal not speech-like p=0.30". While she
    speaks, the audio readings (microphone holds beyond the reference, residual +13..27 dB
    over the echo path's baseline) mean the canceller is not cancelling at its delay: her
    OWN echo, not a second speaker. With no words and no speech the span is held."""
    unmeasured = _alone(speaking=True, word_count=0, vad_prob=None,
                        echo_explained_db=-4.4, residual_margin_db=13.0)
    decision = attribute(unmeasured)
    assert decision.source == UNCERTAIN, decision.reason
    assert not decision.degraded
    assert any("no speech on the cleaned signal" in r for r in decision.reasons), decision.reason

    quiet = attribute(_alone(speaking=True, word_count=0, vad_prob=0.30,
                             echo_explained_db=-5.3, residual_margin_db=26.0))
    assert quiet.source == UNCERTAIN, quiet.reason

    # A real person over her reply still decides user: speech in the cleaned signal, or
    # words, and the guard stays out of the way.
    assert attribute(_over_her(word_count=0)).source == USER
    assert attribute(_over_her(vad_prob=0.10, word_count=3)).source == USER


def test_a_parked_delay_never_carries_a_second_source_alone_issue_72_v11d():
    """Live 22-char v11d run: the exact aligner NEVER confirms (the lock's coherence gate is
    built for the bench's strong echo), so the travel sits PARKED from the pilot chirp - a
    good guess, an unproven one - and a garbled mistranscription of her OWN TTS ("The bell
    station" for "Voxa will still be speaking") interrupted her on "microphone holds
    -5.6 dB beyond the reference" + "residual +23.9 dB over the echo path's baseline" +
    "microphone 1.8x her own level, words 57% unlike". With the delay unproven those two
    echo-path readings describe the echo path's own unconfirmed delay, not a second source:
    the span abstains. With a proved delay the SAME readings decide as before."""
    fields = dict(reference_active=True, reference_power=4e-3, speaking=True,
                  echo_explained_db=-5.6, residual_db=-47.1, residual_margin_db=23.9,
                  vad_prob=None, alignment_healthy=True, level_ratio=1.8,
                  word_count=7, text_overlap=0.2, text_sound=0.43)
    parked = attribute(SourceEvidence(delay_proven=False, **fields))
    assert parked.source != USER, parked.reasons
    assert any("unproven" in r for r in parked.reasons), parked.reasons

    proved = attribute(SourceEvidence(delay_proven=True, **fields))
    assert proved.source == USER, proved.reasons

    # The level tower alone (arabic: 2.0x, words 100% unlike, no VAD measured) also abstains
    # while the delay is unproven.
    tower = attribute(SourceEvidence(delay_proven=False, reference_active=True,
                                     reference_power=4e-3, speaking=True,
                                     echo_explained_db=-4.9, residual_db=-52.0,
                                     residual_margin_db=23.2, vad_prob=None,
                                     alignment_healthy=True, level_ratio=2.0,
                                     word_count=8, text_overlap=0.0, text_sound=0.0))
    assert tower.source != USER, tower.reasons

    # A real person talking over her reply still interrupts while the delay is unproven:
    # speech measured in the CLEANED signal is an independent source whatever the delay is.
    talk = attribute(SourceEvidence(delay_proven=False, reference_active=True,
                                    reference_power=4e-3, speaking=True,
                                    echo_explained_db=-14.0, residual_db=-21.0,
                                    residual_margin_db=9.0, vad_prob=0.93,
                                    alignment_healthy=True, level_ratio=2.6,
                                    word_count=2, text_overlap=0.0, text_sound=0.0))
    assert talk.source == USER, talk.reasons


def test_the_echo_arrival_gap_is_not_reported_as_a_silent_tap_issue_72_v11d():
    """aiyana live (22-char v11d): 0.26 s into the reply a candidate span was captured
    before her first sample could echo, read silent, and the live check condemned the tap -
    the render-tap line of the same reply reported peak 0.68. A live tap abstains without
    the words the reference check reads as a silent tap; a dead tap still says them."""
    gap = attribute(_alone(speaking=True, reference_active=False, reference_power=0.0,
                           vad_prob=None, word_count=6, tap_live=True))
    assert gap.source == UNCERTAIN, gap.reasons
    assert gap.degraded
    assert not any("silent tap" in r for r in gap.reasons), gap.reasons

    dead = attribute(_alone(speaking=True, reference_active=False, reference_power=0.0,
                            vad_prob=None, word_count=6, tap_live=False))
    assert dead.source == UNCERTAIN, dead.reasons
    assert any("silent tap" in r for r in dead.reasons), dead.reasons
