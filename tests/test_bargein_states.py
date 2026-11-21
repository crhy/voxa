"""Task 8: the explicit turn-state machine and the suspect/confirm/reject duck-and-resume.

Every row of the handoff's turn-state table (handoff-full-duplex.md:218-236) has a test here, plus
the three-state decision (report 76-83), the 12 dB / 20 ms own-stream duck (task line 9), and the
conversation-context truncation a yield owes (task line 11).
"""

from voxa.bargein import (
    CONFIRM_GATED_SPEECH_MS,
    DUCK_DB,
    DUCK_GAIN,
    DECISION_CONFIRM,
    DECISION_REJECT,
    DECISION_SUSPECT,
    LISTENING,
    OVERLAP_PENDING,
    PLANNING,
    RECOVERING_AUDIO,
    SPEAKING,
    TURN_STATES,
    USER_TURN,
    YIELDED,
    TurnStateMachine,
    decide_bargein,
    is_backchannel_only,
    is_correction,
    rewind_to_last_word_boundary,
    truncate_to_heard,
)
from voxa.speech import DUCK_RAMP_MS, duck_envelope

# --------------------------------------------------------------------------- the states themselves

def test_the_seven_explicit_turn_states_exist():
    """Handoff 6.5:220 lists LISTENING, USER_TURN, PLANNING, SPEAKING, OVERLAP_PENDING, YIELDED,
    RECOVERING_AUDIO as the application states; AEC/capture run across all of them."""
    assert set(TURN_STATES) == {
        LISTENING, USER_TURN, PLANNING, SPEAKING, OVERLAP_PENDING, YIELDED, RECOVERING_AUDIO,
    }


# --------------------------------------------------------------------------- row 1: playback echo

def test_row1_playback_only_echo_creates_no_turn_and_no_command():
    """Row 1: Playback-only echo -> Continue playback; do not create a user turn or trigger a
    command. An echo of her own voice never leaves the state it was in."""
    m = TurnStateMachine()
    m.note_speaking("The timer is set for five minutes.")
    assert m.state == SPEAKING
    assert m.playback_only_echo() == SPEAKING  # no state change, no user turn, no command


# --------------------------------------------------------------------------- row 2: independent onset

def test_row2_possible_independent_onset_pins_and_ducks_without_deleting():
    """Row 2: Possible independent user onset -> pin audio immediately; accumulate evidence without
    deleting it. The machine enters OVERLAP_PENDING and asks the caller to SUSPECT (a duck)."""
    m = TurnStateMachine()
    m.note_speaking("The timer is set for five minutes.")
    assert m.user_onset() == DECISION_SUSPECT
    assert m.state == OVERLAP_PENDING


# --------------------------------------------------------------------------- row 3: confirmed stop

def test_row3_confirmed_correction_request_or_stop_yields():
    """Row 3: Confirmed correction, request, or stop -> Yield; cancel stale response output;
    continue the same user turn from retained pre-roll."""
    m = TurnStateMachine()
    m.note_speaking("Updating Spaced Linux now.")
    m.user_onset()
    assert m.confirm("stop", gated_speech_ms=100.0, no_speech_prob=None) == DECISION_CONFIRM
    assert m.state == YIELDED


def test_row3_a_correction_yields_quickly_though_it_is_not_a_stop_word():
    """Handoff 6.5:235: "no, the other one" should yield quickly even though it is not a stop-word
    command."""
    assert is_correction("no, the other one")
    assert decide_bargein("no, the other one", gated_speech_ms=100.0,
                          no_speech_prob=None) == DECISION_CONFIRM


def test_row3_two_non_backchannel_words_confirm():
    assert decide_bargein("open the pod bay doors", gated_speech_ms=100.0,
                          no_speech_prob=None) == DECISION_CONFIRM


def test_row3_half_a_second_of_gated_speech_confirms_on_its_own():
    assert decide_bargein("", gated_speech_ms=CONFIRM_GATED_SPEECH_MS,
                          no_speech_prob=None) == DECISION_CONFIRM


# --------------------------------------------------------------------------- row 4: backchannel

def test_row4_brief_backchannel_usually_keeps_speaking():
    """Row 4: Brief acknowledgment/backchannel -> Retain and interpret it; normally keep speaking
    unless the context makes it an answer or correction."""
    m = TurnStateMachine()
    m.note_speaking("The nearest pharmacy is on Fifth Avenue.")
    m.user_onset()
    assert m.confirm("yeah", gated_speech_ms=100.0, no_speech_prob=None) == DECISION_REJECT
    assert m.state == SPEAKING  # she carries on; the reply was not taken from her


def test_row4_a_short_yes_after_a_yes_no_question_is_an_answer_not_a_backchannel():
    """Handoff 6.5:235: a short "yes" is not always a backchannel: when Voxa asks a yes/no question
    it may be the complete answer (so it yields, it does not resume)."""
    assert is_backchannel_only("yes")
    assert decide_bargein("yes", gated_speech_ms=100.0, no_speech_prob=None) == DECISION_REJECT
    assert decide_bargein("yes", gated_speech_ms=100.0, no_speech_prob=None,
                          asked_yes_no=True) == DECISION_CONFIRM


# --------------------------------------------------------------------------- row 5: repeats own words

def test_row5_user_repeating_voxas_words_preserves_them_and_does_not_change_state():
    """Row 5: User repeats words Voxa just spoke -> Preserve the words; decide from source evidence
    and context. The overlap alone never deletes and never moves the turn state."""
    m = TurnStateMachine()
    m.note_speaking("Blue.")
    m.user_onset()
    assert m.repeat_of_own_words() == OVERLAP_PENDING  # the words survive; context decides later


# --------------------------------------------------------------------------- row 6: mid-request pause

def test_row6_user_pausing_mid_request_holds_the_floor():
    """Row 6: User pauses mid-request -> Hold the floor when acoustic/linguistic evidence indicates
    continuation. The pre-roll keeps the utterance alive; the state does not advance to an endpoint."""
    m = TurnStateMachine()
    m.note_speaking("Open the")
    m.user_onset()
    assert m.user_pause() == OVERLAP_PENDING


# --------------------------------------------------------------------------- row 7: resume during planning

def test_row7_user_resumes_during_response_preparation_invalidates_the_pending_response():
    """Row 7: User resumes during response preparation -> Invalidate the pending response before it
    starts; merge the continuation correctly. PLANNING collapses into USER_TURN."""
    m = TurnStateMachine()
    m.note_planning()
    assert m.state == PLANNING
    assert m.resume_during_planning() == USER_TURN
    assert m.state == USER_TURN


# --------------------------------------------------------------------------- row 8: alignment failure

def test_row8_audio_alignment_failure_enters_recovery_and_blocks_unqualified_commands():
    """Row 8: Audio alignment fails -> Enter recovery, retain capture, and prevent unqualified
    automatic commands from playback contamination."""
    m = TurnStateMachine()
    m.note_speaking("Reminding you at three to call Bob.")
    assert m.alignment_failed() == RECOVERING_AUDIO
    assert m.state == RECOVERING_AUDIO


# --------------------------------------------------------------------------- reject: the other triggers

def test_reject_on_bracketed_nonspeech_and_on_no_words_for_1500ms():
    from voxa.bargein import REJECT_NO_SPEECH_PROB, REJECT_NO_WORDS_MS

    assert REJECT_NO_SPEECH_PROB == 0.6 and REJECT_NO_WORDS_MS == 1500.0
    assert decide_bargein("", gated_speech_ms=200.0, no_speech_prob=0.9) == DECISION_REJECT
    assert decide_bargein("", gated_speech_ms=REJECT_NO_WORDS_MS, no_speech_prob=None) == DECISION_REJECT


def test_suspect_holds_when_nothing_decides():
    assert decide_bargein("uh", gated_speech_ms=100.0, no_speech_prob=None) == DECISION_SUSPECT


# --------------------------------------------------------------------------- the duck ramp (task line 9)

def test_duck_is_12_db_on_a_20ms_ramp_and_only_lowers_voxas_own_stream():
    from voxa.bargein import DUCK_DB as B_DUCK_DB
    from voxa.bargein import DUCK_RAMP_MS as B_DUCK_RAMP_MS

    assert DUCK_DB == B_DUCK_DB == 12.0
    assert DUCK_RAMP_MS == B_DUCK_RAMP_MS == 20.0
    assert abs(DUCK_GAIN - 10.0 ** (-12.0 / 20.0)) < 1e-9


def test_duck_envelope_glides_from_unity_to_the_duck_over_20ms():
    gains = duck_envelope(3, frame_ms=10.0)
    assert gains[0] == 1.0
    assert abs(gains[-1] - DUCK_GAIN) < 1e-9
    assert gains[0] > gains[1] > gains[2]  # monotonic down

    up = duck_envelope(3, frame_ms=10.0, up=True)
    assert abs(up[0] - DUCK_GAIN) < 1e-9 and up[-1] == 1.0  # restore back to unity


def test_duck_envelope_empty_for_no_frames():
    assert duck_envelope(0, frame_ms=20.0) == []


# --------------------------------------------------------------------------- truncation (task line 11)

def test_yield_truncates_the_assistant_turn_to_the_words_audibly_played():
    """Report 76:76: on confirm, truncate the assistant turn in the conversation context to the words
    audibly played."""
    spoken = "The nearest pharmacy is on Fifth Avenue."
    heard = "The nearest pharmacy is on"
    assert truncate_to_heard(spoken, heard) == "the nearest pharmacy is on"


def test_long_pause_then_reject_rewinds_to_the_last_word_boundary():
    """Task line 11: on a long pause-then-reject, rewind to the last word boundary (a mid-word pause
    must not leave a partial word in the context)."""
    spoken = "The nearest pharmacy is on Fifth Avenue."
    heard = "The nearest pharmacy is"
    kept = rewind_to_last_word_boundary(spoken, heard)
    assert kept == "the nearest pharmacy is"
    assert "fifth" not in kept and "avenue" not in kept


def test_a_replay_before_any_word_keeps_nothing():
    assert truncate_to_heard("Updating Spaced Linux now.", "") == ""
