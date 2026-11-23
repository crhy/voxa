from __future__ import annotations

from voxa.endpoint import (EndpointPolicy, SmartTurn, arbitrate_endpoint, ends_dangling, is_complete,
                           is_followup, patience_factor)

COMPLETE = [
    "voxa open gmail",
    "close brutal chess",
    "turn it up",
    "start dictating",
    "cancel",
    "open libra office right",
    "open gmail",
    "go back",
    "read this page",
    "mute",
    "louder",
    "google cats",
    "search the web for weather",
    "type hello there",
    "close the window",
    "switch to gimp",
    "next",
    "never mind",
    "go offline",
    "stop",
    "take dictation",
    "what time is it",
]

NOT_COMPLETE = [
    "",
    "voxa",
    "turn off the",
    "play some",
    "play some jazz",
    "open",
    "what is the",
    "open gmail and then",
    "tell me a joke",
    "close brutal chess and open gmail",
    "how are you",
    "the",
    "a",
    "some",
    "please",
    "then",
    "um",
    "uh",
    "turn up the",
    "play music on",
    "voxa and then",
    "hello there",
    "hey",
    "ok",
]


def test_complete_utterances() -> None:
    for text in COMPLETE:
        assert is_complete(text), text


def test_incomplete_utterances() -> None:
    for text in NOT_COMPLETE:
        assert not is_complete(text), text


def test_rows_meet_the_minimum_table_size() -> None:
    assert len(COMPLETE) + len(NOT_COMPLETE) >= 30


def test_custom_wake_word_alone_is_not_complete() -> None:
    assert not is_complete("computer", wake_word="computer")
    assert not is_complete("voxa", wake_word="voxa")
    assert is_complete("voxa open gmail", wake_word="voxa")


FOLLOWUP_TRUE = [
    "and open gmail",
    "also the window",
    "then close it",
    "now play music",
    "next",
    "again",
    "what about the weather",
    "how about jazz",
    "no not that one",
    "actually open gmail",
    "instead close it",
    "make it louder",
    "the other one",
    "another window",
    "more coffee",
    "less sugar",
    "louder please",
    "quieter now",
    "yes please",
    "yeah ok",
    "okay then",
    "what time is it",
    "where is my file",
    "can you open gmail",
    "is it raining",
    "will it rain",
    "do you know",
    "does this work",
    "are we good",
    "could you help",
    "how are you",
    "who is that",
    "when does it open",
    "why not",
    "tell me a joke?",
]

FOLLOWUP_FALSE = [
    "nice weather today",
    "the cat sat",
    "hello there",
    "see you later",
    "the",
    "a",
    "please",
    "um uh",
    "banana",
    "purple dinosaur",
    "I like coffee",
    "walk the dog",
    "green apple pie",
    "table lamp",
    "desk shade",
    "curtain rod",
    "garden gnome",
    "wooden spoon",
    "marble statue",
    "quiet library",
]


def test_followup_true_rows() -> None:
    for text in FOLLOWUP_TRUE:
        assert is_followup(text), text


def test_followup_false_rows() -> None:
    for text in FOLLOWUP_FALSE:
        assert not is_followup(text), text


def test_followup_table_meets_minimum_size() -> None:
    assert len(FOLLOWUP_TRUE) + len(FOLLOWUP_FALSE) >= 20


DANGLING_TRUE = [
    "turn off the",
    "you proposed to",
    "he passed through",
    "I want a",
    "came so",
    "play some",
    "open gmail and",
    "play music on",
    "what",
    "I'll take it from",
]

DANGLING_FALSE = [
    "open gmail",
    "turn it up",
    "what time is it",
    "she said",
    "close the window",
]


def test_ends_dangling_true_rows() -> None:
    for text in DANGLING_TRUE:
        assert ends_dangling(text), text


def test_ends_dangling_false_rows() -> None:
    for text in DANGLING_FALSE:
        assert not ends_dangling(text), text


def test_patience_factors() -> None:
    assert patience_factor("default") == 1.0
    assert patience_factor("longer") == 1.5
    assert patience_factor("longest") == 2.0
    assert patience_factor("nonsense") == 1.0
    assert patience_factor("") == 1.0


def test_endpoint_policy_timings() -> None:
    policy = EndpointPolicy("default")
    assert policy.trigger_seconds == 0.25
    assert policy.complete_seconds == 0.5
    assert policy.incomplete_wait_seconds == 2.0
    assert policy.cap_seconds == 3.0
    assert EndpointPolicy("longer").trigger_seconds == 0.375
    assert EndpointPolicy("longest").cap_seconds == 6.0
    assert EndpointPolicy("default", dictation=True).cap_seconds == 5.0
    assert EndpointPolicy("longer", dictation=True).cap_seconds == 7.5


def test_smart_turn_without_weights_gives_no_verdict(tmp_path) -> None:
    turn = SmartTurn(str(tmp_path / "missing.onnx"))
    assert not turn.available
    assert turn.probability(b"\0\0" * 1600) is None
    assert turn.is_complete(b"\0\0" * 1600) is None


def test_smart_turn_shared_returns_one_session(tmp_path) -> None:
    a = SmartTurn.shared(str(tmp_path / "missing.onnx"))
    b = SmartTurn.shared(str(tmp_path / "missing.onnx"))
    assert a is b


# --------------------------------------------------------------------------
# f178: the endpoint ARBITRATOR (issue #72). Gates a Smart Turn "complete"
# verdict on the shipped joint suspect / turn state, never on the model alone.
# --------------------------------------------------------------------------

def test_arbitrator_joint_suspect_open_waits_not_complete() -> None:
    # Joint-suspect open + a short pause: the endpointer WAITS the pause out
    # (False -> the 2.0 s incomplete floor), it does not fire complete.
    assert arbitrate_endpoint(True, joint_suspect=True) is False


def test_arbitrator_quiet_joint_gate_fires() -> None:
    # A short pause the joint gate says is quiet + a text-complete turn: fires
    # normally at the minimum silence floor.
    assert arbitrate_endpoint(True, joint_suspect=False) is True


def test_arbitrator_fragment_with_suspect_avoids_premature_cut() -> None:
    # A grammatical fragment the model calls complete but a suspect is open
    # (f136's pause-00 / dt_first_syllable-00 class): the premature cut is avoided.
    assert arbitrate_endpoint(True, joint_suspect=True) is False
    # ... and the same fragment with a quiet gate still fires.
    assert arbitrate_endpoint(True, joint_suspect=False) is True


def test_arbitrator_turn_suspect_and_more_speech_hold() -> None:
    assert arbitrate_endpoint(True, turn_suspect=True) is False
    assert arbitrate_endpoint(True, more_speech_arriving=True) is False
    # A quiet turn, no suspect, no arriving speech -> fires.
    assert arbitrate_endpoint(True, joint_suspect=False, turn_suspect=False,
                              more_speech_arriving=False) is True


def test_arbitrator_degraded_none_joint_path_unchanged() -> None:
    # The degraded / INDETERMINATE joint path is unchanged: a None joint suspect
    # adds no veto, so f136's pure-model behaviour is preserved.
    assert arbitrate_endpoint(True, joint_suspect=None) is True
    assert arbitrate_endpoint(True) is True


def test_arbitrator_degraded_none_with_arriving_speech_withholds() -> None:
    # f183: on the degraded (joint-None) path there is NO joint veto, so the ONLY thing
    # that can withhold a Smart Turn "complete" is the widened cleaned-signal Silero
    # hysteresis. A complete over a mid-word dip inside a still-continuing run of speech
    # (the echo-tail clip's frame-207 probe: joint None, prob 0.706, speech arriving) is
    # withheld to the next pause / hard cap, not fired.
    assert arbitrate_endpoint(True, joint_suspect=None, more_speech_arriving=True) is False


def test_arbitrator_degraded_none_quiet_speech_still_fires() -> None:
    # f136's pure behaviour preserved: a degraded joint-None verdict over a GENUINELY quiet
    # cleaned stream (hysteresis closed - the pause is the end of the turn, not a dip) still
    # fires complete. The widening withholds only while a run of speech is still open.
    assert arbitrate_endpoint(True, joint_suspect=None, more_speech_arriving=False) is True


def test_arbitrator_never_adds_a_cut() -> None:
    # The arbitrator only ever HOLDS a turn open; it never turns an incomplete /
    # no-verdict answer into a fire.
    assert arbitrate_endpoint(False, joint_suspect=False) is False
    assert arbitrate_endpoint(None, joint_suspect=True) is None
    assert arbitrate_endpoint(None, joint_suspect=False) is None
    assert arbitrate_endpoint(False, joint_suspect=True) is False


def test_arbitrator_hard_cap_respected_via_policy() -> None:
    # The wait a veto introduces is the incomplete floor, bounded by the hard cap
    # (3 s; 5 s dictation) - the same EndpointPolicy f136 shipped, not a new knob.
    policy = EndpointPolicy("default")
    assert policy.incomplete_floor_seconds() == 2.0
    assert policy.cap_seconds == 3.0
    assert min(policy.incomplete_wait_seconds, policy.cap_seconds) == 2.0
    assert EndpointPolicy("default", dictation=True).cap_seconds == 5.0
