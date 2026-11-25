"""Issue #72, task v08 step 3: the text veto reads her WHOLE TURN, not just the current piece.

The rc4 live log (barge.log, 18:25-18:30) repeats one failure three ways: Voxa's reply left the
speakers in pieces, her voice came back ~1 s late as a Whisper mishearing of the PREVIOUS piece,
the microphone heard it while the NEXT piece played, and the per-piece veto compared it to the
wrong words - every mishearing was read as the user ("reference silent: no echo possible") and
she interrupted herself.

The three logged pairs pinned here:

- "Here are two ways to make lemonade at home:" ... heard "We are two ways to make leveling up
  on one." while piece 2 ("Using a jar...") played: pure echo of piece 1 -> vetoed.
- "Honey is actually one of the best natural sources of protein" ... heard "How he is actually
  one of the box natural sources acquired in the online.": pure echo -> vetoed.
- "You're right, he's definitely one of the box natural sources..." ... heard "What is the
  capital of France? Your life is definitely one of the box natural sources of Thank you.":
  a REAL user question whose words happen to share her content words - the echo words are
  stripped, and what remains still counts as a real user turn (INTERRUPT).
"""

from voxa.attribution import SourceEvidence
from voxa.bargectl import INTERRUPT, OWN_VOICE, BargeInController

DT_MS = 100.0


def _silent_reference() -> SourceEvidence:
    """The exact failing state from the rc4 log: reference silent, microphone loud -> user."""
    return SourceEvidence(reference_active=False, reference_power=0.0, mic_level=0.9, vad_prob=0.9)


def _drive_turn(pieces: list[str], candidate):
    """Open one TURN (first speak-start), read out its pieces, then let the microphone hear a
    loud run of frames while the LAST piece is still playing. Returns (controller, outcome)."""
    control = BargeInController()
    control.note_turn(pieces[0])
    control.reply_started(pieces[0])
    for piece in pieces[1:]:
        control.note_piece(piece)
        control.reply_started(piece)
    for _ in range(2 * int(400.0 / DT_MS)):  # settle the baseline on her quiet reply
        control.observe(0.05, speaking=True, muted=True, dt_ms=DT_MS)
    outcome = None
    for _ in range(60):
        outcome = control.observe(0.9, speaking=True, muted=True, dt_ms=DT_MS,
                                  candidate=candidate, evidence=_silent_reference())
    return control, outcome


LEMONADE = [
    "Here are two ways to make lemonade at home:\n\n1.",
    "**Using a jar:** Place the jar on a pot of boiling water, add the lemon juice and sugar.",
]
HONEY = [
    "Honey is actually one of the best natural sources of protein,",
    "the owner, sparkles.",
]
FRANCE_TURN = [
    "You're right, he's definitely one of the box natural sources of protein, the owner, sparkles.",
    "Sparkles offers this as a direct meal replacement.",
]


def test_the_veto_now_sees_the_whole_turn() -> None:
    """The turn's words are ONE evidence string, current piece or not."""
    control, _ = _drive_turn(LEMONADE, lambda: ("", ""))
    veto_text = control.turn_veto_text()
    assert "lemonade" in veto_text and "jar" in veto_text, veto_text


def test_pure_echo_of_piece_one_while_piece_two_plays_is_vetoed() -> None:
    """Log 18:25:08 - she hears piece 1 ("...two ways to make leveling up on one") while
    piece 2 plays: the per-piece veto never saw those words, the turn-wide one does."""
    _, outcome = _drive_turn(
        LEMONADE,
        lambda: ("We are two ways to make leveling up on one.",
                 "we are two ways to make leveling up on one"))
    assert outcome.action == OWN_VOICE, outcome


def test_echo_of_the_honey_piece_is_vetoed() -> None:
    """Log 18:29:52 - "Honey is actually one of the best natural sources" comes back as "How he
    is actually one of the box natural sources acquired in the online.": pure echo -> vetoed."""
    _, outcome = _drive_turn(
        HONEY, lambda: ("How he is actually one of the box natural sources acquired in the online.",
                        "how he is actually one of the box natural sources acquired in the online"))
    assert outcome.action == OWN_VOICE, outcome


def test_the_logged_france_question_still_counts_as_a_user_turn() -> None:
    """Log 18:30:03 - the heard line LEAKS her words ("...one of the box natural sources of"),
    but mostly-foreign words ("capital", "france", "life") make it a real question: the echo
    words are stripped and what remains interrupts."""
    _, outcome = _drive_turn(
        FRANCE_TURN,
        lambda: ("What is the capital of France? Your life is definitely one of the box "
                 "natural sources of Thank you.",
                 "what is the capital of france your life is definitely one of the box natural "
                 "sources of thank you"))
    assert outcome.action == INTERRUPT, outcome


def test_the_turn_signals_speaking_across_gaps_and_the_echo_tail() -> None:
    """i02's ``speaking`` mark is turn-wide: the gaps between pieces are hers, and her echo is
    still in the microphone for ECHO_TAIL_SECONDS after the last piece."""
    time = {"now": 100.0}
    control = BargeInController(clock=lambda: time["now"])
    assert not control.speaking_now()
    control.note_turn("Here are two ways to make lemonade at home:")
    control.reply_started("Here are two ways to make lemonade at home:")
    assert control.speaking_now()
    time["now"] += 30.0  # a silent gap BETWEEN pieces is still her reply
    assert control.speaking_now()
    control.note_piece("**Using a jar:** Place the jar on a pot of boiling water.")
    control.reply_ended()  # the last piece left the speakers
    time["now"] += 1.4
    assert control.speaking_now(), "the echo tail is hers too"
    time["now"] += 0.2
    assert not control.speaking_now()


def test_the_previous_turns_words_stay_evidence_for_ten_seconds() -> None:
    """The echo of a reply can arrive after the NEXT reply began speaking."""
    time = {"now": 100.0}
    control = BargeInController(clock=lambda: time["now"])
    control.note_turn("Honey is actually one of the best natural sources of protein,")
    time["now"] += 5.0
    control.note_turn("You're right, he's definitely one of the box natural sources")
    assert "honey" in control.turn_veto_text().lower()
    time["now"] += 6.0
    assert "honey" not in control.turn_veto_text().lower()
