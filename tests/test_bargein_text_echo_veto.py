"""Issue #72, step 4: a leaked echo of her OWN words must never interrupt her (task A2 second guard).

In the v05 live run Voxa spoke "The answer is: 5 times 4 is 20." and the microphone came back
with "5 times more." - words her canceller let through - and she INTERRUPTED herself (79 times,
with echo cancelled OFF and the reference silent). DTLN with a live reference removes almost all
of her voice; the few words that leak through the mask still match what she is SAYING right now,
and the text veto below is the second guard behind it.

The veto must stay narrow: a person saying "stop" over her saying "stop" always interrupts
(task A2), and an interruption whose words merely mention the same topic does not match her
sentence letter for letter and still interrupts.
"""

from voxa.attribution import SourceEvidence
from voxa.bargectl import INTERRUPT, OWN_VOICE, BargeInController

DT_MS = 100.0
ARITHMETIC = "The answer is: 5 times 4 is 20."


def _drive(candidate, speaking=ARITHMETIC, evidence=None):
    """Settle the baseline on her voice, then drive a loud run of frames through observe."""
    control = BargeInController()
    control.reply_started(speaking)
    for _ in range(2 * int(400.0 / DT_MS)):  # onset window + settle: the baseline is trusted
        control.observe(0.05, speaking=True, muted=True, dt_ms=DT_MS)
    outcome = None
    for _ in range(60):  # something loud over her reply, past the listening window
        outcome = control.observe(0.9, speaking=True, muted=True, dt_ms=DT_MS,
                                  candidate=candidate, evidence=evidence or _silent_reference())
    return outcome


def _silent_reference() -> SourceEvidence:
    """The exact failing state from the v05 log: reference silent, microphone loud -> USER."""
    return SourceEvidence(reference_active=False, reference_power=0.0, mic_level=0.9, vad_prob=0.9)


def test_the_logged_pair_no_longer_interrupts() -> None:
    """She says "...5 times 4 is 20.", hears "5 times more.": her echo, never an interrupt."""
    outcome = _drive(lambda: ("5 times more.", "five times more"))
    assert outcome.action == OWN_VOICE, outcome


def test_the_echo_leak_never_reaches_interrupt_even_when_wrong() -> None:
    """A near-exact leak of the sentence tail ("4 is 20.") stays her own voice."""
    outcome = _drive(lambda: ("4 is 20.", "four is twenty"))
    assert outcome.action == OWN_VOICE, outcome


def test_a_person_over_her_stop_always_interrupts() -> None:
    """Task A2: a stop word beats the text veto even when she is saying "stop" herself."""
    outcome = _drive(lambda: ("stop", "stop"), speaking="Stop right there and wait.")
    assert outcome.action == INTERRUPT, outcome


def test_similar_but_different_words_still_interrupt() -> None:
    """"open the library" over her arithmetic reply shares no letters of her sentence: interrupt."""
    outcome = _drive(lambda: ("open the library", "open the library"))
    assert outcome.action == INTERRUPT, outcome


def test_the_1755_chessboard_pair_no_longer_interrupts() -> None:
    """The 18:35 audit's live INTERRUPT: she was SAYING "The first sentence is long enough that
    Voxa will still be speaking on the chessboard." and the echo came back as her own sentence
    mis-sung ("...in the long enough that butter will still be..."). The stop-word rule (task
    A2) made the veto step aside for "enough" - a word of HER OWN turn - and she cut herself
    off. A transcript that is her current text near-verbatim is her echo, stop word and all."""
    turn = ("The first sentence is long enough that Voxa will still be speaking on the chessboard.")
    heard = "The first sentence in the long enough that butter will still be speaking on the chessboard."
    outcome = _drive(lambda: (heard, heard), speaking=turn)
    assert outcome.action == OWN_VOICE, outcome


def test_a_short_stop_word_over_a_long_turn_is_still_an_interrupt() -> None:
    """The near-verbatim exception stays narrow: a real person's "enough" over her long chessboard
    turn is short and mostly foreign to it - it interrupts."""
    turn = ("The first sentence is long enough that Voxa will still be speaking on the chessboard.")
    outcome = _drive(lambda: ("enough", "enough"), speaking=turn)
    assert outcome.action == INTERRUPT, outcome
