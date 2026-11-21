"""Task A3's gate: the barge-in decision, its evidence and its transcript OFF the frame clock.

What this file insists on, and nothing else:

* the candidate transcript - the only part of the decision that runs a model - is produced on
  the hearing thread, never on the thread that owns the frame clock;
* the frame clock never waits for a decision, and a worker that falls behind drops the OLDEST
  evidence rather than queueing a backlog that would decide something late;
* every window is counted in milliseconds of AUDIO, not in callbacks and not in wall-clock
  grace: an interruption that starts together with Voxa's own speech is detected inside the
  first 400 ms of the reply, where the old ``BARGE_IN_GRACE_SECONDS`` folded the microphone;
* yielding once per reply, and invalidating the generation, makes the queued evidence, the
  queued decisions and the callbacks of the interrupted reply stale together.

No GTK, no audio device, no model: the transcript is a callable the test supplies.
"""

import threading
import time
from dataclasses import replace

from voxa.bargectl import (HOLD, IGNORED, INTERRUPT, OFFLINE, OWN_VOICE, YIELD,
                          BargeInController, BargeOutcome)
from voxa.hearing import HearingDirector, HearingSample
from voxa.onset import ONSET_EVIDENCE_MS


def sample(*, level=0.05, mic_db=-40.0, reference_db=-30.0, delay_ms=200.0, dt_ms=20.0,
           speaking=True, muted=True, paused=False, candidate=None,
           wake_word="voxa", media_just_started=False) -> HearingSample:
    """One frame, as the frame clock would hand it over."""
    return HearingSample(level01=level, speaking=speaking, paused=paused, muted=muted,
                         wake_word=wake_word, media_just_started=media_just_started,
                         mic_db=mic_db, reference_db=reference_db, delay_ms=delay_ms,
                         dt_ms=dt_ms, candidate=candidate, evidence=None)


def collect(director, timeout: float = 3.0) -> list:
    """Wait for the worker to have judged everything queued, then take the decisions."""
    deadline = time.monotonic() + timeout
    while director.pending and time.monotonic() < deadline:
        time.sleep(0.005)
    outcomes = []
    while True:
        outcome = director.poll()
        if outcome is None:
            return outcomes
        outcomes.append(outcome)


def control(**kwargs) -> BargeInController:
    """A controller whose windows are driven by the audio the test feeds it."""
    return BargeInController(**kwargs)


class BlockingController:
    """A controller that never returns, to make the worker fall behind on purpose."""

    def __init__(self) -> None:
        self.block = threading.Event()
        self.judged = 0

    def observe(self, level01: float, **_kwargs) -> BargeOutcome:
        self.judged += 1
        self.block.wait(5.0)
        return BargeOutcome(action=IGNORED)

    def reply_started(self, text: str = "") -> None:
        pass

    def reply_ended(self) -> None:
        pass

    def note_speaking(self) -> None:
        pass

    def invalidate(self) -> None:
        pass


# --- the transcript is not the frame clock's job -------------------------------------

def test_the_candidate_transcript_runs_on_the_hearing_thread() -> None:
    """The one part of the decision that runs a model never runs on the frame clock."""
    threads: list[threading.Thread] = []

    def candidate() -> tuple[str, str]:
        threads.append(threading.current_thread())
        return "open the library", "open the library"

    director = HearingDirector(sample_queue=96, decision_queue=96)
    director.start()
    director.reply_started("The weather is clear today.")
    for _ in range(25):  # her own level: the speaker's onset window, then the baseline
        director.submit(sample(level=0.05))
    for _ in range(90):  # then someone else, loud, for longer than the listening window
        director.submit(sample(level=0.9, mic_db=-20.0, candidate=candidate))
    outcomes = collect(director)
    director.stop()

    assert threads, "the words stage never ran: the candidate was never asked"
    assert all(thread is not threading.main_thread() for thread in threads), \
        "the candidate transcript ran on the frame clock"
    assert {thread.name for thread in threads} == {"voxa-hearing"}
    assert any(outcome.action in (INTERRUPT, OWN_VOICE, HOLD, OFFLINE) for outcome in outcomes), \
        "no decision was taken from the words: " + repr([outcome.action for outcome in outcomes])


def test_the_decision_is_handed_over_as_a_decision_only() -> None:
    """The hearing thread's callback receives an outcome and nothing else.

    ``voxa/window.py`` answers it with ``idle(...)``, which is what puts the effects - stopping
    playback, unmuting, arming the prompt - back on the frame clock.
    """
    where: list[str] = []
    director = HearingDirector(sample_queue=96, decision_queue=96,
                               on_decision=lambda outcome: where.append(threading.current_thread().name))
    director.start()
    director.reply_started("The weather is clear today.")
    for _ in range(25):
        director.submit(sample(level=0.05))
    for _ in range(60):
        director.submit(sample(level=0.9, mic_db=-20.0))
    outcomes = collect(director)
    director.stop()

    assert outcomes, "no decision was handed to the frame clock"
    assert where, "the callback never ran"
    assert set(where) == {"voxa-hearing"}, where
    assert all(hasattr(outcome, "action") and hasattr(outcome, "generation") for outcome in outcomes)


def test_the_frame_clock_never_waits_for_a_decision() -> None:
    """A submit returns immediately whatever the worker is busy doing."""
    inside = threading.Event()
    release = threading.Event()

    def candidate() -> tuple[str, str]:
        inside.set()
        release.wait(5.0)
        return "open the library", "open the library"

    director = HearingDirector(sample_queue=96, decision_queue=96)
    director.start()
    director.reply_started("The weather is clear today.")
    for _ in range(25):
        director.submit(sample(level=0.05))
    for _ in range(90):
        director.submit(sample(level=0.9, mic_db=-20.0, candidate=candidate))
    assert inside.wait(2.0), "the worker never reached the words stage"

    started = time.perf_counter()
    director.submit(sample(level=0.9, mic_db=-20.0, candidate=candidate))
    waited = time.perf_counter() - started
    release.set()
    director.stop()

    assert waited < 0.05, f"submitting a frame waited {waited:.3f} s for the worker"


def test_a_worker_that_falls_behind_drops_the_oldest_evidence() -> None:
    """The queue is bounded, and what it drops is the past, not the present."""
    stuck = BlockingController()
    director = HearingDirector(controller=stuck, sample_queue=4)
    director.start()
    director.reply_started("The weather is clear today.")
    for _ in range(40):
        director.submit(sample(level=0.9))
    director.stop(join=False)
    stuck.block.set()

    assert director.dropped > 0, "the queue grew instead of dropping the oldest sample"
    assert director.pending <= 4, f"{director.pending} events queued in a queue of 4"


# --- the windows are milliseconds of audio, not callbacks and not grace -------------

def test_an_interruption_at_the_start_of_the_reply_is_detected() -> None:
    """No grace period: the onset fires inside the first 400 ms of the reply.

    ``BARGE_IN_GRACE_SECONDS`` folded the microphone for the first 400 ms of every reply, which
    is exactly where an interruption that starts together with Voxa's speech lives. This is the
    case the old rule could not see at all.
    """
    control_ = control()
    control_.reply_started("The weather is clear today.")
    actions = []
    outcome = None
    for index in range(24):
        # Two frames that teach the coupling what the speaker path sounds like in this room,
        # then the microphone holding sound above the ceiling the echo could have produced.
        mic_db = -40.0 if index < 2 else -20.0
        outcome = control_.observe(0.5, speaking=True, muted=True, mic_db=mic_db,
                                   reference_db=-30.0, delay_ms=200.0, dt_ms=20.0)
        actions.append(outcome.action)
        if outcome.action == YIELD:
            break

    assert outcome is not None and outcome.action == YIELD, actions
    assert outcome.onset, actions
    assert outcome.evidence_ms >= ONSET_EVIDENCE_MS, actions
    assert control_.onset.audio_ms <= 400.0, \
        f"the onset needed {control_.onset.audio_ms:.0f} ms of audio, inside the old grace period"
    assert outcome.onset_audio_ms >= ONSET_EVIDENCE_MS


def test_the_listening_window_counts_audio_not_callbacks() -> None:
    """The same number of frames decides differently when they stand for different audio.

    Fourteen frames of 100 ms pass the listening window; fourteen frames of 20 ms, four times
    less sound, do not. That is what makes the rule portable between the live app (a 50 ms
    level timer) and the bench (20 ms frames), and it is why a sample carries ``dt_ms``.
    """
    def run(dt_ms: float) -> int:
        asked = []

        def candidate() -> tuple[str, str]:
            asked.append(1)
            return "open the library", "open the library"

        control_ = control()
        control_.reply_started("The weather is clear today.")
        for _ in range(int(400.0 / dt_ms)):  # her own voice, long enough to settle the baseline
            control_.observe(0.05, speaking=True, muted=True, dt_ms=dt_ms)
        for _ in range(14):  # then someone else, loud
            control_.observe(0.9, speaking=True, muted=True, dt_ms=dt_ms, candidate=candidate)
        return len(asked)

    assert run(100.0) >= 1, "the words were never judged although 1.4 s of audio passed the window"
    assert run(20.0) == 0, "the window was counted in callbacks, not in audio"


def test_the_gate_settles_in_audio_not_in_ticks() -> None:
    """The baseline is trusted after 400 ms of her own voice, whatever the frame rate is.

    The first 400 ms of the reply is her speaker's onset and counts for nothing; the 400 ms
    after it both teaches the baseline and settles the gate. Two frame rates, same audio, same
    answer - which is what ``dt_ms`` on the sample is for.
    """
    for dt_ms in (20.0, 100.0):
        control_ = control()
        control_.reply_started("Speaking.")
        for _ in range(int(400.0 / dt_ms)):  # the onset window: teaches nothing
            control_.observe(0.05, speaking=True, muted=True, dt_ms=dt_ms)
        assert not control_.gate.settled, f"{dt_ms} ms frames: settled on the speaker's onset"
        for _ in range(int(400.0 / dt_ms) - 1):
            control_.observe(0.05, speaking=True, muted=True, dt_ms=dt_ms)
        assert not control_.gate.settled, f"{dt_ms} ms frames: settled before 400 ms of her voice"
        control_.observe(0.05, speaking=True, muted=True, dt_ms=dt_ms)
        assert control_.gate.settled, f"{dt_ms} ms frames: not settled after 400 ms of audio"


def test_the_onset_rule_says_nothing_without_a_measurable_echo() -> None:
    """A reference that cannot be measured, or an echo whose travel time was never estimated,
    is not "the echo explains nothing": it is no evidence, in either direction."""
    for reference_db, delay_ms in ((None, 200.0), (-30.0, None)):
        control_ = control()
        control_.reply_started("The weather is clear today.")
        for _ in range(60):
            outcome = control_.observe(0.9, speaking=True, muted=True, mic_db=-10.0,
                                       reference_db=reference_db, delay_ms=delay_ms, dt_ms=20.0)
            assert outcome.action != YIELD, (reference_db, delay_ms, outcome.action)
        assert control_.onset.evidence_ms == 0.0, (reference_db, delay_ms)


# --- yielding, and what cancelling a generation means --------------------------------

def test_the_floor_is_yielded_once_per_reply() -> None:
    """A second run of unexplained sound inside the same reply is the same interruption."""
    control_ = control()
    control_.reply_started("The weather is clear today.")
    yields = 0
    for index in range(120):
        mic_db = -40.0 if index < 2 else (-20.0 if index % 30 < 24 else -40.0)
        outcome = control_.observe(0.5, speaking=True, muted=True, mic_db=mic_db,
                                   reference_db=-30.0, delay_ms=200.0, dt_ms=20.0)
        yields += outcome.action == YIELD
    assert yields == 1, f"the same reply yielded the floor {yields} times"


def test_a_stale_generation_is_never_applied() -> None:
    """Yielding invalidates the generation: queued evidence, queued decisions and the callbacks
    of the reply that was interrupted all go stale together."""
    director = HearingDirector()
    director.start()
    director.reply_started("The weather is clear today.")
    for _ in range(60):
        director.submit(sample(level=0.9, mic_db=-20.0))
    generation = director.generation()
    director.invalidate()
    outcomes = collect(director)
    director.stop()

    assert director.generation() == generation + 1
    assert director.stale > 0, "the evidence queued for the yielded reply was judged anyway"
    assert all(outcome.generation != generation for outcome in outcomes), \
        "a decision from the invalidated generation reached the frame clock"
    assert not director.accept(replace(BargeOutcome(action=INTERRUPT), generation=generation)), \
        "a decision from the yielded reply was accepted"
    assert director.accept(replace(BargeOutcome(action=INTERRUPT), generation=generation + 1))


def test_invalidation_clears_the_evidence_but_not_the_words() -> None:
    """The next reply starts its evidence over; the own-voice check still needs what was said."""
    control_ = control()
    control_.reply_started("The weather is clear today.")
    for _ in range(2):  # the coupling: what the speaker path sounds like in this room
        control_.observe(0.9, speaking=True, muted=True, mic_db=-40.0, reference_db=-30.0,
                         delay_ms=200.0, dt_ms=20.0)
    for _ in range(30):  # then the microphone holding sound above that ceiling
        control_.observe(0.9, speaking=True, muted=True, mic_db=-20.0, reference_db=-30.0,
                         delay_ms=200.0, dt_ms=20.0)
    assert control_.onset.evidence_ms > 0.0, "no evidence was accumulating to begin with"
    control_.invalidate()
    assert control_.onset.evidence_ms == 0.0
    assert control_.speaking_text == "The weather is clear today.", \
        "a rejected interruption lost the words it is judged against"


def test_the_yield_cancels_the_evidence_of_a_reply_that_has_already_stopped() -> None:
    """After the floor is yielded, the microphone opens and the reply is over: her own voice,
    the echo tail of the reply that was stopped, and the evidence still queued for it cannot
    yield the floor a second time.
    """
    director = HearingDirector(sample_queue=64, decision_queue=64)
    director.start()
    director.reply_started("The weather is clear today.")
    for _ in range(2):  # the coupling first, then the sound the speaker path cannot explain
        director.submit(sample(level=0.05, mic_db=-40.0))
    for _ in range(30):
        director.submit(sample(level=0.9, mic_db=-20.0))
    outcomes = collect(director)
    director.stop()

    yields = [outcome for outcome in outcomes if outcome.action == YIELD]
    assert yields, "no yield: " + repr([outcome.action for outcome in outcomes])
    assert len(yields) == 1, f"the floor was yielded {len(yields)} times in one reply"
    assert yields[0].onset and yields[0].onset_audio_ms >= ONSET_EVIDENCE_MS
