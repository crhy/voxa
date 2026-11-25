"""Issue #72, task v11b: a live turn's reference must not read silent because the MAP ran off.

``turn_reference_window`` answers "what could the microphone still be carrying of her own voice
at this moment". The moment is the microphone's clock; the render positions come from the sink
clock stamped on every submitted chunk. On the ``whole-reply`` path Edge TTS is handed to
``appsrc`` faster than realtime, so the sink clock stamps the chunks' audible times in the
future - and the map places the microphone moment outside what the render ring still holds.

That is exactly the live failure the release check counted:
``reference active by definition while speaking: silent tap, abstain`` while
``render-tap attached=True samples=220416 peak=0.66 silent=False`` said the tap itself was
healthy. Attribution got silence for a reply that was audibly playing, so it could only abstain
and a real interruption could not be judged.

While the turn HAS render, silence is never the honest answer: the map, not the audio, is what
is off.
"""

import numpy as np

from voxa.timeline import RenderTimeline

RATE = 16000


def _loud(seconds: float, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _reply(timeline: RenderTimeline, clock: list, seconds: int = 10, lead: float = 3.0) -> None:
    """A whole-reply whose chunks arrive at the sink map ``lead`` seconds ahead of the clock.

    The real queue-ahead: ``GstSinkClock`` estimates when a chunk leaves the speakers, and a
    pipeline that is already full estimates later than the microphone's clock reports.
    """
    for _ in range(seconds):
        timeline.submit(_loud(1.0), generation="g1", path="whole-reply", now=clock[0] + lead)


def test_a_live_turn_never_reads_silent_while_the_map_runs_ahead():
    clock = [1000.0]
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: clock[0])
    timeline.open("g1", "whole-reply")
    timeline.open_turn()
    _reply(timeline, clock)

    for offset in (0.0, 1.0, 5.0):  # the microphone moments inside the reply
        window = timeline.turn_reference_window(moment=clock[0] + offset, length=1600)
        power = float(np.mean(window ** 2))
        assert timeline.turn_is_live(clock[0] + offset), "she is speaking at this moment"
        assert power > 1e-5, (
            f"the reference read silent mid-reply (offset {offset}s): the map ran off the "
            "render ring, which is the live 'silent tap, abstain' of issue #72 v11b")


def test_the_window_still_follows_the_map_when_the_map_is_valid():
    """The fix clamps a lost window; it does not freeze it: a later moment sees later audio."""
    clock = [1000.0]
    timeline = RenderTimeline(sample_rate=RATE, clock=lambda: clock[0])
    timeline.open("g1", "whole-reply")
    timeline.open_turn()
    for _ in range(3):
        timeline.submit(_loud(1.0), generation="g1", path="whole-reply", now=clock[0])

    early = timeline.turn_reference_window(moment=clock[0] + 1.0, length=1600)
    late = timeline.turn_reference_window(moment=clock[0] + 2.2, length=1600)
    assert not np.array_equal(early, late), "a valid map still moves the window through the reply"
