"""Issue #72, task v11 step 2: the render tap can attach yet read SILENT.

The v09 live run had the reference reading silent on the real flatpak sink even while Voxa spoke
(``reference active by definition while speaking: silent tap, abstain``), so no words over her
could ever become ``user`` and real barge-ins were dead. v08's test passed only because it ran
with ``GST_AUDIOSINK=fakesink``; the installed bundle uses a real PulseAudio sink and the tap
there did not feed the reference.

This is the pure-Python half of that cause: the render timeline records what the tap handed it,
and ``turn_stats`` reports the tap's own loudness. A reply with samples but a near-zero peak is
exactly the silent tap the live check must catch - and it must not be mistaken for a real, loud
reference.
"""

import time

import numpy as np

from voxa.timeline import RenderTimeline


def _tone(seconds: float, level: float, rate: int = 16000) -> np.ndarray:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    return (level * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silent(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


def test_silent_reply_is_reported_as_a_silent_tap():
    clock = [0.0]
    timeline = RenderTimeline(sample_rate=16000, clock=lambda: clock[0])
    timeline.open_turn()
    for _ in range(5):
        clock[0] += 0.3
        timeline.submit(_silent(0.3), generation="g", path="tts-file")
    stats = timeline.turn_stats()
    assert stats is not None
    assert stats["samples"] > 0, "the tap attached and recorded audio"
    assert stats["peak"] < 1e-3, "the recorded audio is silent"


def test_loud_reply_is_not_reported_silent():
    clock = [0.0]
    timeline = RenderTimeline(sample_rate=16000, clock=lambda: clock[0])
    timeline.open_turn()
    for _ in range(5):
        clock[0] += 0.3
        timeline.submit(_tone(0.3, 0.2), generation="g", path="tts-file")
    stats = timeline.turn_stats()
    assert stats is not None
    assert stats["peak"] > 1e-3, "a real reference is loud, so she is not answered by herself"


def test_one_silent_piece_among_loud_ones_is_still_loud():
    clock = [0.0]
    timeline = RenderTimeline(sample_rate=16000, clock=lambda: clock[0])
    timeline.open_turn()
    clock[0] += 0.3
    timeline.submit(_tone(0.3, 0.3), generation="g", path="tts-file")
    clock[0] += 0.3
    timeline.submit(_silent(0.3), generation="g", path="tts-file")
    clock[0] += 0.3
    timeline.submit(_tone(0.3, 0.3), generation="g", path="tts-file")
    stats = timeline.turn_stats()
    assert stats["peak"] > 1e-3, "a gap between pieces must not make the whole reply read silent"
