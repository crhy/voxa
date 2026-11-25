"""Issue #72, task v11 step 4: the held capture span must not grow across turns.

The rc5 live run held ONE span that grew past 60 s (three VoxaTest prompts plus her replies in
one string, re-transcribed every second by Whisper base on the CPU), producing garbled requests
and slow turns. The span is now bounded at 15 s AND closed at every turn boundary (her reply end,
a promoted request, an accepted barge-in), and a closed span's transcript cache is cleared so a
previous request is never re-transcribed.
"""

from voxa.capture import MAX_PINNED_SECONDS, CaptureRing
from voxa.conversation import ConversationController


def _pcm(seconds: float) -> bytes:
    return b"\x00\x00" * int(16000 * seconds)


def _controller() -> ConversationController:
    return ConversationController(
        wake_whisper=object(),
        prompt_whisper=object(),
        language="en",
        wake_word="voxa",
        threshold=500,
        silence_ms=100,
        voice_detector=False,
        max_segment_seconds=5.0,
        on_woken=lambda: None,
        on_prompt=lambda _t: None,
        on_status=lambda _s: None,
        on_error=lambda _e: None,
    )


def test_pinned_span_is_capped_at_fifteen_seconds():
    assert MAX_PINNED_SECONDS == 15.0
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0], window_seconds=5.0, pre_roll_seconds=0.0)
    ring.pin(seq=None)
    # Keep feeding a long "request": the span may never exceed the cap.
    for i in range(200):
        clock[0] += 0.1
        ring.add(_pcm(0.1), 0.5, seq=i)
    assert ring.span_seconds <= MAX_PINNED_SECONDS + 0.11, ring.span_seconds


def test_close_span_releases_pin_and_clears_cache():
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0], window_seconds=5.0, pre_roll_seconds=0.0)
    controller = _controller()
    controller.capture = ring
    controller._candidate_cache = (0, 9, "old request")
    controller._candidate_text = "old request"
    clock[0] += 0.1
    ring.add(_pcm(0.1), 0.6, seq=0)  # an onset frame exists to pin
    ring.pin(seq=0)
    assert ring.pin_from is not None
    controller.close_span()
    assert ring.pin_from is None, "the held span is dropped at the turn boundary"
    assert controller._candidate_cache is None, "the previous request's transcript is not reused"
    assert controller._candidate_text == ""


def test_two_turns_do_not_share_one_span():
    # The exact live failure shape: prompt, reply, prompt, reply in ONE growing string.
    clock = [0.0]
    ring = CaptureRing(clock=lambda: clock[0], window_seconds=30.0, pre_roll_seconds=0.0)
    controller = _controller()
    controller.capture = ring
    seq = [0]

    def request() -> list[int]:
        # barge-in candidate: an onset frame is pinned, then the audio arrives.
        clock[0] += 0.1
        ring.add(_pcm(0.1), 0.6, seq=seq[0])
        ring.pin(seq=seq[0])
        seq[0] += 1
        for _ in range(20):
            clock[0] += 0.1
            ring.add(_pcm(0.1), 0.6, seq=seq[0])
            seq[0] += 1
        ids = [frame.seq for frame in ring.pinned()]
        # turn boundary: the request is handed on and the span closes.
        controller.close_span()
        return ids

    first = request()
    second = request()
    # The second turn's span starts where the first ended: no frame is shared.
    assert max(first) < min(second), (first[-1], second[0])

