"""An idle Voxa must not flash: face state changes are debounced to >= 300 ms (issue #72, v11d).

The idle face follows every hearing decision (gate-open / hold / echo / user). On room noise those
decisions flip READY <-> LISTENING for a frame or two, and the face jumped ("flashes repeatedly in
the top-right while idle"). ``commit_face_state`` only shows a state once it has held for 300 ms,
so a room-noise hold that never really began a prompt never reaches the face.
"""

from voxa.ui.state import (AssistantState as S, FACE_STATE_DEBOUNCE_S, commit_face_state)


def test_a_short_blip_never_shows() -> None:
    """A READY -> LISTENING -> READY flip shorter than 300 ms never moves the committed face."""
    committed, tracked, at = S.READY, None, None
    # t=0: a room-noise candidate opens LISTENING.
    committed, tracked, at = commit_face_state(S.LISTENING, committed, 0.0, tracked, at)
    assert committed is S.READY  # not shown yet
    # t=0.12 (< 300 ms): the room noise is a HOLD, the prompt closes again.
    committed, tracked, at = commit_face_state(S.READY, committed, 0.12, tracked, at)
    assert committed is S.READY
    # No sub-300 ms blip ever reached the face.
    assert committed is S.READY


def test_a_held_change_shows_after_the_hold() -> None:
    committed, tracked, at = S.READY, None, None
    committed, tracked, at = commit_face_state(S.LISTENING, committed, 0.0, tracked, at)
    assert committed is S.READY
    # 0.299 s: still under the hold.
    committed, tracked, at = commit_face_state(S.LISTENING, committed, 0.299, tracked, at)
    assert committed is S.READY
    # 0.300 s: it has held long enough, the face shows it.
    committed, tracked, at = commit_face_state(S.LISTENING, committed, 0.300, tracked, at)
    assert committed is S.LISTENING


def test_idle_ready_is_never_churned() -> None:
    """Staying idle in READY for many frames never touches the committed face."""
    committed, tracked, at = S.READY, None, None
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        committed, tracked, at = commit_face_state(S.READY, committed, t, tracked, at)
        assert committed is S.READY


def test_hold_ms_is_three_hundred_milliseconds() -> None:
    assert FACE_STATE_DEBOUNCE_S == 0.300
