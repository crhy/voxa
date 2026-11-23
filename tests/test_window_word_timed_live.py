"""f175: the word-timed own-speech EVIDENCE wired into the LIVE path (issue #72, roadmap task 7).

The bench drives the same feature; this mirrors what the live path must do and is what the gate
greps: ``voxa/window.py`` feeds the word-timed playback timeline from the speech
``on_words`` callback (anchor + edge-tts offsets), lets ``_hearing_evidence`` carry
``word_timed_overlap`` as ONE capped weight next to f173's joint verdicts, and resolves the
anaphoric "that one" with ``own_speech.item_being_read`` at the moment the floor is taken.

Like tests/test_window_joint_gate_live.py, the real-window assertions need a GTK display;
without one they skip - the logic they pin lives in voxa/ownspeech.py and is tested there.
"""

from __future__ import annotations

import time

import pytest

try:  # the real-window tests need GTK; the guards keep the module importable without it
    import gi  # noqa: F401
    HAVE_GI = True
except Exception:  # pragma: no cover
    HAVE_GI = False

from voxa.ownspeech import OwnSpeech  # noqa: E402


@pytest.fixture
def word_timed_window(tmp_path, monkeypatch):
    if not HAVE_GI:
        pytest.skip("no GTK display")
    from voxa.window import MainWindow

    win = MainWindow.__new__(MainWindow)
    win.own_speech = OwnSpeech()
    win._word_window_ms = 400.0
    win._reply_audible_anchor = None

    class _Shell:
        def __init__(self) -> None:
            self.word_timelines = []

        def set_word_timeline(self, words):
            self.word_timelines.append(list(words))

    win.shell = _Shell()
    return win


def test_on_reply_words_feeds_the_shell_AND_the_timeline(word_timed_window) -> None:
    win = word_timed_window
    anchor = time.monotonic()
    win._reply_audible_anchor = anchor
    MainWindow_words = [("Cancel", 1.0, 0.4), ("the", 1.4, 0.2)]
    win._on_reply_words(MainWindow_words, "reply-9")
    # the lip-sync shell keeps exactly what it was handed...
    assert win.shell.word_timelines == [MainWindow_words]
    # ...and the word-timed timeline holds the SAME words on the reply's audible clock.
    played = win.own_speech.played_words()
    assert [w.word for w in played] == ["cancel", "the"]
    assert played[0].start == pytest.approx(anchor + 1.0, abs=1e-6)
    assert played[0].generation == "reply-9"


def test_resolve_that_one_answers_with_the_word_on_her_tongue(word_timed_window) -> None:
    win = word_timed_window
    anchor = time.monotonic()
    win._reply_audible_anchor = anchor
    win._on_reply_words([("recipe", 2.4, 0.5), ("file", 2.9, 0.4)], "reply-1")

    class _Speech:
        def position(self) -> float:
            return 2.6  # mid-"recipe" when the user cut in

    win.speech = _Speech()
    item = win._resolve_that_one()
    assert item is not None and item.word == "recipe"
    # unanchored reply (no words played): the anaphor stays unresolved, never crashes.
    win._reply_audible_anchor = None
    assert win._resolve_that_one() is None


def test_hearing_evidence_carries_the_word_timed_share(word_timed_window) -> None:
    """The share lands on the SAME evidence the joint verdicts ride - capped weight, no decision."""
    win = word_timed_window
    from voxa.attribution import SourceEvidence

    # the window's own plumbing for _hearing_evidence (mirrors test_window_joint_gate_live)
    class _Audio:
        delay_samples = 2000

        def echo_measurements(self):
            return 1e-3, None, None

        def drain_joint_frames(self):
            return []

    class _Baseline:
        def note(self, _):
            return None

    class _Gate:
        baseline = 0.05

    class _Ctl:
        gate = _Gate()

    class _Conversation:
        _voice_detector = None

        def __init__(self) -> None:
            self.audio_evidence = None
            self.last_candidate_words = [{"word": "cancel", "onset_s": 1.05, "offset_s": 1.45}]
            self.last_candidate_span_start = 0.0

    win.audio = _Audio()
    win._echo_baseline = _Baseline()
    win._barge_ctl = _Ctl()
    win._latest_level = 1000.0
    win.conversation = _Conversation()
    win._joint_verdicts = lambda: (None, None, None)

    anchor = time.monotonic()
    win._reply_audible_anchor = anchor
    # Play "cancel" at the very moment the candidate's "cancel" lands (span starts at 0.0).
    win.own_speech.started("cancel the download")
    win.own_speech.note_played("cancel", 1.0, 1.4, generation="reply-1")

    ev = win._hearing_evidence()
    assert isinstance(ev, SourceEvidence)
    assert ev.word_timed_overlap == 1.0  # the one candidate word time-matched her "cancel"
    # And the evidence path never deleted anything: the utterance is still there to attribute.
    assert win.conversation.audio_evidence is not None  # attribute(evidence) ran normally
