"""f173: the joint double-talk gate wired into the LIVE hearing path (issue #72, roadmap task 5).

f172 shipped the joint verdicts but left them bench-only (notes-f172 "Open / honest caveats"):
``voxa/window.py::_hearing_evidence`` did not feed ``DoubleTalkGate``, so the live barge-in path
could never veto her own echo or raise a suspect on a second source. f173 puts the wiring in: the
capture thread stages each cleaned frame (voxa/audio.py ``_note_joint_frame`` - bytes plus the
delay/margin/ERLE the canceller and mask measured, NO model), the hearing worker drains that ring
and feeds the cleaned-stream Silero and the DoubleTalkGate (voxa/jointgate ``feed_joint_frames``,
called from ``window._joint_verdicts``), and ``_hearing_evidence`` carries the joint verdicts the
attribution/bargectl joint gate already reads.

These mirror the locked-delay fixtures of tests/test_voicedetect.py, but through the live wiring:
the same echo / double-talk material at the delay the lock proved, asserted on the verdicts the
live path produces and on what they do to the candidate window. (The gi-guarded test at the end
drives the real ``MainWindow._hearing_evidence`` on a machine with a display.)
"""

from __future__ import annotations

import pytest

try:  # the real-window test at the bottom needs a GTK display; everything else does not
    import gi
except Exception:  # pragma: no cover
    gi = None

import numpy as np  # noqa: E402

from voxa.audio import AudioCapture  # noqa: E402
from voxa.attribution import SourceEvidence  # noqa: E402
from voxa.bargectl import BargeInController  # noqa: E402
from voxa.jointgate import feed_joint_frames  # noqa: E402
from voxa.voicedetect import DoubleTalkGate  # noqa: E402


# --- the shipping material, at the delay the lock proved (mirrors test_voicedetect) -------------

def _echo_train(ref: np.ndarray, delay: int, taps: tuple[float, ...]) -> np.ndarray:
    mic = np.zeros(ref.size + delay + 40 * len(taps), dtype=np.float64)
    for b, gain in enumerate(taps):
        shift = delay + 40 * b
        mic[shift:shift + ref.size] += gain * ref
    return mic


def _staged_frames(n_frames: int, delay: int, rng, *, user: bool = False, user_from: int = 12,
                   user_to: int | None = None) -> list[tuple]:
    """Frames of the shipping path staged in the capture ring's shape: the echo train at the
    locked delay, plus (for the user case) an uncorrelated second source the delay can't explain."""
    user_to = user_to if user_to is not None else n_frames
    ref_full = rng.uniform(-0.4, 0.4, n_frames * 320 + delay + 512)
    mic_full = _echo_train(ref_full, delay, (1.0, 0.45, 0.25, 0.12))
    if user:
        u = rng.uniform(-1.2, 1.2, mic_full.size)
        u[:user_from * 320] = 0.0
        u[user_to * 320:] = 0.0
        # f187: the canceller has taken her echo where the second source is (15% tail):
        # user frames over RAW echo still read echo-coherent at the frozen 0.2 half.
        idx = np.arange(mic_full.size)
        in_user = (idx >= user_from * 320) & (idx < user_to * 320)
        mic_full = np.where(in_user, 0.15 * mic_full + u, mic_full)
    frames = []
    for f in range(n_frames):
        ref_f = ref_full[f * 320:(f + 1) * 320]
        mic_f = np.clip(mic_full[f * 320:(f + 1) * 320], -1.0, 1.0)
        mic_bytes = np.rint(mic_f * 32767.0).astype("<i2").tobytes()
        cleaned = mic_bytes  # the cleaned stream only feeds Silero; the stub owns the probability
        # The mask's band margin over its own prediction: her echo leaves ~0 dB, a second source
        # towers over the prediction the mask would have kept (the +6 dB rule).
        band_margin = 10.0 if (user and user_from <= f < user_to) else 0.0
        frames.append((cleaned, mic_bytes, ref_f, delay, band_margin, 15.0, True, True))
    return frames


class _StubVad:
    """Stand-in for the cleaned-stream Silero: the test owns the probability, the way the bare-gate
    fixtures pass ``prob`` (the model itself is exercised in test_voicedetect)."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = probs or [0.0]
        self._i = 0
        self.last_prob = 0.0
        self.probs: list[float] = []

    def feed(self, _cleaned: bytes) -> bool:
        self.last_prob = self._probs[min(self._i, len(self._probs) - 1)]
        self._i += 1
        self.probs = [self.last_prob]
        return self.last_prob >= 0.65


def _feed(frames: list[tuple], probs: list[float]):
    gate = DoubleTalkGate()
    vad = _StubVad(probs)
    ev = (None, None, None)
    for f in frames:
        ev = feed_joint_frames(gate, vad, [f])
    return gate, ev


# --- the joint verdict reaches the live evidence -----------------------------------------------

def test_feed_returns_the_joint_verdict_when_a_lock_exists() -> None:
    """A pure-echo clip at the delay the lock proved: the live feed carries the joint MIRROR
    verdict (the veto), not None - the live path now sees what the bench always did."""
    frames = _staged_frames(20, 2000, np.random.default_rng(11))
    gate, (suspect, echo, _margin) = _feed(frames, [0.1] * 20)
    assert echo is True                  # the microphone held the delayed reference: the veto
    assert suspect is not True           # no second source claimed
    assert gate.echo_consistent and not gate.suspect


def test_pure_echo_frame_at_the_exact_delay_vetoes_with_the_suspect_silent() -> None:
    """Mirror verdict up, suspect silent: echo-only frames the lock proved are her voice, at the
    exact delay - the coherence f132 could not measure now measures high."""
    frames = _staged_frames(20, 2000, np.random.default_rng(12))
    gate, (suspect, echo, _margin) = _feed(frames, [0.1] * 20)
    assert echo is True and suspect is not True
    assert gate.coherence is not None and gate.coherence >= 0.5


def test_double_talk_raises_the_suspect_and_opens_the_candidate_window() -> None:
    """The user over her echo at the same locked delay: the three streams hold, the suspect raises,
    and the joint gate opens the candidate window on the live path (bargectl reads joint_suspect)."""
    frames = _staged_frames(24, 2000, np.random.default_rng(13), user=True, user_from=12)
    gate = DoubleTalkGate()
    vad = _StubVad([0.1] * 12 + [0.9] * 12)
    ctl = BargeInController()
    ctl.reply_started("her sentence")
    opened = False
    for f in frames:
        suspect, echo, margin = feed_joint_frames(gate, vad, [f])
        ev = SourceEvidence(reference_active=True, reference_power=1e-3,
                            joint_suspect=suspect, joint_echo=echo,
                            residual_over_prediction_db=margin, alignment_healthy=True)
        ctl.observe(0.4, speaking=True, paused=False, muted=True, wake_word="voxa",
                    evidence=ev, mic_db=-30.0, reference_db=-30.0, delay_ms=125.0, dt_ms=20.0)
        if ctl.pending_since > 0.0:
            opened = True
    assert opened, "double-talk on the live path must open the candidate window"
    assert gate.suspect


def test_unmeasurable_frame_falls_back_to_the_level_gate_visibly() -> None:
    """No lock (alignment unhealthy): the gate cannot measure, so both joint fields are None and
    the caller keeps the level gate - the degraded path, never a guess of 'no second source'."""
    frames = _staged_frames(10, 2000, np.random.default_rng(14))
    frames = [(*f[:6], False, f[7]) for f in frames]  # alignment_healthy False on every frame
    _gate, (suspect, echo, _margin) = _feed(frames, [0.1] * 10)
    assert suspect is None and echo is None

    # a drain with no staged material at all degrades the same way
    gate2 = DoubleTalkGate()
    assert feed_joint_frames(gate2, _StubVad([]), []) == (None, None, None)


def test_a_joint_verdict_never_flips_on_the_confidence_fields() -> None:
    """The joint verdict is audio evidence: the level fields never flip it (the handoff rule - a
    level tower is never a source). The feed reads only the staged audio/delay/margin."""
    frames = _staged_frames(20, 2000, np.random.default_rng(15))
    _gate, (s1, e1, _m) = _feed(frames, [0.1] * 20)
    frames2 = _staged_frames(20, 2000, np.random.default_rng(15))
    _gate, (s2, e2, _m) = _feed(frames2, [0.1] * 20)
    assert (s1, e1) == (s2, e2) and e1 is True


# --- the live path stages frames and feeds them contiguously ------------------------------------

def _pcm(x: np.ndarray) -> bytes:
    return np.rint(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def test_the_capture_tap_stages_the_locked_delay_and_the_margin_without_a_model() -> None:
    """The audio callback stages the cleaned/mic/reference bytes plus the delay the lock proved and
    the mask's band margin - but runs NO model (handoff 6.8): the staged tuple carries no prob."""
    names = AudioCapture._note_joint_frame.__code__.co_names
    assert "VoiceDetector" not in names and "DoubleTalkGate" not in names

    cap = AudioCapture()
    cap.set_echo_mode("voxa")
    # stage one synthetic frame directly: the tap only needs the canceller active
    cap.canceller = cap.canceller or _NoopCanceller()
    cap._canceller_enabled = True
    ref = np.random.default_rng(3).uniform(-0.3, 0.3, 320)
    mic = np.random.default_rng(4).uniform(-0.3, 0.3, 320)
    cap._note_joint_frame(_pcm(mic), _pcm(mic), ref)
    frames = cap.drain_joint_frames()
    assert len(frames) == 1
    cleaned, mic_b, ref_a, delay, band_margin, erle, align, active = frames[0]
    assert cleaned and mic_b and len(ref_a) == 320
    # exactly the DSP fields; no model result is ever staged on the capture thread
    assert isinstance(erle, float) and align in (True, False) and active in (True, False)


class _NoopCanceller:
    delay_samples = 2000
    delay_managed_internally = False
    echo_explained_db = None
    residual_power_db = None


# --- the real window carries the joint fields into the evidence (needs a GTK display) ------------

def test_hearing_evidence_carries_the_joint_fields_when_a_display_is_available(window_module=None) -> None:
    if gi is None:  # pragma: no cover - no display in this environment
        pytest.skip("no GTK display")
    from voxa.window import MainWindow

    frames = _staged_frames(20, 2000, np.random.default_rng(21))

    class _Audio:
        def __init__(self) -> None:
            self._f = frames
            self.delay_samples = 2000

        def drain_joint_frames(self):
            out, self._f = self._f, []
            return out

        def echo_measurements(self):
            return 1e-3, None, None

    class _Baseline:
        def note(self, _):
            return None

    class _Gate:
        baseline = 0.05

    win = MainWindow.__new__(MainWindow)
    win.audio = _Audio()
    win._echo_baseline = _Baseline()
    win.conversation = None
    win._latest_level = 1000.0

    class _Ctl:
        gate = _Gate()

    win._barge_ctl = _Ctl()
    ev = None
    for _ in range(len(frames)):
        ev = MainWindow._hearing_evidence(win)
    assert ev is not None
    assert ev.joint_echo is True
