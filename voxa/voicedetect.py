"""Streaming voice detector built on the Silero VAD model shipped with faster-whisper, plus the
echo-aware near-end speech detector (roadmap task 5).

Unlike the loudness rule in :mod:`voxa.dictation` (which fires on any chunk above a
fixed level and therefore never sees a pause under background music), this looks at
the *shape* of the audio: the Silero model emits a per-window speech probability, so
music and TV talk score low while a human voice scores high.

Silero is a practical local speech-probability baseline, not an echo detector, speaker
verifier, or end-of-turn predictor (handoff 6.3): while Voxa plays, it runs on the
post-AEC, post-mask signal at a RAISED threshold, and its verdict is only one of the
three evidence streams the joint double-talk test ANDs with the reference.
"""

from __future__ import annotations

import math
import numpy as np

WINDOW = 512  # samples the model scores at a time (32 ms at 16 kHz)
CONTEXT = 64  # samples of the previous window the model re-reads as context


class VoiceDetector:
    """Streaming Silero voice detector over 16 kHz little-endian 16-bit PCM.

    :meth:`feed` takes one chunk of PCM bytes and returns whether the most recent
    model window (512 samples) was classified as speech at ``threshold``. The model
    state (the LSTM hidden/cell states and the 64-sample context) is carried across
    chunks, so a long utterance stays "voice" even when a chunk does not complete a
    window. :meth:`reset` clears that state.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        from faster_whisper.vad import get_vad_model

        self.threshold = float(threshold)
        self.session = get_vad_model().session
        self.probs: list[float] = []
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype="float32")
        self.c = np.zeros((1, 1, 128), dtype="float32")
        self.context = np.zeros((1, CONTEXT), dtype="float32")
        self.pending = np.zeros(0, dtype="float32")
        self.last_prob = 0.0
        self.probs = []

    def feed(self, pcm_int16_bytes: bytes) -> bool:
        samples = np.frombuffer(pcm_int16_bytes, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size:
            self.pending = np.concatenate((self.pending, samples))
        while self.pending.shape[0] >= WINDOW:
            row = self.pending[:WINDOW].reshape(1, WINDOW)
            self.pending = self.pending[WINDOW:]
            inp = np.concatenate((self.context, row), axis=1).reshape(1, WINDOW + CONTEXT)
            probs, self.h, self.c = self.session.run(
                None,
                {"input": inp, "h": self.h, "c": self.c},
            )
            self.context = row[:, -CONTEXT:]
            self.last_prob = float(probs[0])
            self.probs.append(self.last_prob)
        return self.last_prob >= self.threshold


# --- echo-aware near-end speech detection (roadmap task 5) -----------------------------------
# US10586534B1 and the iAEC line (handoff 6.3 [R2]): nobody big trusts a microphone VAD while
# the device is talking. Amazon judges every detection by the AEC's own statistics over the
# detection window - steady ERLE plus high reference-to-microphone correlation means the sound
# came from the loudspeaker; a sharp ERLE drop or correlation collapse means double talk. iAEC
# feeds the playback reference as an extra input to the speech detector, because a residual-echo
# probability is NOT 1 - user_probability: echo and a real user coexist.
#
# So the streams are ANDed, never summed: a second source is only "suspect" while
#   (a) the cleaned signal is speechful at a RAISED threshold (quiet residual echo must not
#       count as a user: 0.5 is the app's dictation threshold, playback uses 0.6-0.7), and
#   (b) the cleaned output stands above the mask's PREDICTED residual by a margin - the report's
#       "output energy >= predicted residual + margin", measured against the echo model, not
#       against a moving microphone baseline (which the residual's own -120 dB floor poisons), and
#   (c) the cleaned output is NOT consistent with the reference: short-time coherence between the
#       two (at the canceller's tracked delay) is low, or the canceller's running ERLE has dropped
#       against its own recent baseline (the patent's double-talk signal).
# All three for >= 150 ms raise SUSPECT; the suspect persists through joint gaps up to the 300 ms
# hold, which is what ignores transient residual bursts (a burst shorter than the hold neither
# confirms a user nor clears a real double talk between syllable dips). While suspect, the
# CALLER freezes the canceller through its own double-talk control (voxa/anc.py
# note_double_talk) - time-bounded, never a blanket permanent freeze.
#
# None is not zero: an unmeasured probability, margin, or coherence (no reference, no delay,
# alignment unhealthy) makes the joint test NOT fire. The caller then falls back to the level
# gate and the words, and says so - the degraded path, not a guess.

SPEECH_PLAYBACK_THRESHOLD = 0.65  # raised Silero threshold while TTS plays (report: 0.6-0.7)
SUSPECT_JOINT_MS = 150.0   # all three streams must hold this long to raise "suspect"
SUSPECT_HOLD_MS = 300.0    # the suspect holds through joint gaps this long: transient bursts
RESIDUAL_MARGIN_DB = 6.0   # cleaned output above the predicted residual = the user, not echo
COHERENCE_ECHO = 0.5       # cleaned-vs-reference coherence at/above this is echo consistency
ERLE_DROP_DB = 3.0         # running ERLE this far under its own recent baseline = double talk
COHERENCE_WINDOW = 512     # samples of the coherence window: 32 ms at 16 kHz, like Silero


class DoubleTalkGate:
    """Joint double-talk evidence over the cleaned signal and the reference, in ms of audio.

    Fed once per frame by whoever owns the frames (the bench runner, the live hearing worker):
    the Silero probability of the CLEANED (post-AEC, post-mask) signal, the cleaned and reference
    samples themselves (the coherence is computed here, per 32 ms window, at the canceller's
    tracked delay), the mask's output-vs-predicted-residual margin, and the canceller's running
    ERLE. It holds no transcript and decides nothing about the WORDS - conversation evidence
    (partials, stop cues, context) is the caller's third stream.
    """

    def __init__(self, *, speech_threshold: float = SPEECH_PLAYBACK_THRESHOLD,
                 joint_ms: float = SUSPECT_JOINT_MS, hold_ms: float = SUSPECT_HOLD_MS,
                 margin_db: float = RESIDUAL_MARGIN_DB,
                 coherence_echo: float = COHERENCE_ECHO,
                 erle_drop_db: float = ERLE_DROP_DB, sample_rate: int = 16000) -> None:
        self.speech_threshold = float(speech_threshold)
        self.joint_ms = float(joint_ms)
        self.hold_ms = float(hold_ms)
        self.margin_db = float(margin_db)
        self.coherence_echo = float(coherence_echo)
        self.erle_drop_db = float(erle_drop_db)
        self.sample_rate = int(sample_rate)
        self.reset()

    def reset(self) -> None:
        """A new reply (or a new clip): the run, the buffers and the ERLE baseline start over."""
        self._joint_ms = 0.0
        self._suspect = False
        self._gap_ms = 0.0
        self._echo_ms = 0.0
        self._echo_consistent = False
        self._suspect_rise_ms = 0.0     # audio seen when the joint run raised the suspect
        self._audio_ms = 0.0
        self._mic = np.zeros(0, dtype=np.float64)
        self._reference = np.zeros(0, dtype=np.float64)
        self._erle_baseline: float | None = None
        # What the last frame measured (None = this stage could not measure it).
        self.coherence: float | None = None
        self.conditions: dict = {}

    def observe(self, *, dt_ms: float, prob: float | None, mic: np.ndarray,
                ref: np.ndarray, delay_samples: int | None,
                residual_over_prediction_db: float | None, erle_db: float | None,
                reference_active: bool, alignment_healthy: bool) -> bool:
        """One frame: update the joint evidence, return whether the frame is SUSPECT.

        Suspect = all three streams held (>= ``joint_ms`` of them), and it persists through
        joint gaps shorter than ``hold_ms``.

        ``mic`` is the MICROPHONE the canceller was fed, not its output: the patent's echo test
        asks whether the microphone is still explained by the delayed reference (high coherence
        = echo, no second source). The cleaned signal is meaningless for that test - the echo was
        already removed from it - which is why the SPEAKING half (``prob``) and the MARGIN half
        (``residual_over_prediction_db``) arrive as scalars measured on the cleaned stream.
        """
        dt = max(0.0, float(dt_ms))
        self._audio_ms += dt
        mic = np.asarray(mic, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        if mic.size:
            self._mic = np.concatenate((self._mic, mic))
            self._reference = np.concatenate((self._reference, ref))
        # Keep enough history to align the coherence window against the reference at the
        # canceller's tracked delay: the reference readback reaches back ``delay_samples`` from
        # the end of the cleaned window, so the buffer must hold window + delay + headroom.
        need = COHERENCE_WINDOW + (int(delay_samples) if delay_samples else 0) + COHERENCE_WINDOW
        keep = max(2 * COHERENCE_WINDOW, need)
        if self._mic.size > keep + COHERENCE_WINDOW:
            self._mic = self._mic[-keep:]
            self._reference = self._reference[-keep:]

        # (c) echo consistency: short-time coherence at the canceller's tracked delay.
        self.coherence = self._coherence(delay_samples if alignment_healthy else None)
        cond_a = prob is not None and float(prob) >= self.speech_threshold
        cond_b = (residual_over_prediction_db is not None
                  and float(residual_over_prediction_db) >= self.margin_db)
        cond_c = False
        if self.coherence is not None and self.coherence < self.coherence_echo:
            cond_c = True
        if erle_db is not None and self._erle_baseline is not None:
            if self._erle_baseline - float(erle_db) >= self.erle_drop_db:
                cond_c = True
        # The ERLE baseline learns only from frames nobody claims as a second source (echo-only
        # frames): a baseline poisoned by double talk cannot "drop" during it.
        if (erle_db is not None and reference_active and alignment_healthy
                and not cond_a and not cond_b):
            e = float(erle_db)
            self._erle_baseline = e if self._erle_baseline is None else (
                0.95 * self._erle_baseline + 0.05 * e)

        measurable = (prob is not None and residual_over_prediction_db is not None
                      and self.coherence is not None and reference_active and alignment_healthy)
        joint = bool(measurable and cond_a and cond_b and cond_c)
        # The MIRROR test (the patent's other half): the microphone IS the delayed reference -
        # high coherence at the tracked delay, and nothing of its own in the cleaned signal.
        # A frame that is neither suspect nor echo-consistent is INDETERMINATE: the caller keeps
        # the fallback rule, because "the joint test did not fire" is not evidence of echo.
        cond_echo_frame = bool(measurable and not cond_c and self.coherence is not None
                               and self.coherence >= self.coherence_echo
                               and not (cond_a and cond_b))
        self.conditions = {"speech": cond_a, "residual": cond_b, "inconsistent": cond_c,
                           "consistent": cond_echo_frame, "measurable": measurable}
        if joint:
            self._joint_ms += dt
            self._gap_ms = 0.0
            if not self._suspect and self._joint_ms >= self.joint_ms:
                self._suspect = True
                self._suspect_rise_ms = self._audio_ms - dt
        else:
            self._joint_ms = 0.0
            if self._suspect:
                self._gap_ms += dt
                if self._gap_ms >= self.hold_ms:
                    self._suspect = False
                    self._gap_ms = 0.0
        if cond_echo_frame:
            self._echo_ms += dt
            if not self._echo_consistent and self._echo_ms >= self.joint_ms:
                self._echo_consistent = True
        else:
            self._echo_ms = 0.0
            if self._echo_consistent and not self._suspect:
                self._echo_consistent = False
        return self._suspect

    def _coherence(self, delay_samples: int | None) -> float | None:
        """Zero-mean normalized correlation of the cleaned window with the delayed reference.

        ``delay_samples`` None (alignment unhealthy, echo travel unknown) measures NOTHING:
        the joint gate does not fire, the caller degrades.
        """
        n = min(self._mic.size, self._reference.size)
        if delay_samples is None or n < COHERENCE_WINDOW:
            return None
        # The cleaned window ends at the stream position ``self._pos``; the echo inside it came
        # from the reference ``delay_samples`` earlier.
        x = self._mic[-COHERENCE_WINDOW:]
        # The reference window aligned to the cleaned one, shifted by the echo delay.
        end = n - int(delay_samples)
        if end < COHERENCE_WINDOW:
            return None
        r = self._reference[end - COHERENCE_WINDOW:end]
        x = x - float(np.mean(x))
        r = r - float(np.mean(r))
        ex = float(np.dot(x, x))
        er = float(np.dot(r, r))
        if ex < 1e-18 or er < 1e-18:
            return None
        return float(abs(np.dot(x, r)) / math.sqrt(ex * er))

    @property
    def suspect(self) -> bool:
        return self._suspect

    @property
    def echo_consistent(self) -> bool:
        """The mirror verdict: this span IS the delayed reference (high coherence, nothing of
        its own). It is the only finding allowed to SUPPRESS the caller's fallback gate -
        "not suspect" alone is indeterminate, never evidence of echo."""
        return self._echo_consistent

    @property
    def joint_run_ms(self) -> float:
        """The current run of joint evidence (0.0 = none)."""
        return self._joint_ms

    @property
    def suspect_ms(self) -> float:
        """How long the current suspect has held (0.0 = not suspect)."""
        return (self._audio_ms - self._suspect_rise_ms) if self._suspect else 0.0

    @property
    def erle_baseline_db(self) -> float | None:
        return self._erle_baseline
