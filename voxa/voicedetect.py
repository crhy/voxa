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
# Roadmap task 6: the Silero off-threshold sits this far below the on-threshold (report line 177:
# "lower Silero's off-threshold (neg threshold ~ threshold - 0.15)"). With it, a 200 ms trailing pad
# (SILERO_TRAILING_PAD_MS) is the shortest pause the caller treats as the end of a turn.
SILERO_OFF_MARGIN = 0.15
SILERO_TRAILING_PAD_MS = 200.0


def silero_hysteresis(voiced: bool, prob: float, on_threshold: float,
                      off_margin: float = SILERO_OFF_MARGIN) -> bool:
    """One step of the Silero on/off state machine (pure, no model): speech begins at
    ``on_threshold`` and ends only ``off_margin`` below it. ``prob`` is the window's speech
    probability. This is the "off-threshold ~ threshold - 0.15" of roadmap task 6."""
    if prob >= on_threshold:
        return True
    if prob < on_threshold - off_margin:
        return False
    return voiced


class VoiceDetector:
    """Streaming Silero voice detector over 16 kHz little-endian 16-bit PCM.

    :meth:`feed` takes one chunk of PCM bytes and returns whether the most recent
    model window (512 samples) was classified as speech at ``threshold``. The model
    state (the LSTM hidden/cell states and the 64-sample context) is carried across
    chunks, so a long utterance stays "voice" even when a chunk does not complete a
    window. :meth:`reset` clears that state.
    """

    def __init__(self, threshold: float = 0.5, *, off_margin: float = SILERO_OFF_MARGIN) -> None:
        from faster_whisper.vad import get_vad_model

        self.threshold = float(threshold)
        # Hysteresis (roadmap task 6): once speech has begun the detector stays voiced until the
        # probability falls this far BELOW the on-threshold. Silero's per-window probability dips
        # under a soft last syllable or a mid-word breath; a single threshold cuts the turn there
        # and eats the user's last word (premature_cutoff_rate). An off-threshold of
        # ``threshold - 0.15`` keeps the run of speech together through those dips.
        self.off_margin = float(off_margin)
        self.session = get_vad_model().session
        self.probs: list[float] = []
        self.reset()

    @property
    def off_threshold(self) -> float:
        """Speech ends only below this probability (the hysteresis floor)."""
        return self.threshold - self.off_margin

    @property
    def voiced(self) -> bool:
        """The Silero hysteresis state: True while a run of cleaned-signal speech is still
        open (it rose at ``threshold`` and has not yet fallen the ``off_margin`` below it).
        This is the f183 "gated speech is still arriving" signal for the endpoint arbitrator:
        a mid-word/soft-syllable probability dip inside a run of speech keeps this True, so a
        Smart Turn "complete" verdict over such a pause is withheld to the next pause / hard cap,
        while a genuine pause (below the hysteresis floor) clears it and lets the turn close."""
        return self._voiced

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype="float32")
        self.c = np.zeros((1, 1, 128), dtype="float32")
        self.context = np.zeros((1, CONTEXT), dtype="float32")
        self.pending = np.zeros(0, dtype="float32")
        self.last_prob = 0.0
        self.probs = []
        self._voiced = False

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
        # Hysteresis: rise at ``threshold``, fall only at ``off_threshold`` (= threshold - 0.15), so
        # a brief probability dip mid-word does not end the run of speech and truncate a word.
        self._voiced = silero_hysteresis(self._voiced, self.last_prob, self.threshold, self.off_margin)
        return self._voiced


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
COHERENCE_ECHO = 0.2       # coherence at/above this is echo consistency (f187: the room leaves
                           # 0.04-0.46 between the raw microphone and the delayed reference on
                           # PURE echo here; 0.5 never saw the echo, so the echo mirror of the
                           # joint gate could not latch and residual bursts stayed "inconsistent")
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
        self._echo_gap_ms = 0.0
        self._echo_consistent = False
        # ``_half_*`` (f188, issue #72): the TWO of the three joint streams that speak about
        # the sound itself - residual margin over the prediction AND inconsistency with the
        # reference at the LOCKED delay - held together for joint_ms. That is the joint
        # evidence building toward the suspect (the seam that reassigned smoke-0020's real
        # barge-in to echo at COHERENCE_ECHO 0.2 - f187's missed_bargein regression): while it
        # claims the span no echo frames accumulate, so an echo mirror that has not latched
        # yet cannot contradict it. It never UNDOES an echo mirror that latched FIRST (the
        # smoke-0039 pause: the mirror locked on 0.78-0.87 coherence, and her own re-heard
        # "Thank you" residue burst later - the residue, not the verdict, is the failure).
        # The THIRD stream (Silero speech) is not in the half: quiet residual bursts hold no
        # half-latch and raise no suspect (the triple and the hold are untouched).
        self._half_ms = 0.0
        self._half_gap_ms = 0.0
        self._half_latched = False
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
        # f188: while the joint evidence claims the span - the triple running, or its residual+
        # inconsistency half held for joint_ms - no echo frames accumulate: the mirror may not
        # latch behind joint evidence that contradicts it. It never loosens the suspect: the
        # triple and its 300 ms hold above are untouched; only what the echo mirror may CLAIM
        # (before it latched) changes.
        cond_half = bool(cond_b and cond_c)
        if cond_half:
            self._half_ms += dt
            self._half_gap_ms = 0.0
            if not self._half_latched and self._half_ms >= self.joint_ms:
                self._half_latched = True
        else:
            self._half_ms = 0.0
            if self._half_latched:
                self._half_gap_ms += dt
                if self._half_gap_ms >= self.hold_ms:
                    self._half_latched = False
                    self._half_gap_ms = 0.0
        joint_claims = bool(self._suspect or self._half_latched or self._joint_ms > 0.0)
        if cond_echo_frame and not joint_claims:
            self._echo_ms += dt
            self._echo_gap_ms = 0.0
            if not self._echo_consistent and self._echo_ms >= self.joint_ms:
                self._echo_consistent = True
        else:
            self._echo_ms = 0.0
            # The MIRROR holds through short gaps exactly like the suspect does: a momentary
            # coherence dip (a word boundary, a quiet frame) is not evidence of a second
            # source, it just fails to re-affirm echo. Only a gap past hold_ms clears it.
            if self._echo_consistent:
                self._echo_gap_ms += dt
                if self._echo_gap_ms >= self.hold_ms:
                    self._echo_consistent = False
                    self._echo_gap_ms = 0.0
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
        "not suspect" alone is indeterminate, never evidence of echo.

        f188 (issue #72): the mirror never contradicts a LIVE suspect - the SAME gate that
        raised the joint three-stream verdict cannot also hand the caller an echo verdict for
        that span (the seam that reassigned a genuine user barge-in, frozen smoke-0020, to
        echo once COHERENCE_ECHO narrowed to 0.2). The residual+inconsistency HALF only stops
        the mirror from LATCHING (see ``observe``); an echo mirror that latched first stands,
        because on a pause clip that evidence is the canceller's residue, not a second source
        (frozen smoke-0039's re-heard "Thank you").
        """
        if self._suspect:
            return False
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
