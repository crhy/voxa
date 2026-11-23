"""Exact echo alignment from Voxa's own render timeline (roadmap Hearing 2, issue #72).

Voxa knows the exact samples she plays and when (the render timeline), so the echo delay
should be KNOWN, not estimated.  f130's :class:`voxa.anc.DelayTracker` estimates the delay
blind every 0.5 s and reports a coarse integer offset with a 1 ms hysteresis dead-band:
f132 measured that estimate drifting 1.5-2.5 ms, which made reference/mic coherence useless.

:class:`ExactAligner` keeps the delay LOCKED instead of re-estimating it:

* a prediction from the render timeline's sample clock plus the audio server's reported
  latencies gives a starting delay (``predict``); without one (the frozen room clips carry
  no device model) the first window does a bounded ``voxa.anc.gccphat_delay`` search;
* a fine search narrows the delay to sub-sample ONLY around the prediction, never a jump;
* a fractional-delay drift resampler - the locked delay plus its measured slope,
  extrapolated between measurements - keeps it locked through clock drift and silence;
* an abrupt path change that persists is an explicit epoch: the lock re-anchors, the slope
  is kept, and the drift resampler reconverges instead of the estimate drifting.

The locked delay (float) and its confidence are what every later stage - canceller,
residual mask, barge-in coherence - reads, so they all agree on one delay.
"""

from __future__ import annotations

import numpy as np

from .anc import SILENT_REFERENCE_POWER, DelayTracker, gccphat_delay

COHERENCE_MIN = 0.12  # overlap correlation at the locked lag: echo present, not a near-end accident


class ExactAligner:
    """A phase-locked loop on the echo delay: prediction + sub-sample refinement + drift.

    ``observe`` is fed aligned microphone/reference chunks with the time they were sampled
    at; it returns the locked delay in samples (float) after the call, or ``None`` before the
    first lock.  Between measurements the delay is held on the drift line ``locked +
    slope * (t - t_locked)`` (the fractional-delay resampler), never reset, never jumped.
    """

    def __init__(self, sample_rate: int = 16000, *, predict_s: float | None = None,
                 window_ms: float = 6.0, epoch_ms: float = 8.0, alpha: float = 0.5,
                 slope_alpha: float = 0.05, epoch_confirm: int = 2,
                 search_ms: float = 400.0, coherence_min: float = COHERENCE_MIN,
                 recent_ms: float = 400.0, jump_frac: float = 0.25) -> None:
        self._sr = int(sample_rate)
        self._window = max(2, int(round(window_ms / 1000.0 * self._sr)))
        self._epoch = float(epoch_ms) / 1000.0 * self._sr
        self._alpha = float(alpha)
        self._slope_alpha = float(slope_alpha)
        self._epoch_confirm = int(epoch_confirm)
        self._search_ms = float(search_ms)
        self._coherence_min = float(coherence_min)
        # f163: the GCC measures the RECENT window only (the echo path that is actually on the
        # mic now), not the whole rolling buffer, so a step's pre-step echo - which dominates a
        # whole-buffer peak for search_ms after the step - cannot keep reporting the OLD delay
        # as confident. |delta| >= jump_frac*window is step evidence, not refinement: the lock
        # snaps to the candidate only once a second frame agrees, and never refines halfway
        # across a step while claiming confidence.
        self._recent = int(round(recent_ms / 1000.0 * self._sr))
        self._jump = max(4, int(round(jump_frac * self._window)))
        self._locked: float | None = None
        self._slope = 0.0          # samples of extra delay per second (the clock drift)
        self._t_locked = 0.0       # seconds the slope is referenced to
        self._pending_epoch = 0.0  # an unconfirmed abrupt jump
        self._pending_n = 0
        self._stable = 0           # consecutive in-window refinements (a lock must prove it twice)
        self._t_now = 0.0
        # A rolling mic/reference buffer: a delay of up to ``search_ms`` means the correlation
        # window has to span the echo path, so each frame appends to it and the search/refine
        # runs over the buffer (the delay is essentially constant across one such window).
        self._keep = int(self._search_ms / 1000.0 * self._sr) + 3 * self._window + 1600
        self._buf_m: list[np.ndarray] = []
        self._buf_r: list[np.ndarray] = []
        self._buf_n = 0
        self.confidence = 0.0
        self.epochs = 0
        self.refined = False  # the last observe() refined the lock inside the narrow window
        self.drift_ppm = 0.0

    # -- prediction ---------------------------------------------------------------
    def predict(self, delay_samples: float, *, confidence: float = 1.0) -> None:
        """Seed the lock from the render timeline (its sample clock + reported latencies).

        A real prediction lands within a few ms of the true delay: the room's acoustic
        travel and the device jitter are all the refinement has left to chase, so it locks
        on the first playback frame instead of after a 0.5 s search.
        """
        self._locked = float(delay_samples)
        self._slope = 0.0
        self._t_locked = 0.0
        self.confidence = float(confidence)

    @property
    def locked_delay(self) -> float | None:
        """The locked echo delay in samples (fractional), or None before the first lock."""
        return None if self._locked is None else float(self._locked)

    @property
    def delay_samples(self) -> int:
        """The integer delay the stages and metrics report (0 = never locked)."""
        return 0 if self._locked is None else int(round(self._locked))

    def _predict_now(self, t_s: float) -> float:
        """The drift resampler: where the lock says the delay is at ``t_s`` right now."""
        return self._locked + self._slope * (t_s - self._t_locked)

    def _coherent(self, mic: np.ndarray, ref: np.ndarray, lag: int, tail: int = 0) -> bool:
        """Does the mic actually carry a copy of the reference delayed by ``lag``? (echo vs near-end)

        The buffer is time-aligned, so mic[t] pairs with ref[t-lag]. With ``tail`` set only the
        last ``tail`` samples of the mic are examined, so an echo that stopped being there (a
        step / path change) is detected even while older buffer still carries the pre-change echo.
        """
        if lag <= 0 or lag >= len(mic):
            return False
        start = max(lag, len(mic) - tail) if tail else lag
        end = len(mic)
        if end - start < 256:
            return False
        m = mic[start:end]
        r = ref[start - lag:end - lag]
        mm = float(np.dot(m, m))
        rr = float(np.dot(r, r))
        if mm <= 0.0 or rr <= 0.0:
            return False
        return float(np.dot(m, r)) ** 2 / (mm * rr) >= self._coherence_min ** 2

    def _measure(self, mic: np.ndarray, ref: np.ndarray, recent: bool = False) -> float | None:
        """Sub-sample delay from a robust GCC-PHAT over the buffer (room-tail resistant).

        This is the "fine search" of the task: it lands on the DIRECT echo path even when the
        room rings, and its parabolic peak is sub-sample. It is only ever used to refine the
        lock around the prediction (or to re-anchor it at an epoch); on its own it never moves
        the lock out of the narrow window the prediction allows. A second, comparable echo path
        in the buffer - a step whose pre- and post-step echoes are both still in the window, or a
        second talkspur - means the delay is not yet resolved, so it reports None and the caller
        holds the drift line rather than locking confidently onto one of them.

        f163: ``recent=True`` runs the SAME gated search over the last ``recent_ms`` of the
        rolling buffer only. Over the whole buffer a step's pre-step echo stays a top peak for
        search_ms after the step, so the OLD delay keeps measuring confidently while the mic is
        already echoing at the new one; over the recent window the pre-step energy ages out and
        the new path wins as soon as it dominates the recent sound - that early flip is the step
        EVIDENCE the observer uses to snap the epoch, while the whole-buffer measurement keeps
        feeding the refinement and the drift slope exactly as h02's numbers were built on.
        """
        if mic.size < 1600:
            return None
        n = min(mic.size, ref.size)
        mic, ref = mic[:n], ref[:n]
        if recent and n > self._recent:
            mic, ref = mic[n - self._recent:], ref[n - self._recent:]
        max_delay = int(self._search_ms / 1000.0 * self._sr)
        size = 1
        while size < n + max_delay:
            size <<= 1
        cross = np.fft.rfft(mic, size) * np.conj(np.fft.rfft(ref, size))
        cross /= np.abs(cross) + 1e-12
        corr = np.fft.irfft(cross, size)
        window = corr[:max_delay + 1]
        peak = int(np.argmax(window))
        floor = float(np.median(window))
        top = float(window[peak])
        if floor > 0.0:
            if top < 8.0 * floor:
                return None
        elif top <= 0.0:
            return None
        band = 2 * self._window + 8
        lo, hi = max(0, peak - band), min(window.size, peak + band + 1)
        # A second, comparable echo path far from the resolved one (a step whose pre- and
        # post-step echoes are both still in the buffer, or a second talkspur) means the delay
        # is not yet resolved. The room tail that hugs the peak is not a second path, so only a
        # distant peak counts, and only a comparable one.
        far_lo, far_hi = max(0, peak - self._sr // 25), min(window.size, peak + self._sr // 25)
        rest = np.concatenate((window[:far_lo], window[far_hi:]))
        if rest.size and float(rest.max()) >= 0.5 * top:
            return None  # an unresolved second path: not confident, hold the drift line
        if 0 < peak < window.size - 1:
            left, mid, right = window[peak - 1], window[peak], window[peak + 1]
            denom = left - 2.0 * mid + right
            if denom != 0.0:
                peak = peak + float(np.clip(0.5 * (left - right) / denom, -1.0, 1.0))
        if not self._coherent(mic, ref, int(round(peak))):
            return None
        # The echo must still be at the resolved lag in the RECENT data: a step whose pre-step
        # energy still dominates the whole-buffer peak would otherwise be reported confidently,
        # so this recent check turns the reconvergence into "not resolved" (None) until the new
        # path is the one actually on the mic.
        if not self._coherent(mic, ref, int(round(peak)), tail=self._window + self._sr // 4):
            return None
        return float(peak)

    def _push(self, mic: np.ndarray, ref: np.ndarray) -> None:
        self._buf_m.append(mic)
        self._buf_r.append(ref)
        self._buf_n += mic.size
        while self._buf_n > self._keep and self._buf_m:
            drop = self._buf_m[0]
            self._buf_m.pop(0)
            self._buf_r.pop(0)
            self._buf_n -= drop.size

    def _window_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        mic = np.concatenate(self._buf_m) if self._buf_m else np.zeros(0)
        ref = np.concatenate(self._buf_r) if self._buf_r else np.zeros(0)
        return mic, ref

    def observe(self, t_s: float, mic, ref) -> float | None:
        """Advance the lock with one aligned (mic, ref) chunk sampled at absolute ``t_s``."""
        self._t_now = float(t_s)
        self.refined = False
        mic = np.asarray(mic, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        n = min(mic.size, ref.size)
        if n == 0:
            return self.locked_delay
        mic, ref = mic[:n], ref[:n]
        self._push(mic, ref)
        buf_m, buf_r = self._window_arrays()
        # Silence: hold the lock on the drift line (the resampler); never jump, never reset.
        if float(np.mean(ref * ref)) <= SILENT_REFERENCE_POWER:
            if self._locked is not None:
                self._locked = self._predict_now(self._t_now)
                self._t_locked = self._t_now
                self._slope *= 0.98  # a held slope decays toward the drift, never extrapolates far
            return self.locked_delay
        # Not yet locked: a bounded GCC-PHAT search over the buffer finds the first delay (the
        # frozen room clips carry no device-latency prediction to seed), then every later call
        # refines the lock narrow around it.
        if self._locked is None:
            est = self._measure(buf_m, buf_r)
            if est is None:
                return None
            self._locked = float(est)
            self._t_locked = self._t_now
            self._stable = 0
            self.confidence = 1.0
            return self.locked_delay
        predicted = self._predict_now(self._t_now)
        measured = self._measure(buf_m, buf_r, recent=False)
        if measured is None:
            # A window with no clean echo (silent / near-end dominated): keep the lock on the
            # drift line, never jump to a guess.
            self._pending_n = 0
            self._locked = predicted
            self._t_locked = self._t_now
            return self.locked_delay
        measured_r = self._measure(buf_m, buf_r, recent=True)
        delta = measured - predicted
        # f163: a disagreement larger than a drift-sized refinement is STEP EVIDENCE, whether or
        # not it fits inside the search window. The two hypotheses are the lock (the pre-step
        # path) and the candidate (the render timeline's predicted post-step delay is exactly
        # this recent-window measurement, which flips to the new path search_ms earlier than the
        # whole-buffer peak can): the frame refines as h02 did (the canceller needs to get BACK
        # to the echo, and h02's reconverge/drift numbers are built on that), the candidate is
        # kept, and a second agreeing frame snaps the lock to it as an epoch. Every frame across
        # a step stays unconfirmed - never "confident while still on the pre-step delay", which
        # is what h02 measured and f163 must fix.
        candidate = measured_r if measured_r is not None else measured
        # f163 v4: the candidate only counts as step evidence once the PRE-step path is actually
        # gone from the newest audio. A recent-window peak that merely correlates - user speech
        # in a pair clip latching the 400 ms GCC to a short lag while the echo at the locked lag
        # is still there - is not a step: the old lag stays coherent over the last few tens of ms
        # and no epoch fires. A real step empties the old lag out of the mic (the render sends
        # pre-step audio no longer echoed at it), which is exactly when the snap belongs.
        step_evidence = (abs(delta) >= self._jump or abs(candidate - predicted) >= self._jump) \
            and not self._coherent(buf_m, buf_r, int(round(predicted)),
                                   tail=max(256, self._sr // 32))
        if abs(delta) <= self._window:
            # Refine around the prediction (never a jump): phase-lock toward the measurement and
            # let the long-term difference update the drift (the resampler's slope).
            self._locked = predicted + self._alpha * delta
            dt = max(self._t_now - self._t_locked, 1.0 / self._sr)
            inst_slope = delta / dt
            self._slope = self._slope + self._slope_alpha * (inst_slope - self._slope)
            self._t_locked = self._t_now
            self.drift_ppm = self._slope / self._sr * 1e6
            self.confidence = 1.0
            if step_evidence:
                if self._pending_n and abs(candidate - self._pending_epoch) <= self._jump:
                    self._locked = float(candidate)
                    self._t_locked = self._t_now
                    # A delay STEP changes the echo path, not the clock: the drift slope is
                    # kept (h02's out-of-window epoch zeroes it because there the device may
                    # have moved too; here the clock did not).
                    self.epochs += 1
                    self._pending_n = 0
                else:
                    self._pending_epoch = float(candidate)
                    self._pending_n = 1
                self._stable = 0
            elif measured_r is None:
                # The recent window cannot resolve the path yet (a second path still ringing,
                # a quiet moment): refinement continues but the lock is NOT re-proven - a
                # frame is never confident on a delay the recent sound cannot confirm.
                self._stable = 0
            else:
                self._pending_n = 0
                self._stable += 1
            # A lock is only trustworthy once it has refined inside the narrow window on
            # CONSECUTIVE frames: a single in-window hit is a first lock / a spurious
            # near-silence peak, not a phase-lock (this is what kept f132's estimate honest).
            self.refined = self._stable >= 2
            return self.locked_delay
        # Out of the search window: h02's epoch path, re-anchored only on a second agreeing
        # frame (the two-frame re-anchor guard stays).
        if self._pending_n and abs(measured - self._pending_epoch) <= self._window:
            self._locked = float(measured)
            self._t_locked = self._t_now
            self._slope = 0.0
            self.epochs += 1
            self._pending_n = 0
            self._stable = 0
        else:
            self._pending_epoch = float(measured)
            self._pending_n = 1
            self._locked = predicted
            self._t_locked = self._t_now
            self.confidence = 0.5 * self.confidence
        return self.locked_delay


class ExactDelayLock:
    """The shipping wiring of the exact delay (issue #72, task f168): lock, honest fallback.

    Production capture (``voxa.audio.AudioCapture``) and the bench's ``voxa`` echo mode now
    drive the canceller's reference through ONE component, so the bench measures the shipping
    path instead of a private arm:

    * while the :class:`ExactAligner` lock is proven (refined inside its narrow window on
      consecutive frames), the canceller is re-pointed at the locked fractional offset -
      :meth:`voxa.anc.EchoCanceller.set_offset` anchors are the drift resampler, so the filter,
      convergence and double-talk state never rebuild, per sentence or otherwise;
    * while the lock cannot prove itself, the shipping :class:`voxa.anc.DelayTracker` estimate
      keeps the reference pointed and ``degraded`` is visible to the app: the canceller is never
      left blind and the microphone is never silently passed through raw;
    * :meth:`epoch` is the explicit route/rate-change reset. A sentence's silence and echo tail
      never are: the lock holds on its drift line and the estimate stays open across the pauses.

    ``observe`` returns the state it applied - "lock", "estimate" or "hold" - so the caller can
    log the transition exactly once instead of smearing it into the steady-state stream.
    """

    def __init__(self, sample_rate: int = 16000, canceller=None) -> None:
        self._sr = int(sample_rate)
        self.canceller = canceller
        self.aligner = ExactAligner(sample_rate=self._sr)
        self.tracker = DelayTracker(sample_rate=self._sr)
        self.degraded = False  # the estimate (not the lock) last re-pointed the canceller
        self.applied = False   # any offset has ever been fed to the canceller

    def epoch(self) -> None:
        """An explicit route/rate change: re-anchor the lock and the estimate together."""
        self.aligner = ExactAligner(sample_rate=self._sr)
        self.tracker = DelayTracker(sample_rate=self._sr)
        self.degraded = False
        self.applied = False

    @property
    def delay_frac(self) -> float | None:
        """The locked reference read offset in samples (fractional), or None before any lock."""
        return self.aligner.locked_delay

    @property
    def delay_samples(self) -> int:
        """The integer offset the stages and metrics report (0 = never aligned)."""
        return self.aligner.delay_samples

    @property
    def delay_confident(self) -> bool:
        """True only while the lock has refined inside its narrow window (never on a guess)."""
        return bool(self.aligner.refined)

    @property
    def drift_ppm(self) -> float:
        return self.aligner.drift_ppm

    def observe(self, t_s: float, mic, ref) -> str:
        """Feed one aligned (microphone, reference) frame sampled at ``t_s``; re-point the
        canceller's reference through the lock when proven, the estimate when not."""
        self.aligner.observe(t_s, mic, ref)
        if self.aligner.refined and self.aligner.locked_delay is not None:
            locked = float(self.aligner.locked_delay)
            if self.canceller is not None:
                self.canceller.set_offset(int(round(locked)), 0.0, locked, 0.0, locked)
            self.degraded = False
            self.applied = True
            return "lock"
        update = self.tracker.observe(mic, np.asarray(ref, dtype=np.float32))
        if update is not None:
            if self.canceller is not None:
                self.canceller.set_offset(update["delay_samples"], *update["anchors"])
            self.degraded = True
            self.applied = True
            return "estimate"
        return "hold"
