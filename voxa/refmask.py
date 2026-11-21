"""Reference mask: use the speaker reference to predict and mask echo.

The mask learns, per frequency band, how loud the reference arrives at the
microphone (coupling). It predicts echo power in every time-frequency cell and
turns cells down when the predicted echo dominates. The user is speaking only
when the microphone holds clearly more energy than the predicted echo.
"""

from collections import deque

import numpy as np


class ReferenceMask:
    def __init__(self, rate=16000, frame=512, hop=256, bands=64, over=1.5,
                 floor=0.1, memory_s=0.4, tail: bool = False, rt_s: float | None = None):
        """
        tail=False (the default, and every existing caller): predict echo from the reference NOW.

        tail=True (roadmap task 2, f129): the echo of speech that ENDED a moment ago is still in
        the room - the clip sets carry RT60 of 0.2-0.8 s - and a mask that only looks at the
        current reference lets that tail through untouched. The prediction then HOLDS the
        reference and lets it decay at the room's own rate: ``rt_s`` when given, else the value
        from ``rt_grid`` that best fits the observed microphone decay. The held reference also
        makes the near-end detector readable while she is playing: ``last_excess_frac`` is the
        share of the microphone that the held echo model cannot explain - the user.
        """
        self.rate = int(rate)
        self.frame = int(frame)
        self.hop = int(hop)
        self.nbands = int(bands)
        self.over = float(over)
        self.floor = float(floor)
        self.tail = bool(tail)
        self.rt_grid = (0.2, 0.35, 0.5, 0.7, 0.9) if rt_s is None else (float(rt_s),)
        self.helds = [np.zeros(self.nbands) for _ in self.rt_grid]
        self.decay = [10.0 ** (-6.9 * (hop / rate) / rt) for rt in self.rt_grid]
        self.rt_scores = [0.0, 0.0, 0.0, 0.0, 0.0][:len(self.rt_grid)]
        self.rt_frames = [0] * len(self.rt_grid)
        self.rt_idx = min(2, len(self.rt_grid) - 1)
        self.mem_frames = max(4, int(memory_s * self.rate / self.hop))
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.frame) / self.frame)
        self.win = np.sqrt(hann)
        self.freqs = np.fft.rfftfreq(self.frame, 1.0 / self.rate)
        edges = np.linspace(1, self.frame // 2 + 1, self.nbands + 1).astype(int)
        self.band_idx = [
            np.arange(edges[b], edges[b + 1]) for b in range(self.nbands)
        ]
        self.coupling = np.zeros(self.nbands)
        self.ratios = [deque(maxlen=self.mem_frames) for _ in range(self.nbands)]
        # Roadmap task 2 (tail=True) learns the coupling only from bands where the reference is
        # STRONG (>= a quarter of the loudest band): on the raw microphone, quiet bands are mostly
        # room noise and would poison the ratio. Roadmap task 4 (f131) feeds the mask the AEC's
        # OWN output - quiet bands there are quiet residual, not noise - and a band that never
        # learns a ratio predicts zero echo and is protected by rule, which is how echo leaks.
        # ResidualMask relaxes this to every band with any reference energy.
        self.ratio_rel_floor = 0.25 if self.tail else 0.0
        # The coupling each band keeps is a percentile of its recent output-vs-reference ratios.
        # Roadmap task 2 (raw mic) used the MEDIAN: the echo model had to survive room noise.
        # Roadmap task 4 (ResidualMask) feeds the canceller's OWN output, where the residue of an
        # adaptive filter jumps 10-20 dB frame to frame; a median prediction calls those jumps
        # "near-end exceeds prediction by 6 dB" and protects the echo. ResidualMask raises the
        # percentile instead.
        self.ratio_quantile = 0.5
        # A second coupling at a higher quantile: the most residue this band has RECENTLY been
        # seen to leave. Roadmap task 4's protection margin is measured against THAT, so a band
        # only counts as near-end when it holds clearly more than the loudest echo this band ever
        # explained - a jump from an adaptive filter's residue cannot fake it.
        self.ratio_quantile_hi = 0.5
        self.coupling_hi = np.zeros(self.nbands)
        self._held_now = np.zeros(self.nbands)
        self._ratio_skip = np.zeros(self.nbands, dtype=bool)
        self.lag = 0
        self.last_user_db = -120.0
        self.last_echo_active = False
        self.last_excess = 0.0
        self.last_excess_frac = 0.0
        self.mic_buf = np.zeros(0, dtype=np.float64)
        self.ref_buf = np.zeros(0, dtype=np.float64)
        self.out_num = np.zeros(0, dtype=np.float64)
        self.out_den = np.zeros(0, dtype=np.float64)
        self.frames_done = 0
        self.emitted = 0
        # Measured on the most recent analysed frame, for attribution (powers in dBFS): what the
        # microphone holds, what the reference predicts at the echo's moment, and the residual
        # left after that prediction. Read-only evidence; nothing here changes the signal.
        self._mic_power = 0.0
        self._echo_power = 0.0

    def _estimate_lag(self):
        n = min(len(self.mic_buf), len(self.ref_buf))
        if n < self.frame * 4:
            return self.lag
        mic = self.mic_buf[-n:]
        ref = self.ref_buf[-n:]
        size = 1
        while size < 2 * n:
            size <<= 1
        fm = np.fft.rfft(mic, size)
        fr = np.fft.rfft(ref, size)
        corr = np.fft.irfft(fm * np.conj(fr), size)
        k = int(np.argmax(np.abs(corr)))
        if k > size // 2:
            k -= size
        return int(k)

    def _held_prediction(self, ref_b: np.ndarray, mic_b: np.ndarray) -> tuple[np.ndarray, int]:
        """Predict echo power per band from the reference HELD at the room's own decay (tail=True).

        Every candidate decay keeps its own held reference; the one whose held prediction explains
        the microphone best on the frames where the reference just went quiet (the tail) wins. The
        microphone is never learned from here - only the reference history and its decay are.
        """
        ref_total = float(ref_b.sum())
        for k, h in enumerate(self.helds):
            np.maximum(h * self.decay[k], ref_b, out=h)
        for k, h in enumerate(self.helds):
            if ref_total < 0.5 * float(h.sum()):  # the tail: speakers quiet, room still ringing
                pred = float((self.coupling * h).sum())
                err = abs(float(mic_b.sum()) - pred) / max(float(mic_b.sum()), 1e-12)
                self.rt_scores[k] += err
                self.rt_frames[k] += 1
        if any(self.rt_frames):
            self.rt_idx = min(range(len(self.rt_grid)),
                              key=lambda k: (self.rt_scores[k] / f if (f := self.rt_frames[k]) else 9e9))
        return self.coupling * self.helds[self.rt_idx], self.rt_idx

    def _note_ref_activity(self, ref_total: float) -> None:
        """Hook: called with the reference band powers on every analysed frame."""
        return None

    def _update_ratio_gate(self, mic_b: np.ndarray, echo_b: np.ndarray) -> None:
        """Hook: may set ``self._ratio_skip[b] = True`` to keep band b's output-vs-reference
        ratio from learning this frame — roadmap task 4 uses it so a band that looks like
        NEAR-END speech (protected by the margin rule) never poisons the echo model.
        Default: every band learns every frame."""
        return None

    def _band_gain(self, mic_b: np.ndarray, echo_b: np.ndarray) -> np.ndarray:
        """Hook: the per-band gain for this frame (1.0 = keep, floor = suppress)."""
        return np.where(mic_b >= self.over * echo_b, 1.0, self.floor)

    def _frame_stats(self, mic_frame, ref_frame):
        fm = np.fft.rfft(mic_frame)
        fr = np.fft.rfft(ref_frame)
        pm = np.abs(fm) ** 2
        pr = np.abs(fr) ** 2
        mic_b = np.array([pm[idx].sum() if idx.size else 0.0 for idx in self.band_idx])
        ref_b = np.array([pr[idx].sum() if idx.size else 0.0 for idx in self.band_idx])
        return fm, pm, mic_b, ref_b

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        if len(ref) == 0 or float(np.max(np.abs(ref))) == 0.0:
            return np.asarray(mic, dtype=np.float32).copy()
        self.mic_buf = np.concatenate([self.mic_buf, mic])
        self.ref_buf = np.concatenate([self.ref_buf, ref])
        total = len(self.mic_buf)
        if total > len(self.out_num):
            self.out_num = np.concatenate([self.out_num, np.zeros(total - len(self.out_num))])
            self.out_den = np.concatenate([self.out_den, np.zeros(total - len(self.out_den))])
        if total >= self.frame:
            nframes = (total - self.frame) // self.hop + 1
            if nframes > self.frames_done and (
                self.frames_done == 0 or (nframes - self.frames_done) % 8 == 0
            ):
                self.lag = self._estimate_lag()
            for f in range(self.frames_done, nframes):
                s = f * self.hop
                mf = self.mic_buf[s:s + self.frame]
                rs = s - self.lag
                rs = max(0, min(rs, max(0, len(self.ref_buf) - self.frame)))
                rf = self.ref_buf[rs:rs + self.frame]
                fm, pm, mic_b, ref_b = self._frame_stats(mf, rf)
                if self.tail:
                    echo_b, self.rt_idx = self._held_prediction(ref_b, mic_b)
                    self._held_now = self.helds[self.rt_idx]
                else:
                    echo_b = self.coupling * ref_b
                    self._held_now = ref_b
                self._mic_power = float(pm.sum())
                self._echo_power = float(echo_b.sum())
                # how much of the microphone the echo model cannot explain (roadmap task 2's
                # near-end evidence: the user, not her own tail)
                self.last_excess = float(np.sum(np.maximum(mic_b - self.over * echo_b, 0.0)))
                self.last_excess_frac = self.last_excess / max(self._mic_power, 1e-12)
                rel = np.max(ref_b) if self.ratio_rel_floor else 0.0
                self._ratio_skip = np.zeros(self.nbands, dtype=bool)
                self._update_ratio_gate(mic_b, echo_b)
                skip = self._ratio_skip
                for b in range(self.nbands):
                    if ref_b[b] > 1e-10 and ref_b[b] >= self.ratio_rel_floor * rel and not skip[b]:
                        self.ratios[b].append(mic_b[b] / ref_b[b])
                        r = sorted(self.ratios[b])
                        self.coupling[b] = r[min(len(r) - 1, int(len(r) * self.ratio_quantile))]
                        self.coupling_hi[b] = r[min(len(r) - 1, int(len(r) * self.ratio_quantile_hi))]
                self._note_ref_activity(float(ref_b.sum()))
                gain_b = self._band_gain(mic_b, echo_b)
                mag = np.sqrt(pm)
                phase = np.angle(fm)
                band_gain = np.ones(pm.shape[0])
                for b in range(self.nbands):
                    band_gain[self.band_idx[b]] = gain_b[b]
                out_spec = (mag * band_gain) * np.exp(1j * phase)
                out_spec[0] = mag[0] * band_gain[0]
                rebuilt = np.fft.irfft(out_spec, self.frame)
                self.out_num[s:s + self.frame] += rebuilt * self.win
                self.out_den[s:s + self.frame] += self.win ** 2
            self.frames_done = nframes
        emit_end = max(self.emitted, total - self.frame + 1)
        emit_end = min(emit_end, total)
        out = np.zeros(emit_end - self.emitted, dtype=np.float32)
        for i, g in enumerate(range(self.emitted, emit_end)):
            d = self.out_den[g] if g < len(self.out_den) else 0.0
            out[i] = np.float32(self.out_num[g] / d) if d > 1e-9 else np.float32(self.mic_buf[g])
        self.emitted = emit_end
        if self.frames_done:
            f = self.frames_done - 1
            s = f * self.hop
            mf = self.mic_buf[s:s + self.frame]
            rs = max(0, min(s - self.lag, max(0, len(self.ref_buf) - self.frame)))
            rf = self.ref_buf[rs:rs + self.frame]
            _fm, pm, mic_b, ref_b = self._frame_stats(mf, rf)
            echo_b = self.coupling * ref_b
            p = max(0.0, float(mic_b.sum() - echo_b.sum()))
            self.last_user_db = 10.0 * float(np.log10(p)) if p > 1e-20 else -120.0
            self.last_echo_active = bool(float(echo_b.sum()) > 1e-8)
        return out

    def user_energy_db(self) -> float:
        return self.last_user_db

    def mic_power_db(self) -> float:
        """Energy the microphone delivered on the last analysed frame, in dB."""
        return 10.0 * float(np.log10(self._mic_power)) if self._mic_power > 1e-20 else -120.0

    def echo_explained_db(self) -> float:
        """How much of the microphone the reference already explains, in dB.

        0.0 dB means the reference accounts for ALL of it (positive echo evidence); a strongly
        negative value means the microphone holds sound the speakers did not play.
        """
        if self._mic_power <= 1e-20:
            return -120.0
        return 10.0 * float(np.log10(max(self._echo_power, 1e-20) / self._mic_power))

    def echo_active(self) -> bool:
        return self.last_echo_active

    def reset(self) -> None:
        self.coupling = np.zeros(self.nbands)
        self.ratios = [deque(maxlen=self.mem_frames) for _ in range(self.nbands)]
        self._ratio_skip = np.zeros(self.nbands, dtype=bool)
        self.lag = 0
        self.last_user_db = -120.0
        self.last_echo_active = False
        self.mic_buf = np.zeros(0, dtype=np.float64)
        self.ref_buf = np.zeros(0, dtype=np.float64)
        self.out_num = np.zeros(0, dtype=np.float64)
        self.out_den = np.zeros(0, dtype=np.float64)
        self.frames_done = 0
        self.emitted = 0


class ResidualMask(ReferenceMask):
    """Residual echo suppression ON THE AEC OUTPUT (roadmap task 4, f131).

    ``ReferenceMask`` predicted the echo in the MICROPHONE and ducked the cells it
    dominated. The full stack has a second, quieter echo: whatever the linear
    canceller FAILED to remove. This mask runs on the canceller's own output with
    the same aligned reference:

    * the predicted residual is the median output-vs-reference ratio per band
      (what the canceller leaves of the reference), held at the room's own decay
      so the tail after TTS is predicted too;
    * the mask is active while the reference plays and for ``tail_s`` after
      (roadmap task 4: TTS + 300 ms); outside that window it never touches the
      near end;
    * over-subtraction rises when the running ERLE is low (cheap speakers, high
      volume: a canceller that removes little leaves more echo, so subtract
      deeper): ``over`` is the base gain and ``over_ramp`` extra dB of
      suppression per dB of ERLE shortfall against ``erle_target_db``;
    * a band whose output exceeds the predicted residual by ``protect_db`` is
      the near-end speaker, not residual echo: it stays untouched (>= 6 dB by
      the report's rule), and so does every band where no echo is predicted at
      all — the user is never deleted for being quiet.

    ``process(aec_output, ref)`` takes the canceller's cleaned stream (float
    -1..1) and the aligned reference; the running ERLE of the canceller is fed
    in with :meth:`set_running_erle` so the over-subtraction can rise.
    """

    def __init__(self, rate=16000, frame=512, hop=256, bands=64, over=0.05,
                 protect_db=6.0, floor=0.1, memory_s=0.4, tail_s=0.3,
                 erle_target_db=20.0, over_ramp=0.5, ref_floor=1e-9, rt_s=None):
        super().__init__(rate=rate, frame=frame, hop=hop, bands=bands, over=over,
                         floor=floor, memory_s=memory_s, tail=True, rt_s=rt_s)
        self.protect = 10.0 ** (float(protect_db) / 10.0)
        # The input here is the canceller's OWN output: every band with reference energy learns
        # its output-vs-reference ratio, including the quiet ones (the nonlinear residue of cheap
        # speakers lives exactly where the reference is quiet).
        self.ratio_rel_floor = 0.0
        # Predict the residual at the median of what this band recently left; protect only above
        # the 98th percentile, the loudest residue it was ever seen to leave.
        self.ratio_quantile = 0.5
        self.ratio_quantile_hi = 0.98
        # A frame whose protected energy exceeds this share of the frame is near-end speech:
        # the echo model learns nothing from it.
        self.freeze_frac = 0.5
        self._frame_user = np.zeros(self.nbands, dtype=bool)
        self._tail_s = float(tail_s)
        self.erle_target_db = float(erle_target_db)
        self.over_ramp = float(over_ramp)
        self.ref_floor = float(ref_floor)
        self._erle_db = 0.0
        self._analysed = 0
        self._last_ref_frame = -(10 ** 9)
        # Roadmap task 5: the mask can say how far this frame's input stood above the residual
        # it predicted. Evidence for the joint double-talk gate: a frame that leaves >> the
        # predicted residual holds a second source, and only this class knows its own
        # prediction (EchoPathBaseline's floor would poison the same measurement elsewhere).
        self.residual_margin_db = float("-inf")

    def reset(self) -> None:
        super().reset()
        self._analysed = 0
        self._last_ref_frame = -(10 ** 9)
        self.residual_margin_db = float("-inf")

    def set_running_erle(self, erle_db: float) -> None:
        """The canceller's most recent ERLE (dB): low values deepen the suppression."""
        self._erle_db = 0.9 * self._erle_db + 0.1 * float(erle_db)

    def _note_ref_activity(self, ref_total: float) -> None:
        self._analysed += 1
        if ref_total > self.ref_floor:
            self._last_ref_frame = self._analysed

    def _near_end_share(self, out_b: np.ndarray, echo_b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(share of the frame the margin rule cannot explain, the median-rule bands, the
        high-quantile-rule bands). ``out_b`` is this frame's input band powers."""
        echo_hi = np.maximum(self.coupling_hi * self._held_now, echo_b)
        user_hi = out_b >= self.protect * echo_hi
        user_med = out_b >= self.protect * echo_b
        out_total = float(out_b.sum())
        share = float(out_b[user_med].sum()) / out_total if out_total > 0.0 else 0.0
        return share, user_med, user_hi

    def _update_ratio_gate(self, out_b: np.ndarray, echo_b: np.ndarray) -> None:
        # Roadmap task 5: how far this frame's input stands above the residual this mask
        # predicted, per the mask's OWN prediction (EchoPathBaseline's -120 dB floor would
        # poison the same measurement elsewhere). Evidence for the joint double-talk gate.
        out_total_margin = float(out_b.sum())
        echo_total_margin = float(echo_b.sum())
        # A frame the mask cannot explain at all (predicted residual ~0 but the output is loud)
        # stands a HUNDRED dB above what was predicted - the clearest second source there is, not
        # "nothing". The floor is relative to the output so the margin saturates instead of
        # dividing by zero; a genuinely silent frame (output ~0) is reported as -inf (no claim).
        if out_total_margin > 0.0:
            echo_floor = out_total_margin * 1e-6
            self.residual_margin_db = float(10.0 * np.log10(
                out_total_margin / max(echo_total_margin, echo_floor)))
        else:
            self.residual_margin_db = float("-inf")
        # One decision per frame: which bands are near-end, and whether the near end holds the
        # whole frame. A band the margin rule protects is near-end speech: its inflated
        # output-vs-reference ratio must NOT teach the echo model, or every later frame of her
        # voice looks like echo; a frame that is mostly that learns nothing at all; and a band
        # with nothing learned yet always bootstraps - that is how the first frames of an echo
        # onset teach their own ratios.
        share, _user_med, user_hi = self._near_end_share(out_b, echo_b)
        self._frame_user = user_hi
        if share > self.freeze_frac:
            self._ratio_skip = self.coupling_hi > 0.0
        else:
            self._ratio_skip = user_hi & (self.coupling_hi > 0.0)

    def _band_gain(self, out_b: np.ndarray, echo_b: np.ndarray) -> np.ndarray:
        if (self._analysed - self._last_ref_frame) * (self.hop / self.rate) > self._tail_s:
            return np.ones_like(out_b)
        extra = self.over_ramp * max(0.0, self.erle_target_db - self._erle_db)
        gain = max(self.over * 10.0 ** (-extra / 20.0), 1e-4)
        return np.where(self._frame_user, 1.0, gain)


def user_present(mask: ReferenceMask, threshold_db: float = 3.0) -> bool:
    return mask.user_energy_db() > threshold_db
