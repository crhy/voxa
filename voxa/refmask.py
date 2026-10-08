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
                 floor=0.1, memory_s=0.4):
        self.rate = int(rate)
        self.frame = int(frame)
        self.hop = int(hop)
        self.nbands = int(bands)
        self.over = float(over)
        self.floor = float(floor)
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
        self.lag = 0
        self.last_user_db = -120.0
        self.last_echo_active = False
        self.mic_buf = np.zeros(0, dtype=np.float64)
        self.ref_buf = np.zeros(0, dtype=np.float64)
        self.out_num = np.zeros(0, dtype=np.float64)
        self.out_den = np.zeros(0, dtype=np.float64)
        self.frames_done = 0
        self.emitted = 0

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
                echo_b = self.coupling * ref_b
                for b in range(self.nbands):
                    if ref_b[b] > 1e-10:
                        self.ratios[b].append(mic_b[b] / ref_b[b])
                        r = sorted(self.ratios[b])
                        self.coupling[b] = r[len(r) // 2]
                gain_b = np.where(
                    mic_b >= self.over * echo_b, 1.0, self.floor
                )
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

    def echo_active(self) -> bool:
        return self.last_echo_active

    def reset(self) -> None:
        self.coupling = np.zeros(self.nbands)
        self.ratios = [deque(maxlen=self.mem_frames) for _ in range(self.nbands)]
        self.lag = 0
        self.last_user_db = -120.0
        self.last_echo_active = False
        self.mic_buf = np.zeros(0, dtype=np.float64)
        self.ref_buf = np.zeros(0, dtype=np.float64)
        self.out_num = np.zeros(0, dtype=np.float64)
        self.out_den = np.zeros(0, dtype=np.float64)
        self.frames_done = 0
        self.emitted = 0


def user_present(mask: ReferenceMask, threshold_db: float = 3.0) -> bool:
    return mask.user_energy_db() > threshold_db
