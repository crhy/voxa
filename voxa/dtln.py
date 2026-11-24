"""DTLN-aec as Voxa's live echo stage (Hearing 2 h04, issue #72).

DTLN-aec is the dual-signal-transform LSTM network (breizhn/DTLN-aec, MIT) that h03 measured as the
strongest echo canceller on Voxa's bench: ERLE 24.41 dB for 7.86 dB of user attenuation, beating
the incumbent on every axis (docs/hearing/frontier.md). h03 measured it offline through the repo's
own LiteRT/``.tflite`` runtime; this module is the shipping stage - the same weights exported to
ONNX (``voxa/data/dtln/``, see that folder's SOURCES.md) run with ``onnxruntime``, which the Flatpak
already carries.

It is ONE canceller the live app and the bench both drive (``--echo-mode dtln`` in
``voxa/bench`` builds exactly this class), so the numbers the frontier quotes are the numbers the
app produces. The streaming contract matches the DTLN-aec repo's ``run_aec.py`` exactly (16 kHz,
block 512 / shift 128, stage 1 = frequency mask, stage 2 = time-domain enhancement); the LSTM state
is kept across ``process`` calls, so it runs frame by frame on a live stream and never rebuilds.
Like AEC3 it learns the echo delay itself (``delay_managed_internally``), so Voxa hands it the raw
aligned far-end reference - the exact samples she played - and does not push a delay into it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SR = 16000
BLOCK_LEN, BLOCK_SHIFT = 512, 128
_PAD = BLOCK_LEN - BLOCK_SHIFT  # 384 leading zeros run_aec pads the stream with


def _model_size() -> str:
    return os.environ.get("VOXA_DTLN_MODEL", "512")


def data_dir() -> Path:
    """The shipped ONNX folder (override with VOXA_DTLN_DIR for tests/benchmarks)."""
    env = os.environ.get("VOXA_DTLN_DIR")
    return Path(env) if env else Path(__file__).parent / "data" / "dtln"


def model_paths(model: str | None = None) -> tuple[Path, Path]:
    model = model or _model_size()
    root = data_dir()
    return (root / f"dtln_aec_{model}_1.onnx", root / f"dtln_aec_{model}_2.onnx")


def dtln_available() -> tuple[bool, str]:
    """(ok, reason): can this machine run the DTLN stage - onnxruntime importable and the weights
    present? Never raises; the reason is what the live app and the bench log when it falls back."""
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - an import failure is simply "not available here"
        return False, f"onnxruntime not importable: {exc}"
    paths = model_paths()
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        return False, (f"DTLN-aec ONNX weights missing under {data_dir()} "
                       f"({', '.join(missing)}); see voxa/data/dtln/SOURCES.md")
    return True, f"DTLN-aec {_model_size()} ONNX via onnxruntime"


class DtlnCanceller:
    """DTLN-aec behind Voxa's canceller contract: ``process(mic, ref)`` on float -1..1 samples,
    frame by frame, keeping the LSTM state. ``continuous`` and ``delay_managed_internally`` match
    AEC3 (it is a full canceller, not a linear stage: it needs every frame, silence and echo tail
    included, and models the delay itself)."""

    continuous = True
    delay_managed_internally = True
    filter_span_ms = BLOCK_LEN / SR * 1000.0  # the 32 ms analysis window it sees

    def __init__(self, sample_rate: int = SR, model: str | None = None) -> None:
        if int(sample_rate) != SR:
            raise ValueError(f"the DTLN echo stage runs at {SR} Hz, not {sample_rate}")
        import onnxruntime as ort
        s1_path, s2_path = model_paths(model)
        opts = ort.SessionOptions()
        # A block is tiny; more threads only add scheduling overhead (measured here: p99 per block
        # 1.01 ms on one thread vs 0.44 ms on two, both far under the 8 ms frame shift). One thread
        # keeps the per-block latency the live path needs and the cost deterministic.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._s1 = ort.InferenceSession(str(s1_path), opts, providers=["CPUExecutionProvider"])
        self._s2 = ort.InferenceSession(str(s2_path), opts, providers=["CPUExecutionProvider"])
        self._s1_names = [d.name for d in self._s1.get_inputs()]
        self._s2_names = [d.name for d in self._s2.get_inputs()]
        self._s1_out = [d.name for d in self._s1.get_outputs()]
        self._s2_out = [d.name for d in self._s2.get_outputs()]
        # The two LSTM states, shape [1, 2, 512, 2] (the model's own state layout), kept warm.
        self._st1 = np.zeros((1, 2, 512, 2), "float32")
        self._st2 = np.zeros((1, 2, 512, 2), "float32")
        # run_aec sliding-window buffers.
        self._inb = np.zeros(BLOCK_LEN, "float32")
        self._inb_l = np.zeros(BLOCK_LEN, "float32")
        self._outb = np.zeros(BLOCK_LEN, "float32")
        self._fifo_m = np.zeros(0, "float32")
        self._fifo_r = np.zeros(0, "float32")
        self._consumed = 0
        self._out_all = np.zeros(0, "float32")
        self._returned = 0
        self._last_mic_power = 0.0
        self._last_residual_power = 0.0
        self._double_talk_notes = 0

    def _feed(self, session, names, out_names, *vals):
        return session.run(out_names, dict(zip(names, vals)))

    def process(self, mic, ref) -> np.ndarray:
        """Clean ``mic`` against ``ref`` (the exact far-end samples Voxa played over the same span).
        Returns exactly len(mic) samples, aligned to the microphone but delayed by one block shift
        so every returned sample is fully reconstructed by stage 2 (output at k needs input k+128)."""
        mic = np.asarray(mic, "float32")
        ref = np.asarray(ref, "float32")
        n = min(mic.size, ref.size)
        if n == 0:
            return np.zeros(0, "float32")
        self._last_mic_power = float(np.mean(mic.astype(np.float64) ** 2))
        self._fifo_m = np.concatenate([self._fifo_m, mic[:n]])
        self._fifo_r = np.concatenate([self._fifo_r, ref[:n]])
        # run_aec sliding windows over the padded stream: PAD leading zeros before the real samples.
        while self._consumed + BLOCK_SHIFT <= self._fifo_m.size + _PAD:
            lo = self._consumed - _PAD
            hi = lo + BLOCK_SHIFT
            left = max(0, -lo)

            def _win(buf):
                seg = buf[max(lo, 0):hi] if hi > 0 else np.zeros(0, "float32")
                return np.concatenate([np.zeros(left, "float32"), seg])[:BLOCK_SHIFT]

            self._inb[:-BLOCK_SHIFT] = self._inb[BLOCK_SHIFT:]
            self._inb[-BLOCK_SHIFT:] = _win(self._fifo_m)
            self._inb_l[:-BLOCK_SHIFT] = self._inb_l[BLOCK_SHIFT:]
            self._inb_l[-BLOCK_SHIFT:] = _win(self._fifo_r)
            self._consumed += BLOCK_SHIFT
            fft = np.fft.rfft(self._inb)
            mag = np.abs(fft).reshape(1, 1, -1).astype("float32")
            lmag = np.abs(np.fft.rfft(self._inb_l)).reshape(1, 1, -1).astype("float32")
            o1 = self._feed(self._s1, self._s1_names, self._s1_out, mag, self._st1, lmag)
            mask = o1[0]
            self._st1 = o1[1]
            est = np.fft.irfft(fft * mask).reshape(1, 1, -1).astype("float32")
            o2 = self._feed(self._s2, self._s2_names, self._s2_out,
                            est, self._st2, self._inb_l.reshape(1, 1, -1).astype("float32"))
            ob = o2[0]
            self._st2 = o2[1]
            self._outb[:-BLOCK_SHIFT] = self._outb[BLOCK_SHIFT:]
            self._outb[-BLOCK_SHIFT:] = 0.0
            self._outb += np.squeeze(ob)
            if self._consumed - _PAD - BLOCK_SHIFT >= 0:
                self._out_all = np.concatenate([self._out_all, self._outb[:BLOCK_SHIFT]])
        # Return exactly len(mic), aligned to the mic but one block shift late, so every returned
        # sample is fully reconstructed (stage-2 output at position k needs input at k+128).
        want = self._returned + n
        start = want - n - BLOCK_SHIFT
        need = want - BLOCK_SHIFT
        out = np.zeros(n, "float32")
        lo = max(start, 0)
        hi = min(need, self._out_all.size)
        if hi > lo:
            out[(lo - start):(hi - start)] = self._out_all[lo:hi]
        self._returned = want
        self._last_residual_power = (float(np.mean(out.astype(np.float64) ** 2))
                                     if out.size else self._last_residual_power)
        return out

    def close(self) -> None:
        # onnxruntime sessions release with the object; nothing external to tear down.
        pass

    def note_double_talk(self, hold_ms: float) -> None:
        """DTLN has no external double-talk control; the call is recorded, never silently dropped."""
        self._double_talk_notes += 1

    @property
    def mic_power_db(self) -> float:
        return 10.0 * float(np.log10(self._last_mic_power)) if self._last_mic_power > 1e-20 else -120.0

    @property
    def residual_power_db(self) -> float:
        return (10.0 * float(np.log10(self._last_residual_power))
                if self._last_residual_power > 1e-20 else -120.0)

    @property
    def echo_explained_db(self) -> float:
        if self._last_mic_power <= 1e-20:
            return -120.0
        ex = max(self._last_mic_power - self._last_residual_power, 0.0)
        return 10.0 * float(np.log10(ex / self._last_mic_power)) if ex > 1e-20 else -120.0

    @property
    def delay_samples(self) -> int:
        # DTLN models the delay internally; it reports none, exactly like Aec3Canceller.
        return 0

    @property
    def converged_db(self) -> float:
        return 0.0
