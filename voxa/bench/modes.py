"""Hearing 2 (task h01): every serious echo-cancellation candidate behind one bench interface.

The bench already runs four echo modes (``voxa``, ``aec3``, ``webrtc``, ``off``). This module adds
the candidates Hearing 2 measures, each behind the SAME canceller contract the existing modes use
(:meth:`process(mic, ref)` on float -1..1 samples, ``continuous``, ``delay_managed_internally``,
``filter_span_ms``, ``close()``), plus an availability probe that never raises and always says WHY
a candidate cannot run here.

A candidate that cannot run is never faked: the runner keeps the mode name, records the reason in
the report and measures nothing (the same rule :func:`voxa.anc.aec3_available` established for
``aec3`` and :func:`voxa.audio.webrtc_available` established for ``webrtc``).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from ..anc import aec3_available
from ..dtln import DtlnCanceller, dtln_available as _dtln_stage_available

SR = 16000

# The modes Hearing 2 compares. The first four are the existing bench arms.
LINEAR_MODES = ("voxa", "aec3", "aec3full", "speex")
MODES = ("voxa", "webrtc", "aec3", "off")
# New arms this task adds; the old arms keep their existing code paths.
HEARING2_MODES = ("aec3full", "speex", "dtln", "nkf", "hybrid")


# ---------------------------------------------------------------------------------------------
# availability probes: (ok, reason) - never raise, and the reason is what the report prints.
# ---------------------------------------------------------------------------------------------

def _soname_candidates() -> tuple[str, ...]:
    return ("libspeexdsp.so.1", "libspeexdsp.so")


def speex_library() -> ctypes.CDLL | None:
    """The system SpeexDSP (the MDF echo canceller and its residual suppressor), or None."""
    for name in _soname_candidates():
        try:
            return _bind(ctypes.CDLL(name))
        except OSError:
            continue
    return None


def speex_available() -> tuple[bool, str]:
    lib = speex_library()
    if lib is None:
        return False, ("libspeexdsp not importable by ctypes ("
                      + ", ".join(_soname_candidates()) + " absent): no MDF canceller here")
    for symbol in ("speex_echo_state_init", "speex_echo_cancellation",
                       "speex_echo_ctl", "speex_echo_state_destroy"):
        if not hasattr(lib, symbol):
            return False, f"libspeexdsp has no {symbol}"
    return True, "libspeexdsp via ctypes"


def aec3full_shim_path() -> Path | None:
    """The compiled AEC3-full shim, if this machine has one (env override, then the bench's
    third_party build tree - never in /tmp, never committed)."""
    env = os.environ.get("VOXA_AEC3FULL_SHIM")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    root = Path(os.environ.get("VOXA_BENCH_THIRDPARTY", "/home/rhy/voxa-bench-data/third_party"))
    for name in ("libvoxa_aec3full.so", "voxa_aec3full.so"):
        p = root / "webrtc-audio-processing" / name
        if p.is_file():
            return p
    return None


def aec3full_available() -> tuple[bool, str]:
    """WebRTC AEC3 with its FULL config needs a shim compiled against the freedesktop
    ``webrtc-audio-processing`` headers. Without those headers (and with no source to build it
    from) the EchoCanceller3Config stays unreachable: the wheel that IS installed exposes no
    config, which is exactly why Hearing 2 asked for the shim."""
    shim = aec3full_shim_path()
    if shim is None:
        headers = sorted(Path("/usr/include").glob("webrtc-audio-processing*/**/audio_processing.h"))
        return False, ("no AEC3-full shim built (set VOXA_AEC3FULL_SHIM); freedesktop "
                       "webrtc-audio-processing headers not installed"
                       + ("" if not headers else f" but found {headers[0]}"))
    try:
        ctypes.CDLL(str(shim))
    except OSError as exc:
        return False, f"AEC3-full shim {shim} does not load: {exc}"
    return True, f"AEC3-full shim {shim}"


def third_party_dir() -> Path:
    return Path(os.environ.get("VOXA_BENCH_THIRDPARTY", "/home/rhy/voxa-bench-data/third_party"))


def dtln_available() -> tuple[bool, str]:
    """DTLN-aec (neural, 16 kHz): the shipping live echo stage in :mod:`voxa.dtln`, run here through
    the SAME code the app uses (ONNX + onnxruntime), so the bench number is the app number."""
    return _dtln_stage_available()


def _dtln_model_size() -> str:
    from ..dtln import _model_size
    return _model_size()


def nkf_available() -> tuple[bool, str]:
    """NKF-AEC (open neural Kalman filter, small). Code + pretrained weights fetched from
    github.com/fjiang9/NKF-AEC (the official repo: a PyTorch streaming-capable reference), run on
    this CPU with torch."""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"torch not importable: {exc}"
    root = third_party_dir() / "NKF-AEC-fjiang9" / "src"
    ckpt = root / "nkf_epoch70.pt"
    if not ckpt.is_file():
        return False, (f"no NKF-AEC weights at {ckpt}; upstream (fjiang9/NKF-AEC) publishes the "
                       "PyTorch reference + nkf_epoch70.pt (LebaneseNanoElectronics/NKF-AEC is 404)")
    return True, f"NKF-AEC weights {ckpt} via torch"


def availability() -> dict[str, dict[str, object]]:
    """Every candidate at once: what runs here and exactly why the rest does not."""
    out: dict[str, dict[str, object]] = {}
    for name, probe in (("aec3full", aec3full_available), ("speex", speex_available),
                        ("dtln", dtln_available), ("nkf", nkf_available)):
        ok, reason = probe()
        out[name] = {"available": bool(ok), "reason": reason}
    out["hybrid"] = {"available": True,
                     "reason": ("linear aec3 (installed wheel) + Voxa's residual suppressor"
                                if aec3_available() else
                                "linear Voxa NLMS + Voxa's residual suppressor")}
    return out


# ---------------------------------------------------------------------------------------------
# SpeexDSP: the MDF (multi-delay) linear canceller + its own residual suppressor, over ctypes.
# ---------------------------------------------------------------------------------------------

# speex_echo_ctl() requests, as they exist in the libspeexdsp actually installed here (1.2.1).
# These are the ones verified against the live library (an unsupported request returns -1 and the
# library warns; the request ids are NEVER guessed at - a wrong one takes a struct pointer, and
# request 29, which the upstream header calls TAU, corrupts this build's state outright: the
# process dies in speex_echo_state_destroy. Measured here, 2026-11-21, and not used.)
_SPEEX_CTL = {
    "max_filter_length": (25, "int"),
    "alpha": (27, "float"),        # the adaptation forgetting factor: the double-talk knob
}
_SPEEX_UNSUPPORTED = ("delay", "double_talk", "use_soft_gain", "residual_leak", "pre_gain", "tau")


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Prototypes BEFORE the first call: without a c_void_p restype ctypes truncates the state
    pointer to an int and the library dies on the next call (measured here, 2026-11-21)."""
    lib.speex_echo_state_init.restype = ctypes.c_void_p
    lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.speex_echo_ctl.restype = ctypes.c_int
    lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.speex_echo_cancellation.restype = None
    lib.speex_echo_cancellation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_void_p]
    lib.speex_echo_state_reset.restype = None
    lib.speex_echo_state_reset.argtypes = [ctypes.c_void_p]
    lib.speex_echo_state_destroy.restype = None
    lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
    return lib


class SpeexCanceller:
    """SpeexDSP's MDF echo canceller behind Voxa's canceller contract.

    ``speex_echo_state_init(frame_size, filter_length)`` runs the multi-delay-FD implementation;
    its built-in soft-gain and residual decorrelator ARE the residual suppressor (SpeexDSP does
    not ship a separable one), so unlike AEC3 the linear stage and the suppressor cannot be
    configured apart. The settings this mode exposes are the delay (fed from Voxa's known render
    timeline, which is the whole point of the bench) and the double-talk behaviour.
    """

    BLOCK = 160                      # a Speex frame: 10 ms at 16 kHz
    continuous = True                # the filter and its soft gain keep state across frames
    delay_managed_internally = False  # the MDF searches the delay itself, inside its filter span
    delay_ctl_supported = False      # measured here: this build exposes no DELAY request
    filter_span_ms: float = 0.0      # set from the taps below

    def __init__(self, sample_rate: int = SR, taps: int = 2048, *, delay: int = 0,
                 alpha: float = 0.3) -> None:
        if int(sample_rate) != SR:
            raise ValueError(f"the bench SpeexDSP arm runs at {SR} Hz, not {sample_rate}")
        self._lib = speex_library()
        if self._lib is None:
            raise RuntimeError("libspeexdsp unavailable: " + speex_available()[1])
        frame = self.BLOCK
        taps = max(int(taps), frame)
        self.filter_span_ms = taps / SR * 1000.0
        self._st = self._lib.speex_echo_state_init(ctypes.c_int(frame), ctypes.c_int(taps))
        if not self._st:
            raise MemoryError("speex_echo_state_init failed")
        self._frame = frame
        self._taps = taps
        self._delay = int(delay)
        self.settings: dict[str, object] = {"taps": taps, "delay": self._delay,
                                            "alpha": float(alpha)}
        self._applied: dict[str, bool] = {}
        self._ctl("max_filter_length", taps)
        self._ctl("alpha", alpha)
        self.settings["settings_reachable"] = dict(self._applied)
        self.settings["settings_unreachable"] = list(_SPEEX_UNSUPPORTED)
        self._mic_rem = np.zeros(0, dtype=np.float32)
        self._ref_rem = np.zeros(0, dtype=np.float32)
        self._converged = 0.0

    def _ctl(self, name: str, value) -> bool:
        """Apply one library setting; False when this SpeexDSP build does not know it."""
        code, kind = _SPEEX_CTL[name]
        buf = (ctypes.c_float * 2)(0.0, 0.0) if kind == "float" else (ctypes.c_int * 2)(0, 0)
        if kind == "float":
            buf[0] = float(value)
        else:
            buf[0] = int(value)
        rc = self._lib.speex_echo_ctl(self._st, ctypes.c_int(code), ctypes.cast(buf, ctypes.c_void_p))
        supported = rc is not None and int(rc) >= 0
        self._applied[name] = bool(supported)
        return bool(supported)

    def set_delay(self, delay_samples: int) -> None:
        """Hand the canceller the delay Voxa KNOWS (her render timeline, f128) instead of letting
        it search. The reference it is fed is the raw render window, so the shift happens here."""
        self._delay = max(0, int(delay_samples))
        self.settings["delay"] = self._delay

    def process(self, mic, ref) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        if self._delay > 0 and ref.size:
            ref = np.concatenate([np.zeros(self._delay, dtype=np.float32),
                                  ref[:-self._delay]])
        length = min(mic.size, ref.size)
        if length == 0:
            return np.zeros(0, dtype=np.float32)
        self._mic_rem = np.concatenate([self._mic_rem, mic[:length]])
        self._ref_rem = np.concatenate([self._ref_rem, ref[:length]])
        out = []
        i = 0
        while i + self._frame <= self._mic_rem.size:
            out.append(self._frame_clean(self._mic_rem[i:i + self._frame],
                                         self._ref_rem[i:i + self._frame]))
            i += self._frame
        self._mic_rem = self._mic_rem[i:]
        self._ref_rem = self._ref_rem[i:]
        if self._mic_rem.size:
            out.append(self._mic_rem)
            self._mic_rem = self._mic_rem[0:0]
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def _frame_clean(self, mic_b: np.ndarray, ref_b: np.ndarray) -> np.ndarray:
        """One 10 ms frame through speex_echo_cancellation (16-bit in, 16-bit out)."""
        n = self._frame
        rec = (np.clip(np.rint(mic_b * 32767.0), -32768, 32767)).astype("<i2")
        play = (np.clip(np.rint(ref_b * 32767.0), -32768, 32767)).astype("<i2")
        out = np.zeros(n, dtype="<i2")
        self._lib.speex_echo_cancellation(
            ctypes.c_void_p(self._st),
            ctypes.cast(rec.ctypes.data, ctypes.c_void_p),
            ctypes.cast(play.ctypes.data, ctypes.c_void_p),
            ctypes.cast(out.ctypes.data, ctypes.c_void_p))
        cleaned = out.astype(np.float32) / 32767.0
        peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
        if peak > 1.0:
            cleaned = cleaned / peak
        self._last_mic_power = float(np.mean(mic_b.astype(np.float64) ** 2))
        self._last_residual_power = float(np.mean(cleaned.astype(np.float64) ** 2))
        self._converged = 10.0 * np.log10(
            max(self._last_mic_power, 1e-20) / max(self._last_residual_power, 1e-20))
        return cleaned

    def reset(self) -> None:
        self._lib.speex_echo_state_reset(ctypes.c_void_p(self._st))

    def note_double_talk(self, hold_ms: float) -> None:
        """Voxa's joint double-talk gate holding the floor: SpeexDSP has no external double-talk
        control on this build (SPEECH_ECHO_DOUBLE_TALK is not reachable), so the call is recorded
        and never silently dropped."""
        self._double_talk_notes = getattr(self, "_double_talk_notes", 0) + 1
        self.settings["double_talk_notes"] = self._double_talk_notes

    def close(self) -> None:
        if getattr(self, "_st", None):
            self._lib.speex_echo_state_destroy(ctypes.c_void_p(self._st))
            self._st = None

    @property
    def mic_power_db(self) -> float:
        """Energy the microphone delivered on the last frame, in dB."""
        return 10.0 * float(np.log10(self._last_mic_power)) if self._last_mic_power > 1e-20 else -120.0

    @property
    def residual_power_db(self) -> float:
        """Energy left after the MDF filter on the last frame, in dB."""
        if self._last_residual_power <= 1e-20:
            return -120.0
        return 10.0 * float(np.log10(self._last_residual_power))

    @property
    def echo_explained_db(self) -> float:
        """How much of the microphone the delayed reference explains, in dB (0.0 = all of it)."""
        if self._last_mic_power <= 1e-20:
            return -120.0
        explained = max(self._last_mic_power - self._last_residual_power, 0.0)
        if explained <= 1e-20:
            return -120.0
        return 10.0 * float(np.log10(explained / self._last_mic_power))

    @property
    def delay_samples(self) -> int:
        return int(self._delay)

    @property
    def converged_db(self) -> float:
        return float(self._converged)


# DTLN-aec (breizhn, MIT) is now the live echo stage: the bench arm builds the SAME canceller the
# app runs (voxa.dtln.DtlnCanceller, ONNX + onnxruntime), so the bench number IS the app number.
# h03 measured it through LiteRT/.tflite; h04 ships it as voxa/data/dtln/*.onnx. See H04-DTLN.md.
# ---------------------------------------------------------------------------------------------
# NKF-AEC (fjiang9/NKF-AEC, the official ICASSP-2023 PyTorch reference): a neural Kalman filter that
# estimates the echo path per STFT bin and subtracts it. It is a LINEAR canceller (no residual
# suppressor). The Kalman loop is frame-by-frame with persistent per-bin and GRU state, so it streams
# exactly; the net definition replicates upstream so nkf_epoch70.pt loads unchanged. STFT/ISTFT use a
# hop-256 (1024-point) overlap-add at 16 kHz.
# ---------------------------------------------------------------------------------------------

_NKF_NFFT, _NKF_HOP = 1024, 256
_NKF_NBIN = _NKF_NFFT // 2 + 1
_NKF_WIN = np.hanning(_NKF_NFFT + 1)[:-1].astype("float32")   # periodic hann, matches torch


def _nkf_classes():
    import torch
    import torch.nn as nn

    class ComplexGRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True,
                     dropout=0, bidirectional=False):
            super().__init__()
            self.gru_r = nn.GRU(input_size, hidden_size, num_layers, bias=bias,
                                batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
            self.gru_i = nn.GRU(input_size, hidden_size, num_layers, bias=bias,
                                batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

        def forward(self, x, h_rr=None, h_ir=None, h_ri=None, h_ii=None):
            Frr, h_rr = self.gru_r(x.real, h_rr)
            Fir, h_ir = self.gru_r(x.imag, h_ir)
            Fri, h_ri = self.gru_i(x.real, h_ri)
            Fii, h_ii = self.gru_i(x.imag, h_ii)
            y = torch.complex(Frr - Fii, Fri + Fir)
            return y, h_rr, h_ir, h_ri, h_ii

    class ComplexDense(nn.Module):
        def __init__(self, in_channel, out_channel, bias=True):
            super().__init__()
            self.linear_real = nn.Linear(in_channel, out_channel, bias=bias)
            self.linear_imag = nn.Linear(in_channel, out_channel, bias=bias)

        def forward(self, x):
            y_real = self.linear_real(x.real)
            y_imag = self.linear_imag(x.imag)
            return torch.complex(y_real, y_imag)

    class ComplexPReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.prelu = torch.nn.PReLU()

        def forward(self, x):
            return torch.complex(self.prelu(x.real), self.prelu(x.imag))

    class KGNet(nn.Module):
        def __init__(self, L, fc_dim, rnn_layers, rnn_dim):
            super().__init__()
            self.L = L; self.rnn_layers = rnn_layers; self.rnn_dim = rnn_dim
            self.fc_in = nn.Sequential(ComplexDense(2 * L + 1, fc_dim, bias=True), ComplexPReLU())
            self.complex_gru = ComplexGRU(fc_dim, rnn_dim, rnn_layers, bidirectional=False)
            self.fc_out = nn.Sequential(ComplexDense(rnn_dim, fc_dim, bias=True), ComplexPReLU(),
                                        ComplexDense(fc_dim, L, bias=True))

        def init_hidden(self, batch_size, device):
            self.h_rr = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
            self.h_ir = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
            self.h_ri = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
            self.h_ii = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)

        def forward(self, input_feature):
            feat = self.fc_in(input_feature).unsqueeze(1)
            rnn_out, self.h_rr, self.h_ir, self.h_ri, self.h_ii = self.complex_gru(
                feat, self.h_rr, self.h_ir, self.h_ri, self.h_ii)
            return self.fc_out(rnn_out).permute(0, 2, 1)

    class NKF(nn.Module):
        def __init__(self, L=4):
            super().__init__()
            self.L = L
            self.kg_net = KGNet(L=L, fc_dim=18, rnn_layers=1, rnn_dim=18)

    return NKF


class NkfCanceller:
    continuous = True
    delay_managed_internally = False    # the bench hands NKF the measured end-to-end delay
    delay_ctl_supported = True
    filter_span_ms = 4 * _NKF_HOP / SR * 1000.0   # L=4 frames of echo path

    def __init__(self, sample_rate: int = SR) -> None:
        if int(sample_rate) != SR:
            raise ValueError(f"the bench NKF arm runs at {SR} Hz, not {sample_rate}")
        import torch
        self._torch = torch
        NKF = _nkf_classes()
        self._dev = torch.device("cpu")
        self._net = NKF(L=4).to(self._dev).eval()
        ckpt = third_party_dir() / "NKF-AEC-fjiang9" / "src" / "nkf_epoch70.pt"
        self._net.load_state_dict(torch.load(str(ckpt), map_location="cpu"), strict=True)
        L, F = self._net.L, _NKF_NBIN
        self._h_prior = torch.zeros(F, L, 1, dtype=torch.complex64, device=self._dev)
        self._h_post = torch.zeros(F, L, 1, dtype=torch.complex64, device=self._dev)
        self._net.kg_net.init_hidden(F, self._dev)
        self._xh = np.zeros((F, L), "complex64")
        self._buf_m = np.zeros(0, "float32")
        self._buf_r = np.zeros(0, "float32")
        self._frame = 0
        self._ola = np.zeros(0, "float32")    # overlap-add output, indexed by input position
        self._oden = np.zeros(0, "float64")
        self._emitted = -_NKF_NFFT   # output lags input by one full frame (see process)
        self._delay = 0
        self._last_mic_power = 0.0
        self._last_residual_power = 0.0

    def set_delay(self, delay_samples: int) -> None:
        self._delay = max(0, int(delay_samples))

    def _run_frame(self, x_spec, y_spec):
        torch = self._torch
        self._xh = np.concatenate([self._xh[:, 1:], x_spec[:, None]], axis=1)
        xt = torch.from_numpy(self._xh).to(self._dev)
        y = torch.from_numpy(y_spec).to(self._dev)
        dh = self._h_post - self._h_prior
        e = y - (xt.unsqueeze(1) @ self._h_prior).squeeze(-1).squeeze(-1)
        feat = torch.cat([xt, e.unsqueeze(1), dh.squeeze(-1)], dim=1)
        with torch.no_grad():
            kg = self._net.kg_net(feat).squeeze(-1)
        self._h_post = self._h_prior + (kg * e.unsqueeze(1)).unsqueeze(-1)
        echo = (xt.unsqueeze(1) @ self._h_post).squeeze()
        self._h_prior = self._h_post.clone()
        self._frame += 1
        return (y - echo).cpu().numpy().reshape(-1)

    def process(self, mic, ref) -> np.ndarray:
        mic = np.asarray(mic, "float32"); ref = np.asarray(ref, "float32")
        n = min(mic.size, ref.size)
        if n == 0:
            return np.zeros(0, "float32")
        self._last_mic_power = float(np.mean(mic.astype(np.float64) ** 2))
        self._buf_m = np.concatenate([self._buf_m, mic[:n]])
        self._buf_r = np.concatenate([self._buf_r, ref[:n]])
        while self._frame * _NKF_HOP + _NKF_NFFT <= self._buf_m.size:
            s0 = self._frame * _NKF_HOP
            r0 = s0 - self._delay
            rlo = max(0, r0)

            def _spec(buf, lo, hi, lead=0):
                seg = buf[lo:hi]
                if seg.size < hi - lo:
                    seg = np.concatenate([seg, np.zeros(hi - lo - seg.size, "float32")])
                if lead > 0:
                    seg = np.concatenate([np.zeros(lead, "float32"), seg])[:hi - lo]
                return np.fft.rfft(seg * _NKF_WIN, n=_NKF_NFFT).astype("complex64")
            clean = self._run_frame(_spec(self._buf_r, rlo, rlo + _NKF_NFFT, max(0, -r0)),
                                    _spec(self._buf_m, s0, s0 + _NKF_NFFT))
            time_sig = (np.fft.irfft(clean, n=_NKF_NFFT) * _NKF_WIN).reshape(-1).astype("float32")
            need = s0 + _NKF_NFFT
            if self._ola.size < need:
                self._ola = np.concatenate([self._ola, np.zeros(need - self._ola.size, "float32")])
                self._oden = np.concatenate([self._oden, np.zeros(need - self._oden.size, "float64")])
            self._ola[s0:s0 + _NKF_NFFT] += time_sig
            self._oden[s0:s0 + _NKF_NFFT] += _NKF_WIN.astype("float64") ** 2
        # Emit the output delayed by one full frame (D = NFFT) so every returned sample is fully
        # covered by the hop-256 / 1024-point periodic-Hann overlap-add. A sample p is only
        # reconstructed once EVERY window covering it has been added, i.e. once the last covering
        # frame (s0 = (p // HOP) * HOP) has run: p covered iff p < self._frame * HOP. The bench
        # hands samples in 320-sample chunks but the STFT frame is 1024, so a chunk's newest
        # samples are never yet covered when it must be returned. Running the output one full
        # frame late makes "covered" and "returned" coincide (the same math as offline, delayed
        # 64 ms; the first NFFT output samples are silence). Emitting them early shatters the
        # output (ERLE 26 dB, recall collapse seen in the smoke before this fix).
        col = 1.5   # COLA constant of the hop-256 / 1024-point periodic-hann overlap-add
        covered = self._frame * _NKF_HOP
        out = np.zeros(n, "float32")
        lo = max(self._emitted, 0)
        hi = min(self._emitted + n, covered)
        if hi > lo:
            if self._ola.size < hi:
                grow = hi - self._ola.size
                self._ola = np.concatenate([self._ola, np.zeros(grow, "float32")])
                self._oden = np.concatenate([self._oden, np.zeros(grow, "float64")])
            out[lo - self._emitted:hi - self._emitted] = (
                self._ola[lo:hi] / np.maximum(self._oden[lo:hi], col))
        self._emitted += n
        self._last_residual_power = float(np.mean(out.astype(np.float64) ** 2)) if out.size else self._last_residual_power
        return out.astype("float32")

    def close(self) -> None:
        pass

    def note_double_talk(self, hold_ms: float) -> None:
        """NKF has no external double-talk control; the call is recorded, never silently dropped."""
        self._double_talk_notes = getattr(self, "_double_talk_notes", 0) + 1

    @property
    def mic_power_db(self) -> float:
        return 10.0 * float(np.log10(self._last_mic_power)) if self._last_mic_power > 1e-20 else -120.0

    @property
    def residual_power_db(self) -> float:
        return 10.0 * float(np.log10(self._last_residual_power)) if self._last_residual_power > 1e-20 else -120.0

    @property
    def echo_explained_db(self) -> float:
        if self._last_mic_power <= 1e-20:
            return -120.0
        ex = max(self._last_mic_power - self._last_residual_power, 0.0)
        return 10.0 * float(np.log10(ex / self._last_mic_power)) if ex > 1e-20 else -120.0

    @property
    def delay_samples(self) -> int:
        return int(self._delay)

    @property
    def converged_db(self) -> float:
        return 0.0


def make_linear(mode: str, *, delay: int = 0, sample_rate: int = SR):
    """The linear stage of a bench echo mode, or None when the mode asked for no linear stage."""
    if mode == "speex":
        return SpeexCanceller(sample_rate, delay=delay)
    if mode == "aec3full":
        raise RuntimeError("aec3full has no shim: " + aec3full_available()[1])
    if mode == "dtln":
        ok, why = dtln_available()
        if not ok:
            raise RuntimeError("dtln unavailable: " + why)
        return DtlnCanceller(sample_rate)
    if mode == "nkf":
        ok, why = nkf_available()
        if not ok:
            raise RuntimeError("nkf unavailable: " + why)
        return NkfCanceller(sample_rate)
    return None
