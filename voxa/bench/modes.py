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

SR = 16000

# The modes Hearing 2 compares. The first four are the existing bench arms.
LINEAR_MODES = ("voxa", "aec3", "aec3full", "speex")
MODES = ("voxa", "webrtc", "aec3", "off")
# New arms this task adds; the old arms keep their existing code paths.
HEARING2_MODES = ("aec3full", "speex", "dtln", "nkf", "hybrid")
# f164 (Hearing 2): every linear arm given Voxa's LOCKED render-timeline delay instead of letting it
# search the bulk delay itself. Their availability is the base arm's availability; the locked delay
# comes from ``voxa.align.ExactAligner`` (h02/f163), which the bench runner hands each arm per frame.
LOCKED_MODES = ("voxa-locked", "aec3-locked", "speex-locked")
# The full set of hearing-2 arms the CLI offers and the runner wires.
HEARING2_MODES = HEARING2_MODES + LOCKED_MODES


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
    """DTLN-aec (neural, 16 kHz stage 1 + 48 kHz stage 2). onnxruntime is installed here, so the
    only missing piece is the official pretrained model, which has to be downloaded."""
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"onnxruntime not importable: {exc}"
    root = third_party_dir() / "DTLN-aec"
    found = sorted(root.glob("**/*.onnx")) + sorted(root.glob("**/*.tflite")) if root.is_dir() else []
    if not found:
        return False, (f"onnxruntime is installed but no DTLN-aec weights under {root} "
                       "(breizhn/DTLN-aec ships .tflite only; this host has no tflite-runtime "
                       "wheel for Python 3.14 and no network to fetch one)")
    return True, f"DTLN-aec weights {found[0]}"


def nkf_available() -> tuple[bool, str]:
    """NKF-AEC (open neural Kalman filter, small). Its published code is MATLAB/Octave plus
    weights; there is no CPU inference path here to time."""
    root = third_party_dir() / "NKF-AEC"
    found = sorted(root.glob("**/*.onnx")) if root.is_dir() else []
    if not found:
        return False, (f"no NKF-AEC weights/code under {root}; upstream (LebaneseNanoElectronics/"
                       "NKF-AEC) publishes MATLAB + .mat weights and no CPU-friendly streaming "
                       "reference implementation, so real-time on this CPU cannot be measured")
    return True, f"NKF-AEC {found[0]}"


def voxa_locked_available() -> tuple[bool, str]:
    """f164: Voxa's NLMS handed her locked render-timeline delay. Always runs (pure Python), the
    same stage as ``voxa`` but with the bulk delay locked instead of searched by the filter."""
    return True, ("Voxa's NLMS + ExactAligner locked delay "
                  "(reference read at the locked offset, span covers the room tail)")


def aec3_locked_available() -> tuple[bool, str]:
    """f164: the installed AEC3 wheel fed that same locked offset (the reference re-timed before it
    reaches the wheel, whose own estimator then only refines the room tail)."""
    if not aec3_available():
        return False, ("aec3-locked needs the pywebrtc-audio AEC3 wheel "
                       "(the aec3 arm's dependency); not importable here")
    return True, ("WebRTC AEC3 wheel fed Voxa's locked render-timeline delay "
                  "(reference re-timed outside the wheel, span covers the room tail)")


def speex_locked_available() -> tuple[bool, str]:
    """f164: the SpeexDSP MDF given a reference re-timed OUTSIDE it (this build has no DELAY
    request, so the shift happens in front of the MDF rather than being handed to it)."""
    ok, why = speex_available()
    if not ok:
        return False, "speex-locked needs the libspeexdsp MDF: " + why
    return True, ("libspeexdsp MDF fed Voxa's locked render-timeline delay "
                  "(reference re-timed outside the MDF, its delay_ctl absent in this Debian build)")


def availability() -> dict[str, dict[str, object]]:
    """Every candidate at once: what runs here and exactly why the rest does not."""
    out: dict[str, dict[str, object]] = {}
    for name, probe in (("aec3full", aec3full_available), ("speex", speex_available),
                        ("dtln", dtln_available), ("nkf", nkf_available),
                        ("voxa-locked", voxa_locked_available),
                        ("aec3-locked", aec3_locked_available),
                        ("speex-locked", speex_locked_available)):
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


def make_linear(mode: str, *, delay: int = 0, sample_rate: int = SR):
    """The linear stage of a bench echo mode, or None when the mode asked for no linear stage."""
    if mode == "speex":
        return SpeexCanceller(sample_rate, delay=delay)
    if mode == "speex-locked":
        # f164: the same MDF; the runner re-times its reference with the locked delay (set_delay).
        return SpeexCanceller(sample_rate, delay=delay)
    if mode == "aec3full":
        raise RuntimeError("aec3full has no shim: " + aec3full_available()[1])
    if mode == "dtln":
        raise RuntimeError("dtln has no weights: " + dtln_available()[1])
    if mode == "nkf":
        raise RuntimeError("nkf has no CPU implementation: " + nkf_available()[1])
    return None
