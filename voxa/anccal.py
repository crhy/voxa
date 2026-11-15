"""Calibrate hearing: Voxa tests her own outputs until she can no longer hear herself.

Pure numpy + json. The devices (a way to play a signal and hear back what the
microphone and the output monitor captured) are injected by the caller, so this
module can be exercised without any sound hardware.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .anc import EchoCanceller, erle_db, estimate_delay

FADE_SECONDS = 0.05


def test_signals(sample_rate: int = 16000) -> list[tuple[str, np.ndarray]]:
    """Signals Voxa plays at herself: a sweep, pink noise and some chords."""
    fade = int(round(FADE_SECONDS * sample_rate))

    def fade_edges(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
        return x

    def peak(x: np.ndarray) -> np.ndarray:
        m = float(np.max(np.abs(x)))
        if m > 0:
            x = x * (0.5 / m)
        return x.astype(np.float32)

    # ("sweep", 3 s log sine sweep 80 Hz-7 kHz)
    n = int(round(3.0 * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    freq = 80.0 * (7000.0 / 80.0) ** (t / 3.0)
    sweep = np.sin(2.0 * np.pi * freq * t)
    sweep = peak(fade_edges(sweep))

    # ("noise", 3 s pink noise): spectrum weighted by 1/sqrt(frequency)
    rng = np.random.default_rng(0x5EED)
    size = 1
    while size < 2 * n:
        size <<= 1
    spec = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    idx = np.arange(size, dtype=np.float64)
    weight = 1.0 / np.sqrt(np.maximum(idx, 1.0))
    spec = spec * weight
    noise = np.fft.irfft(np.conj(spec), n)[:n]
    noise = peak(fade_edges(noise))

    # ("tones", 2 s of chords): two sustained chords
    m = int(round(2.0 * sample_rate))
    tm = np.arange(m, dtype=np.float64) / sample_rate
    chords = [(220.0, 277.2, 330.0), (261.6, 330.0, 392.5)]
    half = m // 2
    tones = np.zeros(m, dtype=np.float64)
    for ci, chord in enumerate(chords):
        lo = 0 if ci == 0 else half
        hi = half if ci == 0 else m
        seg = tm[lo:hi] - tm[lo]
        acc = np.zeros(hi - lo, dtype=np.float64)
        for pitch in chord:
            acc += np.sin(2.0 * np.pi * pitch * seg)
        acc /= len(chord)
        tones[lo:hi] = acc
    tones = peak(fade_edges(tones))

    return [("sweep", sweep), ("noise", noise), ("tones", tones)]


def _safe(name: str) -> str:
    """Filesystem-safe form of a device name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "x"


def profile_path(sink: str, source: str) -> Path:
    """Where a saved hearing profile for this sink/source pair lives."""
    base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local/share"))
    return base / "voxa" / "hearing" / f"{_safe(sink)}__{_safe(source)}.json"


# A profile is reused while the volume stays within these bounds of the level it was learned at
# (one stop either way is 1.41; two stops is the safe outer edge).
LEVEL_RATIO_LOW = 0.5
LEVEL_RATIO_HIGH = 2.0


@dataclass
class CalibrationResult:
    delay: int
    erle_db: float
    passes: int
    per_signal: dict[str, float]
    state: dict
    # Loudest microphone level seen while calibrating (0.0-1.0): the volume range the filter was learned at.
    level: float = 0.0


def calibrate(play_and_record, signals=None, target_db: float = 30.0,
              max_passes: int = 6, sample_rate: int = 16000) -> CalibrationResult:
    """Play each signal, cancel it, repeat until she can no longer hear herself.

    ``play_and_record(signal)`` returns ``(mic, ref)``: what the microphone and the
    output monitor captured while the signal played on the speakers (same length,
    float32). One persistent canceller runs across every signal and pass; the first
    pass estimates the echo delay from the sweep and rebuilds the canceller with it.
    """
    if signals is None:
        signals = test_signals(sample_rate)

    canceller = EchoCanceller(sample_rate=sample_rate)
    delay = 0
    loudest = 0.0
    best_mean = float("-inf")
    best_per: dict[str, float] = {}
    best_state = canceller.export_state()
    best_passes = 0
    prev_mean = float("-inf")

    for p in range(max_passes):
        per: dict[str, float] = {}
        for name, sig in signals:
            mic, ref = play_and_record(sig)
            if len(mic):
                loudest = max(loudest, float(np.max(np.abs(mic))))
            if p == 0 and name == "sweep" and delay == 0:
                delay = estimate_delay(mic, ref)
                if delay > 0:
                    canceller = EchoCanceller(sample_rate=sample_rate, delay=delay)
            cleaned = canceller.process(mic, ref)
            half = len(mic) // 2
            per[name] = erle_db(mic[half:], cleaned[half:])
        mean = sum(per.values()) / len(per)
        if mean > best_mean:
            best_mean = mean
            best_per = per
            best_state = canceller.export_state()
            best_passes = p + 1
        if mean >= target_db:
            break
        if p > 0 and mean - prev_mean < 0.5:
            break
        prev_mean = mean

    return CalibrationResult(delay=delay, erle_db=best_mean, passes=best_passes,
                             per_signal=best_per, state=best_state, level=loudest)


def _write_profile(sink: str, source: str, payload: dict) -> Path:
    """Write one hearing profile for a speaker/microphone pair; returns the path written."""
    path = profile_path(sink, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def save_profile(result: CalibrationResult, sink: str, source: str) -> Path:
    """Write the best filter state to disk; returns the path written."""
    return _write_profile(sink, source, {
        "delay": int(result.delay),
        "state": result.state,
        "sink": sink,
        "source": source,
        "level": float(result.level),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def save_live_state(state: dict, delay: int, sink: str, source: str,
                    level: float | None = None) -> Path:
    """Save what the running canceller learned for this pair, with no calibration run involved.

    The live listening path keeps adapting during normal use; the filter Voxa ends up with becomes
    the starting point for the next time she hears through this speaker and microphone.
    """
    return _write_profile(sink, source, {
        "delay": int(delay),
        "state": state,
        "sink": sink,
        "source": source,
        "level": float(level) if level is not None else 0.0,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def load_profile(sink: str, source: str) -> dict | None:
    """Read a saved profile, or None when there is none."""
    path = profile_path(sink, source)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def profile_is_current(profile: dict | None, sink: str | None, source: str | None,
                       level: float | None = None) -> bool:
    """True when a saved profile belongs to this speaker/microphone pair at a similar volume.

    A profile written by an older build carries no pair of its own (the file name already is the pair)
    and no level, so it is accepted for the pair it is read from. A level is only compared when both
    sides know one: outside that range the speakers or the microphone have changed enough that the
    saved filter is no longer a good starting point.
    """
    if not isinstance(profile, dict) or not sink or not source:
        return False
    saved_sink = profile.get("sink")
    if saved_sink is not None and _safe(str(saved_sink)) != _safe(sink):
        return False
    saved_source = profile.get("source")
    if saved_source is not None and _safe(str(saved_source)) != _safe(source):
        return False
    learned = float(profile.get("level") or 0.0)
    if level is None or learned <= 0.0:
        return True
    ratio = float(level) / learned
    return LEVEL_RATIO_LOW <= ratio <= LEVEL_RATIO_HIGH


def spoken_summary(result: CalibrationResult) -> str:
    """How well she can no longer hear herself, in her own words."""
    db = result.erle_db
    if db >= 30.0:
        return "I can't hear myself any more."
    if db >= 20.0:
        return "I can barely hear myself now."
    if db >= 10.0:
        return ("I still hear myself a little. Turning the speakers down or moving "
                "the microphone away from them will help.")
    return "I couldn't cancel my own sound. Check that the microphone can hear the speakers."
