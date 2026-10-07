"""Shared synthetic signals for the echo-canceller tests (not a test module itself).

The reference is broadband on purpose: a canceller can only learn the speaker-to-microphone path at frequencies
the reference actually contains, and a delay cannot be measured from a few pure tones (their correlation repeats
every period). Music and speech are broadband, so this is also the realistic case.
"""

from __future__ import annotations

import numpy as np

SR = 16000
SECONDS = 8
DELAY_SAMPLES = int(0.037 * SR)  # 592


def make(seconds: int = SECONDS, seed: int = 0):
    """Return (ref, echo, speech, delay_samples), all float32 arrays of the same length."""
    rng = np.random.default_rng(seed)
    n = seconds * SR
    t = np.arange(n) / SR
    # "Music": white noise with a gentle low-pass tilt, a slow beat, plus a few notes.
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spectrum *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    ref = np.fft.irfft(spectrum, n)
    ref *= 0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t) ** 2
    for f in (220.0, 330.0, 440.0):
        ref += 0.15 * np.sin(2 * np.pi * f * t)
    ref = (0.5 * ref / np.max(np.abs(ref))).astype(np.float32)

    # The room: 60 ms of exponentially decaying reflections after a 37 ms delay.
    room_len = int(0.06 * SR)
    room = rng.standard_normal(room_len) * np.exp(-np.arange(room_len) / (0.012 * SR))
    room /= np.sqrt(np.sum(room**2))
    delayed = np.concatenate([np.zeros(DELAY_SAMPLES), ref])[:n]
    echo = (0.6 * np.convolve(delayed, room)[:n]).astype(np.float32)

    # "Speech": three 0.6 s bursts of modulated noise, silence elsewhere.
    speech = np.zeros(n)
    for i in range(3):
        start = int((2.5 + 1.8 * i) * SR)
        length = int(0.6 * SR)
        envelope = np.abs(np.sin(np.linspace(0.0, 3.0 * np.pi, length)))
        speech[start:start + length] = 0.3 * rng.standard_normal(length) * envelope
    return ref, echo, speech.astype(np.float32), DELAY_SAMPLES


def speech_mask(speech: np.ndarray) -> np.ndarray:
    return np.abs(speech) > 0
