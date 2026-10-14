"""Streaming voice detector built on the Silero VAD model shipped with faster-whisper.

Unlike the loudness rule in :mod:`voxa.dictation` (which fires on any chunk above a
fixed level and therefore never sees a pause under background music), this looks at
the *shape* of the audio: the Silero model emits a per-window speech probability, so
music and TV talk score low while a human voice scores high.
"""

from __future__ import annotations

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
