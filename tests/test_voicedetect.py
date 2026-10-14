from __future__ import annotations

import numpy as np
import pytest

from voxa.voicedetect import VoiceDetector


def _load_detector():
    try:
        return VoiceDetector(threshold=0.5)
    except Exception:  # the onnx model or onnxruntime is unavailable
        return None


def test_silence_is_not_voice() -> None:
    det = _load_detector()
    if det is None:
        pytest.skip("silero model unavailable")
    chunk = bytes(320 * 2)  # 20 ms of zeros
    for _ in range(20):
        assert det.feed(chunk) is False


def test_streaming_matches_one_shot() -> None:
    det = _load_detector()
    if det is None:
        pytest.skip("silero model unavailable")
    rng = np.random.default_rng(7)
    n = 512 * 5
    pcm = (rng.uniform(-0.4, 0.4, n) * 32768).astype("<i2")

    for i in range(0, n, 320):
        det.feed(pcm[i : i + 320].tobytes())
    streamed = np.asarray(det.probs, dtype=np.float64)

    det.reset()
    det.feed(pcm.tobytes())  # one-shot: the whole audio in a single chunk
    one_shot = np.asarray(det.probs, dtype=np.float64)

    assert len(streamed) == len(one_shot)
    assert np.max(np.abs(streamed - one_shot)) < 1e-3
