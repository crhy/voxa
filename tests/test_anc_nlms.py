import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import anc_signals  # noqa: E402

from voxa.anc import erle_db, estimate_delay, nlms_cancel

SR = 16000


def signals():
    ref, echo, speech, _ = anc_signals.make()
    delay = max(0, estimate_delay(echo, ref) - 256)
    return ref, echo, speech, delay


def test_echo_only_cancelled():
    ref, echo, speech, delay = signals()
    cleaned, _, _ = nlms_cancel(echo, ref, taps=1024, delay=delay)
    assert erle_db(echo[32000:], cleaned[32000:]) >= 35


def test_chunked_matches_one_call():
    ref, echo, speech, delay = signals()
    one, _, _ = nlms_cancel(echo, ref, taps=1024, delay=delay)
    weights = None
    history = None
    parts = []
    for start in range(0, len(echo), 160):
        part, weights, history = nlms_cancel(
            echo[start:start + 160],
            ref[start:start + 160],
            taps=1024,
            delay=delay,
            weights=weights,
            history=history,
        )
        parts.append(part)
    chunked = np.concatenate(parts)
    assert np.max(np.abs(chunked - one)) < 1e-5


def test_freeze_protects_speech():
    ref, echo, speech, delay = signals()
    mask = anc_signals.speech_mask(speech)
    mic = echo + speech
    cleaned, _, _ = nlms_cancel(mic, ref, taps=1024, delay=delay, freeze=mask)
    corr = np.corrcoef(cleaned[mask], speech[mask])[0, 1]
    assert corr >= 0.98
    tail = 2 * SR
    sel = ~mask[tail:]
    assert erle_db(mic[tail:][sel], cleaned[tail:][sel]) >= 30


def test_no_freeze_loses_protection():
    ref, echo, speech, delay = signals()
    mask = anc_signals.speech_mask(speech)
    mic = echo + speech
    cleaned, _, _ = nlms_cancel(mic, ref, taps=1024, delay=delay)
    tail = 2 * SR
    sel = ~mask[tail:]
    assert erle_db(mic[tail:][sel], cleaned[tail:][sel]) < 15


def test_silence_and_no_echo():
    ref, echo, speech, delay = signals()
    zeros = np.zeros(len(ref), dtype=np.float32)
    cleaned, _, _ = nlms_cancel(zeros, ref, taps=1024, delay=delay)
    assert np.max(np.abs(cleaned)) < 1e-6
    mask = anc_signals.speech_mask(speech)
    cleaned, _, _ = nlms_cancel(speech, ref, taps=1024, delay=delay, freeze=mask)
    mic_ms = float(np.mean(np.asarray(speech, dtype=np.float64) ** 2))
    out_ms = float(np.mean(np.asarray(cleaned, dtype=np.float64) ** 2))
    assert abs(10.0 * np.log10(out_ms / mic_ms)) <= 1.0


def test_speed():
    import time

    ref, echo, speech, delay = signals()
    start = time.time()
    cleaned, _, _ = nlms_cancel(echo, ref, taps=1024, delay=delay)
    elapsed = time.time() - start
    print(f"8 s of audio processed in {elapsed:.3f} s")
    assert elapsed < 2.0
