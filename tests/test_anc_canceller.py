import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import anc_signals

from voxa.anc import EchoCanceller, erle_db, estimate_delay

SR = 16000


def feed(canceller, mic, ref, chunk):
    outs = []
    for i in range(0, len(mic), chunk):
        outs.append(canceller.process(mic[i:i + chunk], ref[i:i + chunk]))
    return np.concatenate(outs)


def test_echo_only_converges():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    feed(c, echo[:2 * SR], ref[:2 * SR], 160)
    assert c.converged_db >= 30.0


def test_double_talk():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    mic = (echo + speech).astype(np.float32)
    c = EchoCanceller(delay=d)
    cleaned = feed(c, mic, ref, 160)
    mask = anc_signals.speech_mask(speech)
    corr = float(np.corrcoef(cleaned[mask], speech[mask])[0, 1])
    assert corr >= 0.95
    p_cleaned = float(np.mean(cleaned[mask] ** 2))
    p_speech = float(np.mean(speech[mask] ** 2))
    assert abs(10.0 * np.log10(p_cleaned / p_speech)) <= 3.0
    tail = slice(2 * SR, None)
    erle = erle_db(mic[tail][~mask[tail]], cleaned[tail][~mask[tail]])
    assert erle >= 25.0


def test_chunk_invariance():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    whole = EchoCanceller(delay=d).process(echo, ref)
    by160 = feed(EchoCanceller(delay=d), echo, ref, 160)
    by137 = feed(EchoCanceller(delay=d), echo, ref, 137)
    assert np.max(np.abs(whole - by160)) < 1e-5
    assert np.max(np.abs(whole - by137)) < 1e-5


def test_state_roundtrip():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    feed(c, echo, ref, 160)
    state = json.loads(json.dumps(c.export_state()))
    c2 = EchoCanceller(delay=d)
    c2.load_state(state)
    cleaned = feed(c2, echo[:SR], ref[:SR], 160)
    assert erle_db(echo[:SR], cleaned) >= 25.0


def test_reset_and_taps():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    feed(c, echo, ref, 160)
    c.reset()
    assert c.converged_db == 0.0
    assert c._blocks_adapted == 0
    assert c.weights is None
    other = EchoCanceller(taps=512, delay=d)
    state = EchoCanceller(delay=d).export_state()
    other.load_state(state)
    assert other.weights is None


def test_speed():
    ref, echo, speech, delay = anc_signals.make()
    d = max(0, estimate_delay(echo, ref) - 256)
    c = EchoCanceller(delay=d)
    t0 = time.time()
    cleaned = feed(c, echo, ref, 160)
    elapsed = time.time() - t0
    print(f"8 s of audio in {elapsed:.2f} s")
    assert elapsed < 3.0
    assert len(cleaned) == len(echo)


def test_a_reference_with_silent_pauses_never_makes_things_worse():
    """Found on real hardware: a voice (which has pauses) as the reference and a microphone that barely hears the
    speakers made the filter blow up and ADD 30 dB of noise. It must stay harmless."""
    import numpy as np

    from voxa.anc import EchoCanceller, erle_db

    rng = np.random.default_rng(1)
    n = 8 * 16000
    ref = 0.2 * rng.standard_normal(n)
    for start in range(0, n, 16000):          # half of every second is digital near-silence
        ref[start + 8000:start + 16000] = 1e-4 * rng.standard_normal(8000)
    mic = 0.0015 * rng.standard_normal(n)     # room noise only: the microphone does not hear the speakers
    canceller = EchoCanceller()
    out = np.concatenate([canceller.process(mic[i:i + 160], ref[i:i + 160]) for i in range(0, n, 160)])
    assert erle_db(mic, out) > -1.0           # at worst a hair louder, never tens of dB
    assert float(np.max(np.abs(out))) < 0.05
