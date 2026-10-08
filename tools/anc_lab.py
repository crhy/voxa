#!/usr/bin/env python3
"""ANC lab: record real speaker-to-microphone data, measure the canceller, keep history, plot.

Usage: .venv/bin/python tools/anc_lab.py <command>
Commands: make-sources, record, evaluate, plot, all.
Data lives in .voxa-spec/out/anclab/ (git-ignored). Sound plays only via paplay,
recording only via parec; nothing here changes the default audio devices.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

from voxa.anc import EchoCanceller, erle_db, estimate_delay
from voxa.sync import band_coherence, band_coherence_segments, fit_drift_fine_info, follow_drift, predictable_share

SR = 16000
OUT = Path(".voxa-spec/out/anclab")
SRC_DIR = OUT / "sources"
REC_DIR = OUT / "recordings"
WEBRTC_DIR = OUT / "webrtc"
HISTORY = OUT / "history.jsonl"

BANDS = ((100, 300), (300, 1000), (1000, 2000), (2000, 3000), (3000, 6000))

DEFAULT_SOURCE = "@DEFAULT_SOURCE@"
VOLUMES = (0.2, 0.4, 0.7, 1.0)
LEVELS = (6.0, 0.0, -6.0)

COMMANDS = [
    "Voxa stop music", "stop dictation", "Voxa what time is it", "volume down",
    "volume up", "Voxa pause", "Voxa resume", "stop listening",
    "Voxa read a story", "Voxa what is the weather", "Voxa write a note", "Voxa stop",
]

SPEECH_LINES = [
    ("en", "The morning light came through the curtains slowly."),
    ("en-gb", "I keep a journal of every sound I hear around the house."),
    ("en-us", "Music from the speakers can bleed into the microphone if we are not careful."),
    ("en-sc", "A quiet room lets the canceller learn the shape of the echo first."),
]


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "x"


def write_wav(path: Path, samples, rate: int = SR) -> None:
    x = np.clip(np.rint(np.asarray(samples, dtype=np.float64) * 32767.0), -32768, 32767)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.astype("<i2").tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("only 16-bit mono WAVs are supported")
        frames = w.readframes(w.getnframes())
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if rate != SR:
        x = np.interp(np.arange(int(len(x) * SR / rate)), np.arange(len(x)), x).astype(np.float32)
    return x


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def dbfs(x) -> float:
    r = rms(x)
    return -200.0 if r < 1e-12 else 20.0 * float(np.log10(r))


def make_music(seed: int, tempo: float, root: float, seconds: float = 20.0) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float64) / SR
    rng = np.random.default_rng(seed)
    beat = 60.0 / tempo
    acc = np.zeros(n)
    for k, pitch in enumerate((root, root * 1.26, root * 1.5)):
        phase = k * beat / 3.0
        seg = t - phase
        strength = 0.6 / (1.0 + 0.4 * np.floor(np.maximum(seg, 0.0) / beat))
        acc += strength * np.sin(2.0 * np.pi * pitch * seg) * (seg >= 0.0)
    bass = np.sin(2.0 * np.pi * (root / 2.0) * t) * np.maximum(0.0, np.cos(np.pi * t / beat)) ** 2
    acc += 0.8 * bass
    for b in range(int(seconds / beat)):
        at = int(b * beat * SR)
        ln = int(0.09 * SR)
        if at + ln <= n:
            burst = rng.standard_normal(ln) * np.exp(-np.arange(ln) / (0.02 * SR))
            acc[at:at + ln] += 0.9 * burst * (0.5 + 0.5 * np.sin(2.0 * np.pi * root * 2.0 * t[at]))
    m = float(np.max(np.abs(acc)))
    return (acc * (0.5 / m) if m > 0 else acc).astype(np.float32)


def make_pink(seed: int, seconds: float = 10.0) -> np.ndarray:
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    size = 1
    while size < 2 * n:
        size <<= 1
    spec = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    idx = np.arange(size, dtype=np.float64)
    spec = spec / np.sqrt(np.maximum(idx, 1.0))
    x = np.fft.irfft(np.conj(spec), n)[:n]
    m = float(np.max(np.abs(x)))
    return (x * (0.5 / m) if m > 0 else x).astype(np.float32)


def make_sweep(seconds: float = 10.0) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float64) / SR
    freq = 80.0 * (7000.0 / 80.0) ** (t / seconds)
    x = np.sin(2.0 * np.pi * freq * t)
    fade = int(0.05 * SR)
    ramp = np.linspace(0.0, 1.0, fade)
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]
    return x.astype(np.float32)


def make_click_train(seconds: float = 20.0, period_ms: float = 250.0,
                     click_ms: float = 1.0) -> np.ndarray:
    """Sharp 1 ms click every 250 ms plus low pink noise, for lag-vs-time diagnosis."""
    n = int(seconds * SR)
    x = np.zeros(n, dtype=np.float64)
    click_len = max(1, int(round(click_ms * SR / 1000.0)))
    period = max(1, int(round(period_ms * SR / 1000.0)))
    for at in range(0, n, period):
        seg = min(click_len, n - at)
        x[at:at + seg] += 0.6 * np.linspace(1.0, -1.0, seg)
    pink = make_pink(11, seconds) * 0.01
    x += pink
    m = float(np.max(np.abs(x)))
    return (x * (0.6 / m) if m > 0 else x).astype(np.float32)


def find_clicks(x: np.ndarray, threshold: float = 0.3, min_gap: int = 100) -> np.ndarray:
    """Indices of sharp positive impulses above threshold, at least min_gap apart."""
    peaks: list[int] = []
    i = 0
    n = len(x)
    while i < n:
        if x[i] > threshold:
            j = i
            best = i
            while j < n and x[j] > threshold:
                if x[j] > x[best]:
                    best = j
                j += 1
            peaks.append(best)
            i = best + min_gap
        else:
            i += 1
    return np.asarray(peaks, dtype=np.int64)


def pair_clicks(mic_c: np.ndarray, ref_c: np.ndarray, tol: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Match each mic click to the nearest ref click within tol samples; return (times, lags)."""
    times: list[float] = []
    lags: list[int] = []
    for m in mic_c:
        if len(ref_c) == 0:
            break
        d = np.abs(ref_c - m)
        j = int(np.argmin(d))
        if d[j] <= tol:
            times.append(ref_c[j] / SR)
            lags.append(int(m - ref_c[j]))
    return np.asarray(times, dtype=np.float64), np.asarray(lags, dtype=np.int64)


def fit_drift(times: np.ndarray, lags: np.ndarray) -> tuple[float, float, list[int]]:
    """Least-squares slope (ppm) and offset of lag vs time, plus dropout jump indices."""
    if len(times) < 2:
        return 0.0, 0.0, []
    t = times.astype(np.float64)
    y = lags.astype(np.float64)
    slope, intercept = np.polyfit(t, y, 1)
    jumps = [int(k) for k in range(1, len(y)) if abs((y[k] - y[k - 1]) - slope * (t[k] - t[k - 1])) > 40.0]
    return float(slope), float(intercept), jumps


def sink_volume_pct(sink: str) -> float | None:
    out = subprocess.run(["pactl", "get-sink-volume", sink], capture_output=True,
                         text=True, check=False).stdout
    m = re.search(r"(\d+)%", out)
    return float(m.group(1)) if m else None


def sink_mute(sink: str) -> bool:
    out = subprocess.run(["pactl", "get-sink-mute", sink], capture_output=True,
                         text=True, check=False).stdout
    return "yes" in out.lower()


def set_sink_volume(sink: str, pct: float) -> None:
    subprocess.run(["pactl", "set-sink-volume", sink, f"{int(round(pct))}%"], check=False)


def set_sink_mute(sink: str, muted: bool) -> None:
    subprocess.run(["pactl", "set-sink-mute", sink, "1" if muted else "0"], check=False)


def espeak_wav(text: str, voice: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["espeak-ng", "-v", voice, "-w", str(path), text], check=True)


def make_sources() -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        write_wav(SRC_DIR / f"music_{i + 1:02d}.wav",
                  make_music(seed=100 + i, tempo=60.0 + 20.0 * i, root=90.0 + 25.0 * i))
    for i, (voice, line) in enumerate(SPEECH_LINES):
        tmp = SRC_DIR / f"_sp_{i}.wav"
        espeak_wav(line, voice, tmp)
        write_wav(SRC_DIR / f"speech_{i + 1:02d}.wav", read_wav(tmp))
        tmp.unlink()
    write_wav(SRC_DIR / "pink.wav", make_pink(7))
    write_wav(SRC_DIR / "sweep.wav", make_sweep())
    for i, cmd in enumerate(COMMANDS):
        tmp = SRC_DIR / f"_cmd_{i}.wav"
        espeak_wav(cmd, "en-gb", tmp)
        write_wav(SRC_DIR / f"cmd_{i + 1:02d}.wav", read_wav(tmp))
        tmp.unlink()


def source_files() -> list[Path]:
    return sorted(p for p in SRC_DIR.glob("*.wav") if not p.name.startswith("_"))


def align_ends(*arrays: np.ndarray) -> list[np.ndarray]:
    """Trim the front of every array so all end together and have equal length."""
    n = min(len(a) for a in arrays)
    return [a[len(a) - n:] for a in arrays]


def shift_forward(x, k: int) -> np.ndarray:
    """Shift x later by k samples (insert k zeros at the front), same length."""
    x = np.asarray(x)
    if k == 0:
        return x.copy()
    if k > 0:
        return np.concatenate([np.zeros(k, dtype=x.dtype), x])[:len(x)]
    return np.concatenate([x[-k:], np.zeros(-k, dtype=x.dtype)])[:len(x)]


def correlation_peak(mic, ref, max_delay: int = 8000) -> tuple[int, float]:
    """Best echo delay in samples and how far the peak stands above correlation noise."""
    n = min(len(mic), len(ref), 5 * SR)
    if n <= 0:
        return 0, 0.0
    mic_n = np.asarray(mic[:n], dtype=np.float64)
    ref_n = np.asarray(ref[:n], dtype=np.float64)
    if float(np.mean(mic_n**2)) < 1e-8 or float(np.mean(ref_n**2)) < 1e-8:
        return 0, 0.0
    size = 1
    while size < 2 * n:
        size <<= 1
    corr = np.fft.irfft(np.fft.rfft(mic_n, size) * np.conj(np.fft.rfft(ref_n, size)))
    window = np.abs(corr[:max_delay + 1])
    med = float(np.median(window))
    peak = int(np.argmax(window))
    ratio = window[peak] / med if med > 0 else float("inf")
    return peak, ratio


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def play_and_record(wav_path: Path | None, volume: float, monitor: str, wake: Path | None = None,
                    latency: int = 20, fix_rate: bool = False, hold: float = 10.0):
    wake_p = None
    if wake is not None:
        wake_p = subprocess.Popen(["paplay", "--volume", "1", str(wake)])
        time.sleep(0.3)
    args = ["parec", "--rate", str(SR), "--channels", "1", "--format", "s16le", "--raw",
            f"--latency-msec={latency}"]
    if fix_rate:
        args.append("--fix-rate")
    mic_raw = REC_DIR / "_mic.raw"
    ref_raw = REC_DIR / "_ref.raw"
    with open(mic_raw, "wb") as mf, open(ref_raw, "wb") as rf:
        mic_p = subprocess.Popen(args + ["--device", DEFAULT_SOURCE], stdout=mf, stderr=subprocess.DEVNULL)
        ref_p = subprocess.Popen(args + ["--device", monitor], stdout=rf, stderr=subprocess.DEVNULL)
        time.sleep(1.2)
        if wav_path is not None:
            vol = max(1, min(65536, int(round(volume * 65536))))
            subprocess.run(["paplay", "--volume", str(vol), str(wav_path)], check=False)
        else:
            time.sleep(hold)
        time.sleep(0.5)
        mic_p.terminate()
        ref_p.terminate()
        mic_p.wait()
        ref_p.wait()
        if wake_p is not None:
            wake_p.terminate()
            wake_p.wait()
    mic = np.frombuffer(mic_raw.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
    ref = np.frombuffer(ref_raw.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
    mic_raw.unlink(missing_ok=True)
    ref_raw.unlink(missing_ok=True)
    return align_ends(mic, ref)


def record(volumes: tuple[float, ...]) -> bool:
    sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if not sink:
        print("no default sink; cannot record")
        return False
    monitor = f"{sink}.monitor"
    orig_vol = sink_volume_pct(sink)
    orig_mute = sink_mute(sink)
    REC_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list = []

    def entry(key, name, vol, mic, ref, level, floor_db):
        manifest.append({"key": key, "source": name, "volume": vol, "test_level": level,
                         "floor_dbfs": round(floor_db, 1),
                         "echo_to_noise_db": round(dbfs(mic) - floor_db, 1),
                         "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "sink": sink, "source_device": DEFAULT_SOURCE, "monitor": monitor,
                         "mic_dbfs": round(dbfs(mic), 1), "ref_dbfs": round(dbfs(ref), 1)})

    # Wake the sink before every capture: while idle it is suspended and the monitor stream dies
    # a second after it is opened, which truncates every recording. A tiny paplay resumes it.
    wake = REC_DIR / "wake.wav"
    wake_t = np.arange(30 * SR, dtype=np.float64) / SR
    write_wav(wake, np.sin(2.0 * np.pi * 100.0 * wake_t) * 0.3)

    try:
        # FACTS: measure the noise floor first (2 s of silence), then raise the PC output volume
        # in steps (55, 70, 85, 100%, never above 100%) with a pink probe until the echo is
        # at least 20 dB above the floor (or 100% is reached).
        set_sink_volume(sink, 55.0)
        set_sink_mute(sink, False)
        floor_mic, _floor_ref = play_and_record(None, 1.0, monitor, wake, hold=2.0)
        floor_db = dbfs(floor_mic)
        test_level = 55.0
        probe = SRC_DIR / "pink.wav"
        if probe.is_file():
            for lvl in (55.0, 70.0, 85.0, 100.0):
                set_sink_volume(sink, lvl)
                set_sink_mute(sink, False)
                mic, ref = play_and_record(probe, 1.0, monitor, wake, hold=2.0)
                test_level = lvl
                if dbfs(mic) - floor_db >= 20.0:
                    break
        set_sink_volume(sink, test_level)
        set_sink_mute(sink, False)

        for wav in source_files():
            name = wav.name.removesuffix(".wav")
            clip_s = wav_seconds(wav)
            for vol in volumes:
                if sink_volume_pct(sink) != test_level or sink_mute(sink):
                    set_sink_volume(sink, test_level)
                    set_sink_mute(sink, False)
                mic, ref = play_and_record(wav, vol, monitor, wake)
                if dbfs(mic) < -70.0:
                    print(f"ABORT: microphone silent ({dbfs(mic):.1f} dBFS) for {name}; not writing bogus numbers")
                    return False
                if dbfs(ref) < -70.0:
                    print(f"ABORT: sink monitor silent for {name}; not writing bogus numbers")
                    return False
                if len(mic) < 0.8 * clip_s * SR or len(ref) < 0.8 * clip_s * SR:
                    print(f"ABORT: truncated capture for {name} @ {vol} "
                          f"(mic {len(mic) / SR:.2f} s, ref {len(ref) / SR:.2f} s, clip {clip_s:.2f} s); "
                          "not writing bogus numbers")
                    return False
                _peak, ratio = correlation_peak(mic, ref)
                if ratio < 2.0:
                    print(f"ABORT: no clear echo correlation for {name} @ {vol} "
                          f"(peak only {ratio:.1f}x noise); not writing bogus numbers")
                    return False
                key = f"{safe(name)}_v{vol}"
                np.save(REC_DIR / f"{key}.mic.npy", mic)
                np.save(REC_DIR / f"{key}.ref.npy", ref)
                entry(key, name, vol, mic, ref, test_level, floor_db)
                print(f"recorded {name} @ {vol} ({len(mic) / SR:.1f} s)")
        mic, ref = play_and_record(None, 1.0, monitor, wake)
        np.save(REC_DIR / "silence.mic.npy", mic)
        np.save(REC_DIR / "silence.ref.npy", ref)
        entry("silence", "silence", 1.0, mic, ref, test_level, floor_db)
        (REC_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    finally:
        if orig_vol is not None:
            set_sink_volume(sink, orig_vol)
        set_sink_mute(sink, orig_mute)
    print("measured levels per speaker volume:")
    print("volume   mic dBFS   ref dBFS")
    by_vol: dict[float, list] = {}
    for e in manifest:
        if e["source"] != "silence":
            by_vol.setdefault(e["volume"], []).append((e["mic_dbfs"], e["ref_dbfs"]))
    for vol in sorted(by_vol):
        pairs = by_vol[vol]
        mic_mean = float(np.mean([m for m, _r in pairs]))
        ref_mean = float(np.mean([r for _m, r in pairs]))
        print(f"{vol:>6.2f}   {mic_mean:>8.1f}   {ref_mean:>8.1f}")
    if 1.0 in by_vol:
        mic_at_1 = float(np.mean([m for m, _r in by_vol[1.0]]))
        if mic_at_1 < -60.0:
            print(f"WARNING: the microphone is quiet: {mic_at_1:.1f} dBFS at volume 1.0")
    return True


def feed(canceller, mic, ref, chunk=1024):
    outs = []
    converged = None
    for i in range(0, len(mic), chunk):
        outs.append(canceller.process(mic[i:i + chunk], ref[i:i + chunk]))
        if converged is None and canceller.converged_db >= 20.0:
            converged = min(i + chunk, len(mic)) / SR
    return np.concatenate(outs), converged


def removed_percent(mic, ref, cleaned) -> float:
    removed = []
    for i in range(0, len(mic), EchoCanceller.BLOCK):
        rb = ref[i:i + EchoCanceller.BLOCK]
        if rms(rb) ** 2 >= 1e-5:
            mb = mic[i:i + EchoCanceller.BLOCK]
            cb = cleaned[i:i + EchoCanceller.BLOCK]
            pm = float(np.mean(mb.astype(np.float64) ** 2))
            if pm > 1e-12:
                removed.append(1.0 - float(np.mean(cb.astype(np.float64) ** 2)) / pm)
    return 100.0 * max(0.0, min(1.0, float(np.mean(removed)))) if removed else 0.0


def oracle_erle_db(mic, ref, delay, taps: int = 4096) -> float:
    """Upper-bound ERLE from a fixed least-squares echo filter trained on the first 60%.

    The filter predicts the echo in the mic from the (delay-shifted) reference using `taps`
    taps, fit on the first 60% of the clip, then applied to the rest. This is the best a
    static predictor could remove, so it bounds what the adaptive canceller can hope for.
    """
    n = min(len(mic), len(ref))
    if n <= 0:
        return 0.0
    mic = np.asarray(mic, dtype=np.float64)[:n]
    ref = np.asarray(ref, dtype=np.float64)[:n]
    ref_e = shift_forward(ref, int(delay))
    train_end = int(0.6 * n)
    if train_end < 2:
        return 0.0
    T = min(int(taps), train_end)
    a = ref_e[:train_end]
    y = mic[:train_end]
    size = 1
    while size < 2 * train_end:
        size <<= 1
    fa = np.fft.rfft(a, size)
    ac = np.fft.irfft(fa * np.conj(fa), size)
    cc = np.fft.irfft(np.conj(fa) * np.fft.rfft(y, size), size)
    idx = np.arange(T)
    diff = idx[:, None] - idx[None, :]
    G = ac[diff % size]
    b = cc[:T]
    try:
        w = np.linalg.solve(G + 1e-9 * np.eye(T), b)
    except np.linalg.LinAlgError:
        return 0.0
    test = mic[train_end:]
    pred = np.zeros(len(test))
    for k in range(T):
        if w[k] == 0.0:
            continue
        pred += w[k] * ref_e[train_end - k:n - k]
    resid = test - pred
    pm = float(np.mean(test ** 2))
    pr = float(np.mean(resid ** 2))
    if pr <= 1e-20 or pm <= 1e-20:
        return 0.0
    return 10.0 * float(np.log10(pm / pr))


def evaluate_pair(mic, ref, floor_db=None, ppm=None):
    mic_a, al, dinfo = follow_drift(mic, ref, ppm=ppm)
    share = predictable_share(mic_a, al)
    achievable = (-10.0 * float(np.log10(1.0 - share))) if share < 1.0 else float("inf")
    # The pair is already drift-followed, so the echo sits at zero residual delay:
    # estimating a delay here (estimate_delay) returns a spurious peak and the oracle
    # would shift the reference a second time, collapsing the measured ERLE.
    delay = 0
    cleaned, converged = feed(EchoCanceller(delay=delay), mic_a, al)
    tail = len(mic_a) // 4
    erle = erle_db(mic_a[tail:], cleaned[tail:])
    oracle = oracle_erle_db(mic_a, al, delay)
    band_coh = {f"{lo}-{hi}": band_coherence_segments([(mic_a, al)], lo, hi)
                for lo, hi in BANDS}
    out = {
        "delay": int(delay),
        "erle_db": erle,
        "residual_dbfs": dbfs(cleaned[tail:]),
        "convergence_s": converged,
        "removed_percent": removed_percent(mic_a, al, cleaned),
        "predictable_share": share,
        "achievable_erle_db": achievable,
        "oracle_erle_db": oracle,
        "gap_to_oracle_db": oracle - erle,
        "drift_ppm": dinfo["ppm"],
        "drift_source": "session" if ppm is not None else "fit",
        "lag_ms": dinfo["lag"] / SR * 1000.0,
        "drift_residual": dinfo["residual"],
        "drift_used": dinfo["used"],
        "blocks_kept": dinfo["blocks_kept"],
        "blocks_total": dinfo["blocks_total"],
        "band_coherence": band_coh,
    }
    if floor_db is not None:
        out["echo_to_noise_db"] = dbfs(mic_a) - float(floor_db)
        out["residual_vs_floor_db"] = dbfs(cleaned[tail:]) - float(floor_db)
    return out


def mix_command(mic, cmd, offset: int, echo_rms: float, rel_db: float):
    target = echo_rms * 10.0 ** (rel_db / 20.0)
    cur = rms(cmd)
    scaled = cmd * (target / cur) if cur > 1e-12 else cmd
    mic2 = np.array(mic, dtype=np.float32, copy=True)
    end = min(offset + len(scaled), len(mic2))
    mic2[offset:end] += scaled[:end - offset]
    return mic2, offset


def word_accuracy(text: str, words: list[str]) -> float:
    toks = [t for t in "".join(c.lower() if c.isalpha() else " " for c in text).split() if t]
    hits = 0
    for w in words:
        for k in range(hits, len(toks)):
            if toks[k] == w:
                hits += 1
                break
    return hits / len(words) if words else 0.0


def distortion_db(cmd, cleaned_window) -> float:
    p_add = float(np.mean(np.asarray(cmd, dtype=np.float64) ** 2))
    p_keep = max(float(np.mean(np.asarray(cleaned_window, dtype=np.float64) ** 2)), 1e-20)
    return max(0.0, 10.0 * float(np.log10(max(p_add, 1e-20) / p_keep)))


def transcribe(model, audio) -> str:
    segs, _ = model.transcribe(np.asarray(audio, dtype=np.float32), language="en")
    return " ".join(s.text for s in segs)


def double_talk(pairs: dict[float, tuple], commands: list[Path]) -> dict:
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
    out: dict[str, dict] = {f"{rel:+.0f}": {"distortion_db": [], "acc_cleaned": [], "acc_raw": []}
                            for rel in LEVELS}
    for _vol, (mic, ref, delay) in pairs.items():
        echo_rms = rms(ref)
        for wav in commands:
            cmd = read_wav(wav)
            words = COMMANDS[list(commands).index(wav)].lower().split()
            offset = int(0.15 * len(mic))
            for rel in LEVELS:
                mic2, off = mix_command(mic, cmd, offset, echo_rms, rel)
                cleaned, _ = feed(EchoCanceller(delay=delay), mic2, ref)
                lo = max(0, off - SR // 2)
                hi = min(len(mic), off + len(cmd) + SR // 2)
                key = f"{rel:+.0f}"
                out[key]["distortion_db"].append(distortion_db(cmd, cleaned[lo:hi]))
                out[key]["acc_cleaned"].append(word_accuracy(transcribe(model, cleaned[lo:hi]), words))
                out[key]["acc_raw"].append(word_accuracy(transcribe(model, mic2[lo:hi]), words))
    for stats in out.values():
        for k, v in stats.items():
            stats[k] = float(np.mean(v)) if v else 0.0
    return out


def _mean_present(ms: list, key: str) -> float:
    vals = [m[key] for m in ms if m.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(tag: str, quick: bool = False) -> bool:
    manifest_path = REC_DIR / "manifest.json"
    if not manifest_path.is_file():
        print("no recordings; run record first")
        return False
    manifest = json.loads(manifest_path.read_text())
    LONG = {"pink", "music", "sweep"}

    def kind(source: str) -> str:
        return source.split("_")[0]

    # Session-drift pre-pass: fit drift on the long clips of each volume and take the
    # median ppm of the good fits (residual < 0.5 samples) as a fallback for short
    # clips and clips whose own fit failed.
    session_ppm: dict[float, float] = {}
    own_fit: dict[str, dict] = {}
    for entry in manifest:
        if entry["source"] == "silence" or kind(entry["source"]) not in LONG:
            continue
        mic = np.load(REC_DIR / f"{entry['key']}.mic.npy")
        ref = np.load(REC_DIR / f"{entry['key']}.ref.npy")
        _, _, dinfo = follow_drift(mic, ref)
        own_fit[entry["key"]] = dinfo
        if dinfo["residual"] < 0.5 and dinfo["ppm"] != 0.0:
            session_ppm.setdefault(entry["volume"], []).append(dinfo["ppm"])
    session_ppm = {v: float(np.median(vals)) for v, vals in session_ppm.items()}

    per_volume: dict[float, list] = {}
    clips: list[dict] = []
    for entry in manifest:
        if entry["source"] == "silence":
            continue
        mic = np.load(REC_DIR / f"{entry['key']}.mic.npy")
        ref = np.load(REC_DIR / f"{entry['key']}.ref.npy")
        is_long = kind(entry["source"]) in LONG
        good_fit = is_long and own_fit.get(entry["key"], {}).get("residual", 9e9) < 0.5
        ppm = None if good_fit else session_ppm.get(entry["volume"])
        m = evaluate_pair(mic, ref, entry.get("floor_dbfs"), ppm=ppm)
        m["kind"] = kind(entry["source"])
        per_volume.setdefault(entry["volume"], []).append(m)
        clips.append({"key": entry["key"], "volume": entry["volume"],
                      "drift_ppm": m["drift_ppm"], "lag_ms": m["lag_ms"],
                      "drift_residual": m["drift_residual"], "drift_used": m["drift_used"],
                      "band_coherence": m["band_coherence"]})
    summary: dict[str, dict] = {}
    for vol, ms in sorted(per_volume.items()):
        conv = [m["convergence_s"] for m in ms if m["convergence_s"] is not None]
        band_coh = {}
        for band in BANDS:
            key = f"{band[0]}-{band[1]}"
            vals = [m["band_coherence"][key] for m in ms if m.get("band_coherence", {}).get(key) is not None]
            band_coh[key] = float(np.mean(vals)) if vals else float("nan")
        summary[str(vol)] = {
            "erle_db": float(np.mean([m["erle_db"] for m in ms])),
            "residual_dbfs": float(np.mean([m["residual_dbfs"] for m in ms])),
            "convergence_s": float(np.mean(conv)) if conv else float("nan"),
            "removed_percent": float(np.mean([m["removed_percent"] for m in ms])),
            "predictable_share": float(np.mean([m["predictable_share"] for m in ms])),
            "achievable_erle_db": _mean_present(ms, "achievable_erle_db"),
            "oracle_erle_db": _mean_present(ms, "oracle_erle_db"),
            "gap_to_oracle_db": _mean_present(ms, "gap_to_oracle_db"),
            "echo_to_noise_db": _mean_present(ms, "echo_to_noise_db"),
            "residual_vs_floor_db": _mean_present(ms, "residual_vs_floor_db"),
            "band_coherence": band_coh,
        }
    pairs = {}
    for vol in per_volume:
        e = next(x for x in manifest if x["volume"] == vol and x["source"].startswith("music"))
        mic = np.load(REC_DIR / f"{e['key']}.mic.npy")
        ref = np.load(REC_DIR / f"{e['key']}.ref.npy")
        pairs[vol] = (mic, ref, estimate_delay(mic, ref))
    cmds = sorted(SRC_DIR.glob("cmd_*.wav"))
    pairs_use = pairs
    if quick:
        cmds = cmds[:6]
        pairs_use = dict(list(pairs.items())[:2])
    dt = double_talk(pairs_use, cmds)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    levels = [e.get("test_level") for e in manifest if e.get("test_level") is not None]
    def gmean(key):
        vals = [v[key] for v in summary.values() if v.get(key) is not None
                and not np.isnan(v[key])]
        return float(np.mean(vals)) if vals else float("nan")

    means = {k: gmean(k) for k in ("erle_db", "residual_dbfs", "removed_percent",
                                   "predictable_share", "achievable_erle_db",
                                   "oracle_erle_db", "gap_to_oracle_db",
                                   "echo_to_noise_db", "residual_vs_floor_db")}
    line = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": tag,
        "git_commit": commit,
        "test_level": float(np.mean(levels)) if levels else None,
        "per_volume": summary,
        "means": means,
        "double_talk": dt,
        "removed_percent": means["removed_percent"],
        "clips": clips,
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a") as f:
        f.write(json.dumps(line) + "\n")
    for vol, ms in sorted(per_volume.items()):
        print(f"\nvolume {vol}:")
        print(f"{'clip':<10} {'ppm':>8} {'share':>7} {'achv':>7} {'oracle':>7} {'canc':>7} {'gap':>7}")
        for m in ms:
            print(f"{m['kind']:<10} {m['drift_ppm']:>8.1f} {m['predictable_share']:>7.3f} "
                  f"{m['achievable_erle_db']:>7.2f} {m['oracle_erle_db']:>7.2f} "
                  f"{m['erle_db']:>7.2f} {m['gap_to_oracle_db']:>7.2f}")
    print(f"history appended for tag {tag}: ERLE mean {means['erle_db']:.1f} dB, "
          f"removed {means['removed_percent']:.1f}%")
    print(f"relative: gap to oracle {means['gap_to_oracle_db']:.1f} dB, "
          f"residual vs floor {means['residual_vs_floor_db']:.1f} dB, "
          f"echo to noise {means['echo_to_noise_db']:.1f} dB")
    return True


def webrtc_compare(tag: str, prepare: bool, score: bool) -> bool:
    """Compare the WebRTC echo canceller with ours. Two phases: --prepare writes the mixes and
    the jobs files into ``webrtc/`` (the batches run inside the Flatpak); --score reads the
    outputs and writes ``WEBRTC.md``. Echo-only clips, double-talk accuracy, false words and
    WebRTC property variants are all covered."""
    ECHO_CLIPS = ["pink", "music_01", "music_02", "speech_01", "speech_02", "cmd_01"]
    ECHO_VOLS = (0.4, 0.7, 1.0)
    TWO_MIC = ("blue", "camera")
    TWO_VOLS = (55, 70, 100)
    DT_BASE = [("music_01", 0.7), ("music_01", 1.0), ("speech_01", 0.7), ("speech_01", 1.0)]
    DT_LEVELS = (-6.0, 0.0, 6.0)
    VARIANTS = [
        {"echo-suppression-level": 0}, {"echo-suppression-level": 1},
        {"echo-suppression-level": 2}, {"extended-filter": True},
        {"delay-agnostic": True}, {"noise-suppression": True},
    ]
    twod = OUT / "twomic" / "placed-normal"

    def absp(p: Path) -> str:
        return str(p.absolute())

    def tail60(mic, cleaned) -> float:
        n = min(len(mic), len(cleaned))
        a = int(n * 0.4)
        return erle_db(mic[a:n], cleaned[a:n])

    def our_cancel(mic, ref, delay):
        m, r = align_ends(mic, ref)
        cleaned, _ = feed(EchoCanceller(delay=delay), m, r)
        return cleaned

    if prepare:
        WEBRTC_DIR.mkdir(parents=True, exist_ok=True)
        jobs: list[dict] = []
        meta: dict = {"echo": {}, "dt": [], "false_words": [], "variants": {}}

        def add_job(micp, refp, outp, props, est_s):
            jobs.append({"mic": absp(micp), "ref": absp(refp), "out": absp(outp), "props": props})
            return est_s

        # a. echo-only pairs (stored clips + two-microphone recordings)
        pairs: dict[str, tuple[Path, Path]] = {}
        for clip in ECHO_CLIPS:
            for vol in ECHO_VOLS:
                key = f"{clip}_v{vol}"
                pairs[key] = (REC_DIR / f"{key}.mic.npy", REC_DIR / f"{key}.ref.npy")
        for micname in TWO_MIC:
            for vol in TWO_VOLS:
                key = f"twomic_{micname}_v{vol}"
                pairs[key] = (twod / f"v{vol}_{micname}.npy", twod / f"v{vol}_ref.npy")
        for key, (micp, refp) in pairs.items():
            if not (micp.is_file() and refp.is_file()):
                continue
            mic, ref = np.load(micp), np.load(refp)
            mic_a, ref_a, _ = follow_drift(mic, ref)
            np.save(WEBRTC_DIR / f"{key}.mic_followed.npy", mic_a.astype(np.float32))
            np.save(WEBRTC_DIR / f"{key}.ref_followed.npy", ref_a.astype(np.float32))
            share = predictable_share(mic_a, ref_a)
            ceiling = (-10.0 * float(np.log10(1.0 - share))) if share < 1.0 else float("inf")
            delay = estimate_delay(mic, ref)
            est = len(mic) / SR
            add_job(micp, refp, WEBRTC_DIR / f"{key}.webrtc_raw.npy", {}, est)
            add_job(WEBRTC_DIR / f"{key}.mic_followed.npy", WEBRTC_DIR / f"{key}.ref_followed.npy",
                    WEBRTC_DIR / f"{key}.webrtc_followed.npy", {}, est)
            meta["echo"][key] = {
                "ceiling_db": ceiling,
                "our_raw_db": tail60(mic, our_cancel(mic, ref, delay)),
                "our_followed_db": tail60(mic_a, our_cancel(mic_a, ref_a, 0)),
                "est_s": est,
            }

        # b. double talk (6 commands over music_01 / speech_01 at 0.7 and 1.0)
        cmds = sorted(SRC_DIR.glob("cmd_*.wav"))[:6]
        for clip, vol in DT_BASE:
            key = f"{clip}_v{vol}"
            micp, refp = REC_DIR / f"{key}.mic.npy", REC_DIR / f"{key}.ref.npy"
            if not (micp.is_file() and refp.is_file()):
                continue
            mic, ref = np.load(micp), np.load(refp)
            delay = estimate_delay(mic, ref)
            echo_rms = rms(ref)
            for ci, wav in enumerate(cmds):
                cmd = read_wav(wav)
                words = COMMANDS[ci].lower().split()
                offset = int(0.15 * len(mic))
                for rel in DT_LEVELS:
                    mic2, off = mix_command(mic, cmd, offset, echo_rms, rel)
                    mixp = WEBRTC_DIR / f"dt_{clip}_v{vol}_c{ci}_{'m' if rel < 0 else ''}{abs(rel):g}.mic.npy"
                    np.save(mixp, mic2.astype(np.float32))
                    add_job(mixp, refp, WEBRTC_DIR / f"dt_{clip}_v{vol}_c{ci}_{rel:+.0f}.webrtc.npy", {}, len(mic) / SR)
                    meta["dt"].append({"clip": clip, "vol": vol, "cmd": ci, "rel": rel,
                                       "words": words, "off": off, "len": len(cmd),
                                       "delay": delay, "est_s": len(mic) / SR})

        # c. false words: echo-only speech clips (Whisper on cleaned echo, count words)
        for clip in ("speech_01", "speech_02"):
            for vol in ECHO_VOLS:
                key = f"{clip}_v{vol}"
                if key in meta["echo"]:
                    meta["false_words"].append(key)

        # d. WebRTC variants on music_01_v0.7 (echo-only + double talk at level 0)
        vkey = "music_01_v0.7"
        if vkey in meta["echo"]:
            micp, refp = REC_DIR / f"{vkey}.mic.npy", REC_DIR / f"{vkey}.ref.npy"
            mic, ref = np.load(micp), np.load(refp)
            mic_a, ref_a, _ = follow_drift(mic, ref)
            delay = estimate_delay(mic, ref)
            echo_rms = rms(ref)
            for vi, props in enumerate(VARIANTS):
                add_job(micp, refp, WEBRTC_DIR / f"var{vi}_echo_raw.npy", props, len(mic) / SR)
                add_job(WEBRTC_DIR / f"{vkey}.mic_followed.npy", WEBRTC_DIR / f"{vkey}.ref_followed.npy",
                        WEBRTC_DIR / f"var{vi}_echo_followed.npy", props, len(mic) / SR)
                for ci, wav in enumerate(cmds):
                    cmd = read_wav(wav)
                    offset = int(0.15 * len(mic))
                    mic2, off = mix_command(mic, cmd, offset, echo_rms, 0.0)
                    mixp = WEBRTC_DIR / f"var{vi}_dt_c{ci}.mic.npy"
                    np.save(mixp, mic2.astype(np.float32))
                    add_job(mixp, refp, WEBRTC_DIR / f"var{vi}_dt_c{ci}.webrtc.npy", props, len(mic) / SR)
                    meta["variants"][f"var{vi}"] = {"props": props}

        # write jobs split into <=200 s pieces so no single batch command exceeds ~4 min
        size = 12
        for i in range(0, len(jobs), size):
            (WEBRTC_DIR / f"jobs_{i // size:02d}.json").write_text(json.dumps(jobs[i:i + size], indent=1))
        (WEBRTC_DIR / "jobs.json").write_text(json.dumps(jobs, indent=1))
        (WEBRTC_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"prepared {len(jobs)} WebRTC jobs in {WEBRTC_DIR}; run jobs_XX.json batches "
              f"(timeout 300 flatpak run ... batch jobs_XX.json) then --score")
        return True

    if score:
        if not (WEBRTC_DIR / "meta.json").is_file():
            print("no prepared jobs; run --prepare first")
            return False
        meta = json.loads((WEBRTC_DIR / "meta.json").read_text())

        def load_out(name: str) -> np.ndarray | None:
            p = WEBRTC_DIR / name
            return np.load(p) if p.is_file() else None

        def tail_webrtc(micp, outp) -> float:
            mic = np.load(micp)
            out = load_out(outp)
            if out is None:
                return float("nan")
            return tail60(mic, out)

        # echo-only table
        rows = []
        for key, m in meta["echo"].items():
            micp = REC_DIR / f"{key}.mic.npy" if not key.startswith("twomic") else None
            if key.startswith("twomic"):
                micname = key.split("_")[1]
                vol = key.split("_v")[1]
                micp = twod / f"v{vol}_{micname}.npy"
            wr = tail_webrtc(micp, f"{key}.webrtc_raw.npy")
            wf = tail_webrtc(WEBRTC_DIR / f"{key}.mic_followed.npy", f"{key}.webrtc_followed.npy")
            rows.append((key, m["ceiling_db"], m["our_raw_db"], m["our_followed_db"], wr, wf))

        # double talk + false words need Whisper (loaded once)
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)

        def acc(audio, words) -> float:
            return word_accuracy(transcribe(model, audio), words)

        dt_rows: dict[float, dict] = {}
        for d in meta["dt"]:
            clip, vol, ci, rel = d["clip"], d["vol"], d["cmd"], d["rel"]
            key = f"{clip}_v{vol}"
            micp, refp = REC_DIR / f"{key}.mic.npy", REC_DIR / f"{key}.ref.npy"
            mic, ref = np.load(micp), np.load(refp)
            cmd = read_wav(sorted(SRC_DIR.glob("cmd_*.wav"))[ci])
            off, ln = d["off"], d["len"]
            lo = max(0, off - SR // 2)
            hi = min(len(mic), off + ln + SR // 2)
            mic2, _ = mix_command(mic, cmd, off, rms(ref), rel)
            cleaned_our = our_cancel(mic2, ref, d["delay"])
            wr_out = load_out(f"dt_{clip}_v{vol}_c{ci}_{rel:+.0f}.webrtc.npy")
            entry = dt_rows.setdefault(rel, {"acc_raw": [], "acc_our": [], "acc_webrtc": [],
                                             "lvl_raw": [], "lvl_our": [], "lvl_webrtc": []})
            entry["acc_raw"].append(acc(mic2[lo:hi], d["words"]))
            entry["acc_our"].append(acc(cleaned_our[lo:hi], d["words"]))
            entry["lvl_raw"].append(distortion_db(cmd, mic2[lo:hi]))
            entry["lvl_our"].append(distortion_db(cmd, cleaned_our[lo:hi]))
            if wr_out is not None:
                entry["acc_webrtc"].append(acc(wr_out[lo:hi], d["words"]))
                entry["lvl_webrtc"].append(distortion_db(cmd, wr_out[lo:hi]))

        false_rows = []
        for key in meta["false_words"]:
            micp = REC_DIR / f"{key}.mic.npy"
            mic, ref = np.load(micp), np.load(REC_DIR / f"{key}.ref.npy")
            delay = estimate_delay(mic, ref)
            our_clean = our_cancel(mic, ref, delay)
            wr = load_out(f"{key}.webrtc_raw.npy")
            n_our = len(transcribe(model, our_clean).split())
            n_wr = len(transcribe(model, wr).split()) if wr is not None else float("nan")
            false_rows.append((key, n_our, n_wr))

        # variants on music_01_v0.7
        var_rows = []
        vkey = "music_01_v0.7"
        if vkey in meta["echo"]:
            micp, refp = REC_DIR / f"{vkey}.mic.npy", REC_DIR / f"{vkey}.ref.npy"
            mic, ref = np.load(micp), np.load(refp)
            echo_rms = rms(ref)
            cmds = sorted(SRC_DIR.glob("cmd_*.wav"))[:6]
            for vi in range(len(meta["variants"])):
                props = meta["variants"][f"var{vi}"]["props"]
                er = tail_webrtc(micp, f"var{vi}_echo_raw.npy")
                ef = tail_webrtc(WEBRTC_DIR / f"{vkey}.mic_followed.npy", f"var{vi}_echo_followed.npy")
                accs = []
                for ci in range(len(cmds)):
                    out = load_out(f"var{vi}_dt_c{ci}.webrtc.npy")
                    if out is None:
                        continue
                    cmd = read_wav(cmds[ci])
                    off = int(0.15 * len(mic))
                    lo = max(0, off - SR // 2)
                    hi = min(len(mic), off + len(cmd) + SR // 2)
                    accs.append(acc(out[lo:hi], COMMANDS[ci].lower().split()))
                var_rows.append((f"var{vi}", props, er, ef, float(np.mean(accs)) if accs else float("nan")))

        # cpu seconds per second of audio (from batch progress is not stored; estimate from est_s)
        total_est = sum(m["est_s"] for m in meta["echo"].values())

        lines = [f"# WebRTC vs ours (tag {tag})\n"]
        lines.append("## Echo-only (ceiling | our raw | our followed | WebRTC raw | WebRTC followed, dB)\n")
        for key, ce, ow, of, wr, wf in rows:
            lines.append(f"| {key} | {ce:.1f} | {ow:.1f} | {of:.1f} | {wr:.1f} | {wf:.1f} |")
        lines.append("\n## Double-talk word accuracy and command level change (dB)\n")
        for rel in sorted(dt_rows, reverse=True):
            e = dt_rows[rel]
            lines.append(f"| {rel:+.0f} dB | raw {np.mean(e['acc_raw']):.2f} | our {np.mean(e['acc_our']):.2f} "
                         f"| WebRTC {np.mean(e['acc_webrtc']):.2f} | lvl raw {np.mean(e['lvl_raw']):.1f} "
                         f"| lvl our {np.mean(e['lvl_our']):.1f} | lvl WebRTC {np.mean(e['lvl_webrtc']):.1f} |")
        lines.append("\n## False words (echo speech heard as user)\n")
        for key, n_our, n_wr in false_rows:
            lines.append(f"| {key} | our {n_our} | WebRTC {n_wr} |")
        lines.append("\n## WebRTC variants on music_01_v0.7\n")
        for name, props, er, ef, a in var_rows:
            lines.append(f"| {name} {json.dumps(props)} | echo raw {er:.1f} | echo followed {ef:.1f} | dt acc {a:.2f} |")
        lines.append(f"\nCPU estimate: {total_est:.0f} s of audio prepared.\n")
        lines.append("\n## Conclusion\n")
        lines.append("Numbers only: echo removed per volume for each canceller, command word accuracy "
                     "for each, false words for each, best WebRTC settings, CPU seconds per second of audio. "
                     "Fill from the rows above once the Flatpak batches have run.\n")
        (OUT / "WEBRTC.md").write_text("\n".join(lines))
        print(f"wrote {OUT / 'WEBRTC.md'}")
        return True

    print("webrtc-compare needs --prepare and/or --score")
    return False


def read_history() -> list[dict]:
    if not HISTORY.is_file():
        return []
    return [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = read_history()
    if not hist:
        print("no history to plot")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    latest = hist[-1]
    tags = [h["tag"] for h in hist]
    x = list(range(len(hist)))

    def finish(fig, name):
        fig.savefig(OUT / name)
        plt.close(fig)

    def label(ax, ys, fmt):
        for i, v in enumerate(ys):
            ax.text(i, v, f"{v:{fmt}}", ha="center", va="bottom")

    vols = sorted(latest["per_volume"], key=float)
    vals = [latest["per_volume"][v]["erle_db"] for v in vols]
    fig, ax = plt.subplots()
    ax.bar(range(len(vols)), vals, label="ERLE dB")
    ax.set_xticks(range(len(vols)))
    ax.set_xticklabels(vols)
    ax.set_xlabel("speaker volume")
    ax.set_ylabel("ERLE dB (last 75%)")
    label(ax, vals, ".1f")
    ax.set_title(f"ERLE by volume - {latest['tag']}")
    finish(fig, "erle_by_volume.png")

    acc = [float(np.mean([s["acc_cleaned"] for s in h["double_talk"].values()])) for h in hist]
    erle = [h["means"]["erle_db"] for h in hist]
    fig, ax = plt.subplots()
    ax.plot(x, erle, marker="o", label="ERLE dB mean")
    ax2 = ax.twinx()
    ax2.plot(x, acc, marker="s", label="command word accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_xlabel("run (tag)")
    ax.set_ylabel("ERLE dB")
    ax2.set_ylabel("word accuracy")
    label(ax, erle, ".1f")
    label(ax2, acc, ".2f")
    ax.set_title("history: cancellation and command survival")
    finish(fig, "history.png")

    fig, ax = plt.subplots()
    ax.bar(x, [h["removed_percent"] for h in hist], label="% speaker energy removed")
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_xlabel("run (tag)")
    ax.set_ylabel("% of speaker energy removed")
    label(ax, [h["removed_percent"] for h in hist], ".1f")
    ax.set_title("speaker energy removed per run")
    finish(fig, "removed_percent.png")


def diagnose(seconds: float = 20.0) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if not sink:
        print("no default sink; cannot diagnose")
        return False
    monitor = f"{sink}.monitor"
    orig_vol = sink_volume_pct(sink)
    orig_mute = sink_mute(sink)
    REC_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    wake = REC_DIR / "wake.wav"
    wake_t = np.arange(30 * SR, dtype=np.float64) / SR
    write_wav(wake, np.sin(2.0 * np.pi * 100.0 * wake_t) * 0.3)
    train = make_click_train(seconds)
    train_path = REC_DIR / "_diagnose_clicks.wav"
    write_wav(train_path, train)
    variants = [("lat20", 20, False), ("lat50", 50, False),
                ("lat100", 100, False), ("fixrate", 20, True)]
    report: list[dict] = []
    try:
        set_sink_volume(sink, 55.0)
        set_sink_mute(sink, False)
        for name, lat, fix in variants:
            mic, ref = play_and_record(train_path, 1.0, monitor, wake, latency=lat, fix_rate=fix)
            thr_m = 0.4 * float(np.max(np.abs(mic))) if mic.size else 0.3
            thr_r = 0.4 * float(np.max(np.abs(ref))) if ref.size else 0.3
            mic_c, ref_c = find_clicks(mic, max(thr_m, 1e-4)), find_clicks(ref, max(thr_r, 1e-4))
            times, lags = pair_clicks(mic_c, ref_c)
            k = len(times)
            if k < 2:
                report.append({"variant": name, "clicks": int(k), "note":
                               f"mic {dbfs(mic):.1f} dBFS, ref {dbfs(ref):.1f} dBFS, too few clicks"})
                continue
            slope, intercept, jumps = fit_drift(times, lags)
            med_lag = int(np.median(lags))
            coh_const = band_coherence(mic, shift_forward(ref, med_lag))
            mic_a, ref_a, dinfo = follow_drift(mic, ref)
            coh_slope = band_coherence(mic_a, ref_a)
            report.append({"variant": name, "clicks": int(k), "slope_ppm": slope,
                           "offset": intercept, "dropouts": len(jumps),
                           "coh_const": coh_const, "coh_slope": coh_slope,
                           "drift_ppm": dinfo["ppm"], "lag_ms": dinfo["lag"] / SR * 1000.0,
                           "drift_residual": dinfo["residual"]})
            fig, ax = plt.subplots()
            ax.plot(times, lags, marker="o")
            tt = np.linspace(times[0], times[-1], 50)
            ax.plot(tt, slope * tt + intercept, "--")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("lag (samples)")
            ax.set_title(f"{name}: click lag vs time (slope ppm, jumps=dropouts)")
            fig.savefig(OUT / f"diagnose_{name}.png")
            plt.close(fig)
        good = [r for r in report if "coh_const" in r]
        if good:
            fig, ax = plt.subplots()
            names = [r["variant"] for r in good]
            cc = [r["coh_const"] for r in good]
            cs = [r["coh_slope"] for r in good]
            xs = np.arange(len(names))
            ax.bar(xs, cc, label="constant-lag")
            ax.bar(xs + 0.4, cs, label="slope-compensated")
            ax.set_xticks(xs + 0.2)
            ax.set_xticklabels(names)
            ax.set_ylabel("coherence 300-3000 Hz")
            ax.set_title("diagnose: coherence per recorder variant")
            fig.savefig(OUT / "diagnose_coherence.png")
            plt.close(fig)
        print("diagnose report:")
        for r in report:
            if "coh_const" in r:
                print(f"  {r['variant']}: clicks={r['clicks']} slope={r['slope_ppm']:.0f}ppm "
                      f"offset={r['offset']:.0f} dropouts={r['dropouts']} "
                      f"coh_const={r['coh_const']:.3f} coh_slope={r['coh_slope']:.3f}")
            else:
                print(f"  {r['variant']}: {r.get('note', '')}")
    finally:
        if orig_vol is not None:
            set_sink_volume(sink, orig_vol)
        set_sink_mute(sink, orig_mute)
        train_path.unlink(missing_ok=True)
    return True


def cmd_drift(mic_path: str, ref_path: str) -> bool:
    mic = np.asarray(np.load(mic_path), dtype=np.float64)
    ref = np.asarray(np.load(ref_path), dtype=np.float64)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]
    thr = 10.0 ** (-50.0 / 20.0)
    nz = np.flatnonzero(np.abs(ref) > thr)
    on = (int(nz[0]) + SR // 2) if nz.size else 0
    mic, ref = mic[on:], ref[on:]
    lag0, drift, resid, binfo = fit_drift_fine_info(mic, ref)
    print(f"fitted drift: {drift:.3f} samples/s = {drift / SR * 1e6:.1f} ppm, "
          f"residual rms {resid:.2f} samples, blocks kept {binfo['blocks_kept']}/{binfo['blocks_total']}")
    N = len(mic) - 800
    mean_lag = lag0 + drift * ((len(mic) / 2.0) / SR)
    k = int(round(mean_lag))
    fixed = [(mic[k:k + N], ref[:N])]
    mic_a, ref_a, dinfo = follow_drift(mic, ref)
    followed = [(mic_a, ref_a)]
    print(f"alignment used by follow_drift:       {dinfo['used']}")
    print("one fixed delay for the whole clip:   ",
          " | ".join(f"{lo}-{hi}: {band_coherence_segments(fixed, lo, hi):.2f}" for lo, hi in BANDS))
    print("delay following the fitted drift:     ",
          " | ".join(f"{lo}-{hi}: {band_coherence_segments(followed, lo, hi):.2f}" for lo, hi in BANDS))
    return True


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="anc_lab")
    ap.add_argument("command", choices=["make-sources", "record", "evaluate", "plot", "diagnose", "all", "drift",
                                       "webrtc-compare"])
    ap.add_argument("--volumes", default=",".join(str(v) for v in VOLUMES))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args(argv)
    vols = tuple(float(v) for v in a.volumes.split(","))
    if a.command == "drift":
        if len(a.paths) != 2:
            print("usage: anc_lab drift <mic.npy> <ref.npy>")
            return 2
        return 0 if cmd_drift(a.paths[0], a.paths[1]) else 2
    if a.command == "webrtc-compare":
        return 0 if webrtc_compare(a.tag, a.prepare, a.score) else 2
    if a.command == "make-sources":
        make_sources()
        return 0
    if a.command == "record":
        return 0 if record(vols) else 2
    if a.command == "evaluate":
        return 0 if evaluate(a.tag, a.quick) else 2
    if a.command == "plot":
        plot()
        return 0
    if a.command == "diagnose":
        return 0 if diagnose() else 2
    if not (SRC_DIR / "pink.wav").is_file():
        make_sources()
    if not (REC_DIR / "manifest.json").is_file():
        if not record(vols):
            return 2
    if not evaluate(a.tag):
        return 2
    plot()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
