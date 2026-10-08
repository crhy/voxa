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
from voxa.sync import band_coherence, band_coherence_segments, fit_drift_fine_info, follow_drift

SR = 16000
OUT = Path(".voxa-spec/out/anclab")
SRC_DIR = OUT / "sources"
REC_DIR = OUT / "recordings"
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


def evaluate_pair(mic, ref, floor_db=None):
    mic_a, al, dinfo = follow_drift(mic, ref)
    coh = band_coherence(mic_a, al)
    delay = estimate_delay(mic_a, al)
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
        "coherence": coh,
        "oracle_erle_db": oracle,
        "gap_to_oracle_db": oracle - erle,
        "drift_ppm": dinfo["ppm"],
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
    per_volume: dict[float, list] = {}
    clips: list[dict] = []
    for entry in manifest:
        if entry["source"] == "silence":
            continue
        mic = np.load(REC_DIR / f"{entry['key']}.mic.npy")
        ref = np.load(REC_DIR / f"{entry['key']}.ref.npy")
        m = evaluate_pair(mic, ref, entry.get("floor_dbfs"))
        per_volume.setdefault(entry["volume"], []).append(m)
        clips.append({"key": entry["key"], "volume": entry["volume"],
                      "drift_ppm": m["drift_ppm"], "lag_ms": m["lag_ms"],
                      "drift_residual": m["drift_residual"], "drift_used": m["drift_used"],
                      "band_coherence": m["band_coherence"]})
    summary: dict[str, dict] = {}
    for vol, ms in sorted(per_volume.items()):
        conv = [m["convergence_s"] for m in ms if m["convergence_s"] is not None]
        summary[str(vol)] = {
            "erle_db": float(np.mean([m["erle_db"] for m in ms])),
            "residual_dbfs": float(np.mean([m["residual_dbfs"] for m in ms])),
            "convergence_s": float(np.mean(conv)) if conv else float("nan"),
            "removed_percent": float(np.mean([m["removed_percent"] for m in ms])),
            "coherence": float(np.mean([m["coherence"] for m in ms])),
            "oracle_erle_db": _mean_present(ms, "oracle_erle_db"),
            "gap_to_oracle_db": _mean_present(ms, "gap_to_oracle_db"),
            "echo_to_noise_db": _mean_present(ms, "echo_to_noise_db"),
            "residual_vs_floor_db": _mean_present(ms, "residual_vs_floor_db"),
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

    means = {k: gmean(k) for k in ("erle_db", "residual_dbfs", "removed_percent", "coherence",
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
    print(f"history appended for tag {tag}: ERLE mean {means['erle_db']:.1f} dB, "
          f"removed {means['removed_percent']:.1f}%")
    print(f"relative: gap to oracle {means['gap_to_oracle_db']:.1f} dB, "
          f"residual vs floor {means['residual_vs_floor_db']:.1f} dB, "
          f"echo to noise {means['echo_to_noise_db']:.1f} dB")
    return True


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
    ap.add_argument("command", choices=["make-sources", "record", "evaluate", "plot", "diagnose", "all", "drift"])
    ap.add_argument("--volumes", default=",".join(str(v) for v in VOLUMES))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args(argv)
    vols = tuple(float(v) for v in a.volumes.split(","))
    if a.command == "drift":
        if len(a.paths) != 2:
            print("usage: anc_lab drift <mic.npy> <ref.npy>")
            return 2
        return 0 if cmd_drift(a.paths[0], a.paths[1]) else 2
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
