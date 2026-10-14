"""Measure the Silero voice detector against today's loudness rule on the read-aloud rounds.

For every round x {blue, blue.webrtc} it reports, over 20 ms chunks:
  * recall  - share of the owner's speech windows marked as voice (want high)
  * false+  - share of the time OUTSIDE those windows marked as voice (want low: music/TV)
for (a) today's rule (chunk RMS*32768 >= 200) and (b) the detector at thresholds 0.3/0.5/0.7.
The table is written to ``.voxa-spec/out/anclab/VOICE.md``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Voxa")
sys.path.insert(0, "/Voxa/tools")

from anc_lab import SR, read_wav  # noqa: E402

from voxa.voicedetect import VoiceDetector

OUT = Path("/Voxa/.voxa-spec/out/anclab/readscript")
CHUNK = 320  # 20 ms at 16 kHz
RULE_LEVEL = 200.0  # today's rule: chunk RMS * 32768 >= 200


def xc(a: np.ndarray, b: np.ndarray) -> int:
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    k = int(np.argmax(np.abs(c)))
    return k if k < n // 2 else k - n


def in_window(n_chunks: int, windows: list[tuple[int, int]]) -> np.ndarray:
    flags = np.zeros(n_chunks, dtype=bool)
    for s, e in windows:
        lo = max(0, s // CHUNK)
        hi = min(n_chunks, (e + CHUNK - 1) // CHUNK)
        flags[lo:hi] = True
    return flags


def rule_voice(mic: np.ndarray) -> np.ndarray:
    n = len(mic) // CHUNK
    blocks = mic[: n * CHUNK].reshape(n, CHUNK).astype(np.float64)
    rms = np.sqrt((blocks ** 2).mean(axis=1))
    return rms * 32768.0 >= RULE_LEVEL


def detector_voice(mic: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    pcm = (np.clip(mic, -1.0, 1.0) * 32768.0).astype("<i2")
    det = VoiceDetector(threshold=threshold)
    n = len(pcm) // CHUNK
    out = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    for i in range(n):
        out[i] = det.feed(pcm[i * CHUNK : (i + 1) * CHUNK].tobytes())
    dt = time.perf_counter() - t0
    return out, (dt / n * 1000.0 if n else 0.0)


def score(voice: np.ndarray, win: np.ndarray) -> tuple[float, float]:
    inside = int(win.sum())
    outside = int((~win).sum())
    rec = float((voice & win).sum()) / inside if inside else float("nan")
    fp = float((voice & ~win).sum()) / outside if outside else float("nan")
    return rec, fp


def main() -> None:
    man = json.load(open(OUT / "manifest.json"))
    rows: list[str] = []
    for rnd, info in man["rounds"].items():
        ref = np.load(OUT / f"{rnd}_ref.npy").astype(np.float32)
        played = np.asarray(read_wav(OUT / f"{rnd}_played.wav"), np.float32)
        off = xc(ref, played)
        for mic_name in ("blue", "blue.webrtc"):
            mic = np.load(OUT / f"{rnd}_{mic_name}.npy").astype(np.float32)
            n = len(mic) // CHUNK
            windows = [
                (max(0, off + int(a * SR)), off + int(b * SR))
                for a, b in info["line_windows_s"]
            ]
            win = in_window(n, windows)
            methods = {"rule": (rule_voice(mic[: n * CHUNK]), 0.0)}
            for th in (0.3, 0.5, 0.7):
                v, ms = detector_voice(mic, th)
                methods[f"vad{th}"] = (v, ms)
            for name, (voice, ms) in methods.items():
                rec, fp = score(voice, win)
                rows.append((rnd, mic_name, name, rec, fp, ms))

    lines = [
        "# Voice detector vs loudness rule (N09a)",
        "",
        "recall = share of owner speech windows marked voice (want high);",
        "false+ = share of time outside those windows marked voice (want low).",
        "rule = today's RMS*32768>=200; vadX = Silero at threshold X; ms = per 20 ms chunk.",
        "",
        "| round | mic | method | recall | false+ | ms/chunk |",
        "|-------|-----|--------|--------|--------|----------|",
    ]
    for rnd, mic_name, name, rec, fp, ms in rows:
        lines.append(f"| {rnd} | {mic_name} | {name} | {rec*100:.1f}% | {fp*100:.1f}% | {ms:.3f} |")
    text = "\n".join(lines) + "\n"
    (Path("/Voxa/.voxa-spec/out/anclab") / "VOICE.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
