#!/usr/bin/env python3
"""Guided session recorder: ONE ~10-minute session of real hearing data for the bench (crhy/voxa#72).

Usage:
  .venv/bin/python3 tools/record_session.py --prepare       # fetch the CC0 music + render Voxa's TTS lines
  .venv/bin/python3 tools/record_session.py                 # run the guided session (opens mic + speakers)
  .venv/bin/python3 tools/record_session.py --to-bench DIR  # turn a recorded session into a bench suite
  .venv/bin/python3 tools/record_session.py --dry-run       # print the session plan, touch nothing

Extends tools/anc_lab.py, the only live-capture code on main (there is no wall_run.py; the hearing
bench itself lives on the f1xx branches, and its suite format is reproduced here). Every take records
the DEFAULT MICROPHONE and the exact playback reference, sample-aligned: the sink's monitor source
(``parec --device <sink>.monitor``) or, when no monitor source exists, the samples this tool played
itself, placed on a clock-aligned timeline. 48 kHz, WAV per take plus one JSON per take. The JSON
carries what played (file + exact start sample), rhY's prompted text, the expected label
(user/echo/noise/bargein/backchannel), devices, sample rates, the measured playback->mic delay
(cross-correlation) and the SYSTEM VOLUME AS READ. The tool never lowers, sets or mutes the volume
(rhY's rule): it reads it and records it, nothing else. It closes every recorder it opened before
it ends.

The converter writes a bench-shaped suite (mic.wav + ref.wav at 16 kHz int16 mono + truth.json per
clip, the format voxa/bench's runner.load_clip reads) under ``<bench-root>/real/``, so
``python -m voxa.bench run --suite real`` can run it once the bench lands on main.

Never run this on the fleet: it opens the microphone and plays sound. Building and testing it here
uses FAIRLY cheap fakes - see tests/test_record_session.py, which never touches a real device.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR = 48000  # every recording and asset: 48 kHz (the bench downmixes to its 16 kHz)
BENCH_SR = 16000  # voxa/bench works at 16 kHz (voxa/audio.py REFERENCE_RATE)
DATA = Path("/home/rhy/voxa-bench-data")  # bench data home (never /tmp: it is RAM)
MUSIC_DIR = DATA / "music"
TTS_CACHE = DATA / "voxa-tts-cache"
RECORDINGS = Path("/home/rhy/voxa-recordings")

# The one CC0 music track, fetched once by --prepare into MUSIC_DIR (URL + SHA-256 in music.json).
MUSIC_URL = "https://archive.org/download/cum.131-Aysu-Sinusitis/08.Pretzel.mp3"
MUSIC_FILE = "aysu-sinusitis-pretzel.mp3"
MUSIC_LICENSE = "CC0 1.0 (Cold Unity Records CUM 131, Aysu - Sinusitis, track 08 Pretzel)"

# Voxa's own voice: voxa/config.py Settings.tts_voice + DEFAULT_TTS_RATE. The echo we record has to
# be hers, so no stand-in voice ever substitutes it.
VOXA_TTS_VOICE = "en-US-AriaNeural"


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "x"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# WAV helpers (anc_lab's contract: 16-bit mono PCM WAVs)
# ---------------------------------------------------------------------------------------------

def write_wav(path: Path, samples: np.ndarray, rate: int = SR) -> None:
    x = np.clip(np.rint(np.asarray(samples, dtype=np.float64) * 32767.0), -32768, 32767)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.astype("<i2").tobytes())


def read_wav_mono(path: Path, expect_rate: int | None = None) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit mono WAVs are supported")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if expect_rate is not None and rate != expect_rate:
        raise ValueError(f"{path}: expected {expect_rate} Hz, found {rate} Hz")
    return x, rate


def resample_to_rate(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited resample. The bench path is the exact 3:1 48k->16k decimation; anything else
    falls back to linear interpolation (fine for an asset, never used for a measurement)."""
    x = np.asarray(x, dtype=np.float64)
    if src_rate == dst_rate or len(x) == 0:
        return x.astype(np.float32)
    g = np.gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    if up == 1 and down >= 2:
        taps = 24 * down + 1
        k = np.arange(taps)
        fir = np.sinc((k - (taps - 1) / 2.0) / down) * np.kaiser(taps, 8.0) / down
        y = np.convolve(x, fir, mode="full")
        return y[taps - 1::down].astype(np.float32)
    n = max(1, int(len(x) * dst_rate / src_rate))
    return np.interp(np.arange(n), np.linspace(0.0, len(x) - 1.0, len(x)), x).astype(np.float32)


def to_wav48(src: Path, dst: Path) -> Path:
    """Any audio file (mp3/ogg/wav) to a 48 kHz 16-bit mono WAV via ffmpeg, a system dependency
    already assumed by tools/anc_lab.py helpers and voxa/bench/render.py on the bench branches."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )
    return dst


# ---------------------------------------------------------------------------------------------
# delay / alignment (anc_lab.correlation_peak generalized to any sample rate)
# ---------------------------------------------------------------------------------------------

def correlation_peak(mic: np.ndarray, ref: np.ndarray, sr: int = SR,
                     max_delay: int | None = None) -> tuple[int, float]:
    """Best playback->mic delay in samples, and how far its peak stands above correlation noise."""
    n = min(len(mic), len(ref), 10 * sr)
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
    window = np.abs(corr[:(max_delay if max_delay is not None else sr) + 1])
    med = float(np.median(window))
    peak = int(np.argmax(window))
    ratio = float(window[peak]) / med if med > 0 else float("inf")
    return peak, ratio


def measure_delay(mic: np.ndarray, ref: np.ndarray, sr: int = SR,
                  max_delay_ms: float = 500.0) -> dict | None:
    """The measured playback->mic delay of a take that played something; None when nothing in this
    take correlates. The whole point of r01: one number per take, measured, never assumed."""
    samples, ratio = correlation_peak(mic, ref, sr=sr, max_delay=int(sr * max_delay_ms / 1000.0))
    if ratio < 2.0:
        return None
    return {"samples": samples, "ms": round(samples * 1000.0 / sr, 3), "ratio": round(ratio, 1),
            "method": "fft cross-correlation (anc_lab.correlation_peak generalized to 48 kHz)"}


def asset_start(big: np.ndarray, small: np.ndarray, sr: int = SR) -> int | None:
    """Where the exact played samples sit inside a capture (mic or reference): their start sample.
    One-sided search: correlate the WHOLE capture against the short exact signal, so a play that
    happens 30 s into a take is still found (the prefix cap of correlation_peak is not enough)."""
    a = np.asarray(big, dtype=np.float64)
    b = np.asarray(small, dtype=np.float64)
    if len(a) == 0 or len(b) == 0 or len(b) > len(a):
        return None
    size = 1
    while size < len(a) + len(b):
        size <<= 1
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)))
    window = np.abs(corr[: len(a) - len(b) + 1])
    med = float(np.median(window))
    peak = int(np.argmax(window))
    ratio = float(window[peak]) / med if med > 0 else float("inf")
    return peak if ratio >= 2.0 else None


# ---------------------------------------------------------------------------------------------
# devices: player, recorders, probes. The tests inject fakes with the SAME methods.
# ---------------------------------------------------------------------------------------------

class Player:
    """Plays one file to the default sink and blocks until it is done (anc_lab's paplay)."""

    def play(self, path: Path) -> None:
        subprocess.run(["paplay", str(path)], check=False)


class Recorder:
    """One parec capture on one device: start() opens it, stop() closes it and returns samples.
    stop() is what closes the device; the session always calls it (rule: close everything)."""

    device: str

    def __init__(self, device: str, sr: int, raw_path: Path) -> None:
        self.device = device
        self.sr = sr
        self.raw_path = raw_path
        self._proc: subprocess.Popen | None = None
        self._file = None

    def start(self) -> None:
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_path.unlink(missing_ok=True)
        self._file = open(self.raw_path, "wb")
        self._proc = subprocess.Popen(
            ["parec", "--rate", str(self.sr), "--channels", "1", "--format", "s16le", "--raw",
             "--latency-msec=20", "--device", self.device],
            stdout=self._file, stderr=subprocess.DEVNULL,
        )

    def stop(self) -> np.ndarray:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
        if self._file is not None:
            self._file.close()
            self._file = None
        raw = self.raw_path.read_bytes()
        self.raw_path.unlink(missing_ok=True)
        n = len(raw) // 2 * 2
        return np.frombuffer(raw[:n], dtype="<i2").astype(np.float32) / 32768.0


def _pactl(*args: str) -> str:
    proc = subprocess.run(["pactl", *args], capture_output=True, text=True, check=False)
    return proc.stdout if proc.returncode == 0 else ""


def default_sink() -> str:
    return _pactl("get-default-sink").strip()


def default_source() -> str:
    return _pactl("get-default-source").strip()


def monitor_source(sink: str) -> str:
    """The sink's monitor source, exactly like anc_lab.record(); verified against the source list.
    Empty when the sound server has none (then the tool falls back to synthetic reference)."""
    listed = [line.split("\t")[1] for line in _pactl("list", "short", "sources").splitlines()
              if "\t" in line]
    wanted = f"{sink}.monitor"
    if wanted in listed:
        return wanted
    for cand in listed:
        if cand.endswith(".monitor") and sink in cand:
            return cand
    return ""


def read_system_volume(sink: str) -> dict:
    """READ ONLY, rhY's rule (docs/notes/voxatest-never-lower-volume.md): never set-sink-volume,
    never set-sink-mute. Read the volume, record it in the metadata, nothing else."""
    m = re.search(r"(\d+)%", _pactl("get-sink-volume", sink))
    mute_out = _pactl("get-sink-mute", sink)
    muted = ("yes" in mute_out.lower()) if mute_out.strip() else None
    return {"pct": float(m.group(1)) if m else None, "muted": muted,
            "tool": "pactl get-sink-volume / get-sink-mute (read only, never set)"}


def devices_info(sink: str, mic: str, monitor: str) -> dict:
    return {"default_sink": sink, "default_source": mic, "monitor_source": monitor,
            "recorder": "parec (PulseAudio / PipeWire-Pulse)", "player": "paplay"}


# ---------------------------------------------------------------------------------------------
# the session script: one guided ~10-minute session, 17 takes
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Playback:
    asset: str  # asset key: "voxa:<text>" or "music:<file>"
    from_s: float = 0.0  # start this many seconds into the asset (music excerpts)
    seconds: float | None = None  # cap the asset length


@dataclasses.dataclass
class Take:
    id: str
    kind: str  # room / voxa / music / bargein / music_over / backchannel / far / noise
    label: str  # expected label: user / echo / noise / bargein / backchannel
    instruction: str  # what rhY is told to do, shown on screen
    prompt: str | None = None  # the exact text rhY says, when there is one
    playback: tuple[Playback, ...] = ()
    note: str = ""  # timing hint shown during the take ("say it ~4 s in")
    tail_s: float = 3.0  # keep recording this long after playback ends / after the countdown

    def seconds(self) -> float:
        played = sum(spec.seconds or 7.0 for spec in self.playback)  # 7 s: typical Voxa line
        return 4.0 + self.tail_s + played  # enter + wake + preroll + countdown + capture window


SCRIPT: tuple[Take, ...] = (
    # rhY alone, quiet room: the reference "user" clips
    Take("room-01", "room", "user", "Read this line, normally close to the mic, normal voice.",
         prompt="Remind me to buy milk, eggs and coffee after six o'clock.", tail_s=20.0),
    Take("room-02", "room", "user", "Read this line, same distance, same voice.",
         prompt="Turn the brightness down and open the notes app, please.", tail_s=20.0),
    # Voxa speaking alone: pure echo
    Take("voxa-01", "voxa", "echo", "Do NOT say anything while Voxa speaks.",
         playback=(Playback("voxa:Hello! I am Voxa. What can I do for you?"),), tail_s=3.0),
    Take("voxa-02", "voxa", "echo", "Do NOT say anything while Voxa speaks.",
         playback=(Playback("voxa:You have three reminders. Standup at nine, call Rho at noon, "
                            "and take out the trash at six."),), tail_s=3.0),
    # music alone: pure noise
    Take("music-01", "music", "noise", "Do NOT say anything while the music plays.",
         playback=(Playback(f"music:{MUSIC_FILE}", from_s=2.0, seconds=12.0),), tail_s=3.0),
    Take("music-02", "music", "noise", "Do NOT say anything while the music plays.",
         playback=(Playback(f"music:{MUSIC_FILE}", from_s=20.0, seconds=12.0),), tail_s=3.0),
    # rhY talking OVER Voxa (barge-in): "stop", "wait", a real request
    Take("barge-01", "bargein", "bargein",
         "Voxa reads a long line. Interrupt her about four seconds in by saying: stop",
         prompt="stop",
         playback=(Playback("voxa:Opening your calendar. You have standup at nine, lunch with "
                            "Dana at twelve, and the dentist at four."),),
         note="say 'stop' ~4 s after her voice starts", tail_s=8.0),
    Take("barge-02", "bargein", "bargein",
         "Voxa reads the long line again. Interrupt about four seconds in by saying: wait",
         prompt="wait",
         playback=(Playback("voxa:Opening your calendar. You have standup at nine, lunch with "
                            "Dana at twelve, and the dentist at four."),),
         note="say 'wait' ~4 s after her voice starts", tail_s=8.0),
    Take("barge-03", "bargein", "bargein",
         "Voxa reads the long line again. Interrupt about four seconds in with a real request: "
         "Voxa, what time is it?",
         prompt="Voxa, what time is it?",
         playback=(Playback("voxa:Opening your calendar. You have standup at nine, lunch with "
                            "Dana at twelve, and the dentist at four."),),
         note="say the request ~4 s after her voice starts", tail_s=8.0),
    # rhY talking over music
    Take("music-over-01", "music_over", "user",
         "Music plays. Say this OVER it, normal voice.",
         prompt="What is the weather tomorrow?",
         playback=(Playback(f"music:{MUSIC_FILE}", from_s=8.0, seconds=8.0),), tail_s=10.0),
    Take("music-over-02", "music_over", "user",
         "Music plays. Read this line OVER it.",
         prompt="Write a note: call the garage about the brakes on Friday.",
         playback=(Playback(f"music:{MUSIC_FILE}", from_s=26.0, seconds=10.0),), tail_s=10.0),
    # quiet back-channel while Voxa talks
    Take("back-01", "backchannel", "backchannel",
         "Voxa speaks. While she talks, ONCE only, quietly: mm-hm",
         prompt="mm-hm",
         playback=(Playback("voxa:The weather in Thessaloniki is clear with a high of eighteen "
                            "degrees."),),
         note="say 'mm-hm' once, quietly, while she talks", tail_s=6.0),
    Take("back-02", "backchannel", "backchannel",
         "Voxa speaks. While she talks, ONCE only, quietly: okay",
         prompt="okay",
         playback=(Playback("voxa:I found one message from Rho. He says the flight is "
                            "confirmed."),),
         note="say 'okay' once, quietly, while she talks", tail_s=6.0),
    # far from the mic
    Take("far-01", "far", "user",
         "Step two or three meters away from the mic and read this line, normal voice.",
         prompt="Search for the best hotel in Athens under eighty euros a night.", tail_s=20.0),
    Take("far-02", "far", "user", "Still away from the mic: read this line.",
         prompt="Play some jazz on the radio and keep the volume low.", tail_s=20.0),
    # the room makes sounds
    Take("noise-01", "noise", "noise", "Do NOT speak. Cough ONCE, normally, about two seconds in.",
         note="one cough ~2 s after the countdown", tail_s=6.0),
    Take("noise-02", "noise", "noise",
         "Do NOT speak. Make a real room noise: knock on the desk, close a door, or type a few "
         "keys, about two seconds in.",
         note="one noise ~2 s after the countdown", tail_s=6.0),
)


def load_script(path: Path | None) -> tuple[Take, ...]:
    """--script FILE overrides the built-in list. JSON: [{id, kind, label, instruction, prompt?,
    playback?: [{asset, from_s?, seconds?}], note?, tail_s?}, ...]."""
    if path is None:
        return SCRIPT
    takes = []
    for item in json.loads(path.read_text()):
        takes.append(Take(
            id=item["id"], kind=item["kind"], label=item["label"],
            instruction=item["instruction"], prompt=item.get("prompt"),
            playback=tuple(Playback(**p) for p in item.get("playback", ())),
            note=item.get("note", ""), tail_s=float(item.get("tail_s", 3.0)),
        ))
    return tuple(takes)


# ---------------------------------------------------------------------------------------------
# assets: Voxa's own TTS lines + the CC0 music, cached as 48 kHz WAVs (offline processing only)
# ---------------------------------------------------------------------------------------------

def asset_key(kind: str, payload: str) -> str:
    return hashlib.sha256(f"{kind}|{payload}|{SR}".encode()).hexdigest()[:24]


def ensure_music() -> dict:
    """Fetch the CC0 track ONCE into MUSIC_DIR and record URL + SHA-256 in music.json. Never
    touches a sound device; --prepare does this while online and rhY never has to."""
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    target = MUSIC_DIR / MUSIC_FILE
    meta_path = MUSIC_DIR / "music.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if not target.exists():
        subprocess.run(["curl", "-fsSL", "-o", str(target), MUSIC_URL], check=False)
        if not target.exists() or target.stat().st_size < 100_000:
            target.unlink(missing_ok=True)
            raise SystemExit(
                f"could not fetch the CC0 music from {MUSIC_URL}; run --prepare while online, "
                f"or drop the file in by hand: {target}")
    meta[MUSIC_FILE] = {"url": MUSIC_URL, "sha256": sha256_file(target),
                        "license": MUSIC_LICENSE, "fetched_once": True}
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def ensure_voxa_tts(text: str) -> Path:
    """Voxa's OWN TTS path (voxa/speech.py), same voice and rate as the app, cached as a 48 kHz
    WAV. No substitute voice: the echo recorded must be the echo her app will later cancel."""
    from voxa.config import DEFAULT_TTS_RATE
    from voxa.speech import SpeechService, _run_quietly

    TTS_CACHE.mkdir(parents=True, exist_ok=True)
    key = asset_key("voxa", f"{VOXA_TTS_VOICE}|{DEFAULT_TTS_RATE}|{text}")
    wav = TTS_CACHE / f"{key}.wav"
    if wav.exists():
        return wav
    service = SpeechService(allow_online_fallback=False, allow_offline_fallback=False)
    mp3, _words = _run_quietly(
        service._fetch_to_file(text, DEFAULT_TTS_RATE, VOXA_TTS_VOICE, lambda: False))
    to_wav48(Path(mp3), wav)
    Path(mp3).unlink(missing_ok=True)
    return wav


def ensure_asset(spec: Playback) -> Path:
    """The 48 kHz WAV an asset key names, sliced to (from_s, seconds) when the take says so."""
    kind, _, payload = spec.asset.partition(":")
    if kind == "voxa":
        src = ensure_voxa_tts(payload)
        out_dir = TTS_CACHE
    elif kind == "music":
        ensure_music()
        raw = MUSIC_DIR / payload
        if not raw.exists():
            raise FileNotFoundError(f"music asset {raw} missing; run --prepare first")
        src = MUSIC_DIR / f"{asset_key('music', sha256_file(raw))}.wav"
        if not src.exists():
            to_wav48(raw, src)
        out_dir = MUSIC_DIR
    else:
        raise ValueError(f"unknown asset kind {kind!r}")
    if spec.from_s == 0.0 and spec.seconds is None:
        return src
    x, rate = read_wav_mono(src)
    lo = int(spec.from_s * rate)
    hi = len(x) if spec.seconds is None else min(len(x), lo + int(spec.seconds * rate))
    dest = out_dir / f"{src.stem}_slice_{int(spec.from_s * rate)}_{hi - lo}.wav"
    if not dest.exists():
        write_wav(dest, x[lo:hi], rate)
    return dest


# ---------------------------------------------------------------------------------------------
# the session runner
# ---------------------------------------------------------------------------------------------

PREROLL_S = 0.5  # inside the capture, before the first playback (the room settles, streams run)


def countdown(seconds: int = 3, emit=print) -> None:
    """The on-screen countdown before each take (no sound: this is a terminal tool)."""
    for k in range(seconds, 0, -1):
        emit(f"  {k}...")
        time.sleep(1.0)
    emit("  record.")


def wake_sink(out_dir: Path) -> None:
    """Resume the sink for a split second before opening the monitor stream: an idle sink suspends
    and a cold monitor truncates the take (tools/anc_lab.py learned this the hard way). 30 ms of a
    100 Hz tone at 5% gain: quiet, not a test signal, and it lands in both streams anyway."""
    tone = np.sin(2.0 * np.pi * 100.0 * np.arange(int(0.03 * SR)) / SR) * 0.05
    tmp = out_dir / ".wake.wav"
    write_wav(tmp, tone.astype(np.float32), SR)
    subprocess.run(["paplay", str(tmp)], check=False)
    tmp.unlink(missing_ok=True)


def run_take(take: Take, assets: dict[str, Path], player: Player, mic_rec: Recorder,
             ref_rec: Recorder | None, sr: int, take_dir: Path, *,
             synthetic_ref: bool = False, volume: dict | None = None,
             devices: dict | None = None) -> dict:
    """One take: record mic (+ reference), play the take's playback, measure the delay, write the
    WAVs and the take JSON. The recorders are ALWAYS closed (stop()), even when playback fails."""
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    if synthetic_ref:
        ref_rec = None
    elif ref_rec is None:
        raise RuntimeError("no reference recorder and synthetic_ref is off")
    played: list[tuple[Playback, Path, np.ndarray, int]] = []
    try:
        mic_rec.start()
        if ref_rec is not None:
            ref_rec.start()
        time.sleep(PREROLL_S)
        for spec in take.playback:
            path = assets[spec.asset]
            x, _rate = read_wav_mono(path, sr)
            played.append((spec, path, x, int(time.monotonic() * sr)))
            player.play(path)
        time.sleep(take.tail_s)
    finally:
        mic = mic_rec.stop()
        ref = ref_rec.stop() if ref_rec is not None else None
    if synthetic_ref:  # the exact samples we played, placed on the clock timeline
        base = min((at for _s, _p, _x, at in played), default=0)
        ref = np.zeros(len(mic), dtype=np.float32)
        for _spec, _path, x, at in played:
            off = max(0, at - base)
            if off < len(ref):
                ref[off:off + len(x)] += x[: len(ref) - off]
    # a synthetic ref starts at the FIRST play, so its window also covers the preroll
    delay = (measure_delay(mic, ref, sr=sr, max_delay_ms=2000.0 if synthetic_ref else 500.0)
             if (played and ref is not None and len(ref)) else None)
    playback_meta = []
    for spec, path, x, at in played:
        start_ref = asset_start(ref, x, sr=sr) if (ref is not None and len(ref) > len(x)) else None
        start_mic = asset_start(mic, x, sr=sr) if len(mic) > len(x) else None
        if start_ref is None:
            start_ref = at - min(a for _s, _p, _x, a in played)
        if start_mic is None and delay is not None:
            start_mic = start_ref + delay["samples"]
        playback_meta.append({"asset": spec.asset, "file": str(path), "sha256": sha256_file(path),
                              "duration_samples": len(x),
                              "start_sample_ref": int(start_ref),
                              "start_sample_mic": int(start_mic) if start_mic is not None else None})
    take_json = {
        "id": take.id, "kind": take.kind, "expected_label": take.label,
        "instruction": take.instruction, "prompt": take.prompt, "note": take.note,
        "started": started, "sample_rate": sr,
        "ref_source": ("synthetic: the exact samples this tool played, placed on a clock timeline"
                       if synthetic_ref else "monitor: parec on the sink monitor source"),
        "playback": playback_meta, "delay": delay,
        "counts": {"mic_samples": int(len(mic)), "ref_samples": int(0 if ref is None else len(ref))},
        "devices": devices or {},
        "system_volume": volume if volume is not None else read_system_volume(default_sink()),
    }
    write_wav(take_dir / "mic.wav", mic, sr)
    if ref is not None and len(ref):
        write_wav(take_dir / "ref.wav", ref, sr)
    (take_dir / "take.json").write_text(json.dumps(take_json, indent=1))
    return take_json


def estimate_session_seconds(takes: tuple[Take, ...]) -> float:
    return sum(t.seconds() + 9.0 for t in takes)  # 9 s: enter + wake + countdown + gaps


def run_session(takes: tuple[Take, ...], out_dir: Path, player: Player, mic_rec: Recorder,
                ref_rec: Recorder | None, sr: int = SR, *, synthetic_ref: bool = False,
                assets: dict[str, Path] | None = None, wake: bool = True, countdown_s: int = 3,
                on_ready=None) -> Path:
    """The guided session: for every take announce on screen, wait for rhY, count down, capture.
    Returns the session dir. Everything opened is closed before returning (run_take's finally)."""
    if assets is None:
        assets = {spec.asset: ensure_asset(spec) for t in takes for spec in t.playback}
    out_dir.mkdir(parents=True, exist_ok=True)
    sink, mic_dev = default_sink(), default_source()
    info = devices_info(sink, mic_dev, monitor_source(sink))
    volume = read_system_volume(sink)
    if on_ready is None:
        def on_ready(take: Take, _i: int) -> None:
            input(f"\n[{take.id}] {take.instruction}\n  press ENTER when ready (Ctrl-C to stop): ")
    print(f"session: {len(takes)} takes, about {estimate_session_seconds(takes) / 60:.1f} minutes; "
          "volume as read (never set): "
          f"{volume['pct'] if volume['pct'] is not None else '?'}% muted={volume['muted']}")
    for i, take in enumerate(takes, 1):
        print(f"\n=== take {i}/{len(takes)} [{take.id}] label={take.label} ===")
        print(f"  {take.instruction}")
        if take.prompt:
            print(f"  say: {take.prompt!r}")
        if take.note:
            print(f"  timing: {take.note}")
        on_ready(take, i)
        if wake:
            wake_sink(out_dir)
        if countdown_s:
            countdown(countdown_s)
        take_dir = out_dir / safe(take.id)
        info_take = dict(info)
        info_take["recorder_device"] = mic_rec.device
        if ref_rec is not None:
            info_take["ref_device"] = ref_rec.device
        meta = run_take(take, assets, player, mic_rec, ref_rec, sr, take_dir,
                        synthetic_ref=synthetic_ref, volume=volume, devices=info_take)
        delay = meta["delay"]["ms"] if meta["delay"] else "-"
        print(f"  recorded: mic {meta['counts']['mic_samples']} samples, delay {delay} ms")
    session = {"takes": [t.id for t in takes], "sample_rate": sr, "devices": info,
               "system_volume_as_read": volume, "volume_was_never_set": True,
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "session.json").write_text(json.dumps(session, indent=1))
    return out_dir


# ---------------------------------------------------------------------------------------------
# --to-bench: session dir -> a bench suite the hearing bench can run (--suite real)
# ---------------------------------------------------------------------------------------------

EXPECT_BY_LABEL = {"user": "keep", "bargein": "yield", "backchannel": "backchannel",
                   "echo": "none", "noise": "none"}


def load_bench_clip(clip_dir: Path) -> dict:
    """The contract voxa/bench's runner.load_clip enforces: truth.json plus 16-bit mono 16 kHz
    mic.wav and ref.wav. A suite that loads under THIS function loads under the bench too."""
    truth_path = clip_dir / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"{truth_path} missing: --to-bench the session first")
    arrays = {}
    for name in ("mic", "ref"):
        with wave.open(str(clip_dir / f"{name}.wav"), "rb") as handle:
            if (handle.getnchannels() != 1 or handle.getsampwidth() != 2
                    or handle.getframerate() != BENCH_SR):
                raise ValueError(f"{clip_dir / name}.wav: needs 16-bit mono {BENCH_SR} Hz")
            arrays[name] = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return {"truth": json.loads(truth_path.read_text()), **arrays}


def to_bench(session_dir: Path, bench_root: Path, suite: str = "real") -> Path:
    """Every recorded take becomes one bench clip: mic.wav + ref.wav resampled 48k -> 16k (exact
    decimation, band-limited), truth.json carrying the bench fields (id, scenario, expect, pair)
    plus the recording metadata. A take with no reference gets a silent one: the bench needs both
    sides equal length."""
    suite_dir = bench_root / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for take_dir in sorted(p for p in session_dir.iterdir() if (p / "take.json").exists()):
        meta = json.loads((take_dir / "take.json").read_text())
        clip = suite_dir / safe(f"real-{meta['id']}")
        clip.mkdir(parents=True, exist_ok=True)
        mic, rate = read_wav_mono(take_dir / "mic.wav")
        mic16 = resample_to_rate(mic, rate, BENCH_SR)
        write_wav(clip / "mic.wav", mic16, BENCH_SR)
        if (take_dir / "ref.wav").exists():
            ref, rrate = read_wav_mono(take_dir / "ref.wav")
            ref = resample_to_rate(ref, rrate, BENCH_SR)
            if len(ref) > len(mic16):
                ref = ref[: len(mic16)]
            else:
                ref = np.pad(ref, (0, len(mic16) - len(ref)))
        else:
            ref = np.zeros(len(mic16), dtype=np.float32)
        write_wav(clip / "ref.wav", ref, BENCH_SR)
        pb = meta.get("playback") or []
        text = None
        if pb:
            kind, _, payload = str(pb[0].get("asset", "")).partition(":")
            if kind == "voxa":
                text = payload
        truth = {
            "id": f"real-{meta['id']}", "scenario": suite,
            "expect": EXPECT_BY_LABEL.get(meta["expected_label"], "none"),
            "label": meta["expected_label"], "overlap_position": None, "pair": None,
            "playback": {"text": text},
            "recorded": {k: meta.get(k) for k in
                         ("kind", "prompt", "note", "sample_rate", "ref_source", "delay",
                          "system_volume", "devices", "counts", "started", "playback")},
        }
        (clip / "truth.json").write_text(json.dumps(truth, indent=1))
        load_bench_clip(clip)  # fail loudly the moment a clip would not load under the bench
        clips.append(clip.name)
    if not clips:
        raise SystemExit(f"{session_dir}: no take dirs with take.json; nothing to convert")
    (suite_dir / "suite.json").write_text(json.dumps(
        {"suite": suite, "sr": BENCH_SR, "clips": clips, "source_session": str(session_dir)},
        indent=1))
    return suite_dir


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def print_plan(takes: tuple[Take, ...]) -> None:
    print(f"session plan: {len(takes)} takes, "
          f"about {estimate_session_seconds(takes) / 60:.1f} minutes")
    for take in takes:
        print(f"  [{take.id}] label={take.label} ~{take.seconds():.0f}s  {take.instruction}")
        if take.prompt:
            print(f"      say: {take.prompt!r}")
        if take.note:
            print(f"      timing: {take.note}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="record_session.py",
        description="guided ~10-minute recording session for real hearing data (never run this "
                    "unattended: it opens the microphone and plays sound)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"session dir (default {RECORDINGS}/<date>)")
    ap.add_argument("--script", type=Path, help="JSON session script (default: the built-in one)")
    ap.add_argument("--ref", choices=("auto", "monitor", "synthetic"), default="auto",
                    help="reference stream: sink monitor, the samples played here, or auto")
    ap.add_argument("--prepare", action="store_true",
                    help="fetch the CC0 music and render Voxa's TTS lines, then stop")
    ap.add_argument("--to-bench", type=Path, metavar="SESSION_DIR",
                    help="convert a recorded session into a bench suite")
    ap.add_argument("--bench-root", type=Path, default=DATA,
                    help=f"bench data root for --to-bench (default {DATA})")
    ap.add_argument("--suite", default="real", help="suite name for --to-bench (default real)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, open nothing")
    args = ap.parse_args(argv)

    takes = load_script(args.script)
    if args.dry_run:
        print_plan(takes)
        return 0
    if args.prepare:
        ensure_music()
        for take in takes:
            for spec in take.playback:
                if spec.asset.startswith("voxa:"):
                    ensure_voxa_tts(spec.asset.partition(":")[2])
        print("prepared: music fetched once (URL + SHA-256 in music.json), Voxa lines cached")
        return 0
    if args.to_bench is not None:
        out = to_bench(args.to_bench, args.bench_root, args.suite)
        print(f"wrote bench suite {out} (run it with: python -m voxa.bench run --suite {args.suite})")
        return 0

    # the live session: only ever run by rhY, in person
    sink, source = default_sink(), default_source()
    if not sink or not source:
        print("no default sink/source; nothing opened", file=sys.stderr)
        return 2
    monitor = monitor_source(sink)
    synthetic = args.ref == "synthetic" or (args.ref == "auto" and not monitor)
    if args.ref == "monitor" and not monitor:
        print(f"no monitor source found for sink {sink!r}; refusing (use --ref synthetic)",
              file=sys.stderr)
        return 2
    out_dir = args.out or (RECORDINGS / time.strftime("%Y-%m-%d"))
    if any(out_dir.glob("*/take.json")):
        print(f"{out_dir} already has takes; refusing to mix takes (use --out)", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = {}
    for take in takes:
        for spec in take.playback:
            assets[spec.asset] = ensure_asset(spec)
    raw = out_dir / "_raw"
    mic_rec = Recorder(source, SR, raw / "mic.raw")
    ref_rec = None if synthetic else Recorder(monitor, SR, raw / "ref.raw")
    try:
        run_session(takes, out_dir, Player(), mic_rec, ref_rec,
                    synthetic_ref=synthetic, assets=assets)
    finally:
        raw.mkdir(parents=True, exist_ok=True)
        for f in raw.glob("*.raw"):
            f.unlink(missing_ok=True)
        raw.rmdir()
    print(f"session written to {out_dir}; convert it with: --to-bench {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
