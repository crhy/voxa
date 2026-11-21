"""B1: the seeded synthetic clip renderer for the Voxa hearing bench.

Roadmap B1 (``research/voxa-hearing/voxa-hearing-research.md``): for every clip write the near-end
speech ``s``, the echo ``d``, the noise ``n``, the microphone mix, the far-end reference (Voxa's exact
TTS samples) and a truth JSON holding the user word onsets/offsets, the TTS word timings, SER, SNR,
RT60, delay, drift, the nonlinearity used, and the expected turn decision (``expect``).

Everything is offline: no microphone, no speakers, no audio device is touched. Sources are built once
into ``/home/rhy/voxa-bench-data/sources`` (override with ``VOXA_BENCH_DATA`` or ``--data-root``):

* Voxa's replies go through Voxa's own TTS path, :meth:`voxa.speech.SpeechService._fetch_to_file`
  (Edge TTS with ``WordBoundary`` timings; eSpeak NG when the network is unavailable), in Voxa's
  default voice ``en-US-AriaNeural`` at ``DEFAULT_TTS_RATE`` (``voxa/config.py``).
* The user's long requests come from LibriSpeech ``test-clean`` (openslr.org/12), word-labelled once
  with the already installed faster-whisper ``base`` model and cached.
* The user's short answers and matched-pair phrases are rendered with ``en-US-GuyNeural``, the voice
  voxatest already uses for the test "user" (``voxatest/config.py``) - deliberately NOT Voxa's voice.

Determinism: one ``numpy`` ``default_rng`` per plan and per clip, cached sources, and no wall-clock
value in any written file, so two renders with the same seed give byte-identical clips and manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..speech import SpeechService, _run_quietly, espeak_voice

SR = 16000  # voxa/audio.py REFERENCE_RATE
DEFAULT_DATA_ROOT = Path("/home/rhy/voxa-bench-data")

# Voxa's own voice and pace (voxa/config.py Settings.tts_voice / DEFAULT_TTS_RATE).
VOXA_TTS_VOICE = "en-US-AriaNeural"
VOXA_TTS_RATE = 230
# The bench "user" voice: voxatest/config.py DEFAULT_TTS_VOICE. A different voice from Voxa's, so a
# matched pair really is the same WORDS from two different speakers.
USER_TTS_VOICE = "en-US-GuyNeural"

# Levels are ratios at the microphone, as the handoff test matrix asks, not volume sliders.
SER_LEVELS = (-10.0, -5.0, 0.0, 5.0, 10.0)  # user relative to Voxa's echo
SNR_LEVELS = (30.0, 20.0, 10.0)  # user relative to noise; None means no noise at all
ECHO_RMS = 0.08  # nominal echo level in the mix; the mix is then guarded against clipping
MIX_PEAK = 0.98  # uniform gain keeps both ratios exact while staying inside int16

# Scenario -> clip count for the smoke suite (48 clips). The full suite multiplies every count by 10.
SCENARIO_COUNTS = {
    "fst": 6,
    "nst": 6,
    "dt_onset": 3,
    "dt_first_syllable": 2,
    "dt_middle": 3,
    "dt_last_word": 2,
    "dt_echo_tail": 2,
    "short": 6,
    "pair": 6,
    "quiet": 3,
    "pause": 3,
    "cold": 3,
    "path_change": 3,
}
SUITE_MULTIPLIER = {"smoke": 1, "full": 10}
TTS_POOL_SIZE = {"smoke": 24, "full": 200}

# Voxa alone, including the words that must never become a command of their own (self_command_rate).
FST_TEXTS = (
    "Say stop to end this reply.",
    "Cancel the timer.",
    "You can say Voxa to get my attention.",
    "I am offline right now.",
    "The weather in San Diego is sunny with a high of seventy two degrees.",
    "Opening LibreOffice.",
)

# The short answers of the handoff matrix, spoken by the user over Voxa's playback.
SHORT_ANSWERS = ("yes", "no", "blue", "that one", "stop", "no, the other one")
# Whole commands the user says over Voxa, taken from tools/anc_lab.py:39 (COMMANDS). They used to be
# played back through a speaker in the lab; here they are rendered, because nothing may be recorded or
# played in the bench.
COMMAND_PHRASES = (
    "Voxa stop music", "stop dictation", "Voxa what time is it", "volume down",
    "volume up", "Voxa pause", "Voxa resume", "stop listening",
    "Voxa read a story", "Voxa what is the weather", "Voxa write a note", "Voxa stop",
)
# What each answer means for turn-taking: a command she must obey by stopping her own stream,
# a back-channel she must record WITHOUT interrupting her turn, or a new request to keep.
BACKCHANNEL_WORDS = {"yes", "no", "ok", "okay", "thanks", "thank you"}
COMMAND_WORDS = {"stop", "cancel", "pause", "shut up"}


def expected_decision(user_text: str | None) -> str:
    """The turn decision a clip expects: keep, yield, backchannel or none."""
    if user_text is None:
        return "none"
    phrase = user_text.lower().strip(".,!?").strip()
    if phrase in COMMAND_WORDS:
        return "yield"
    if phrase in BACKCHANNEL_WORDS:
        return "backchannel"
    return "keep"

# Matched pairs: the same words spoken by Voxa alone and by the user over Voxa (pair_accuracy).
PAIRS = (
    {"id": "stop", "voxa_text": "Stop.", "user_text": "stop"},
    {"id": "phrase", "voxa_text": "The library opens at nine in the morning.",
     "user_text": "The library opens at nine in the morning."},
    {"id": "colour", "voxa_text": "Blue.", "user_text": "blue"},
)

REPLY_TEXTS = (
    "Okay, I stopped the music.",
    "Playing Dirty Old Town by the Pogues.",
    "The timer is set for five minutes.",
    "I saved the file as meeting notes.",
    "Reminding you at three to call Bob.",
    "The nearest pharmacy is on Fifth Avenue.",
    "I cannot hear you while I am speaking.",
    "Your to do list has three items.",
    "Updating Spaced Linux now.",
    "The stock price of Tesla is up two point one percent.",
    "Pressing Install in Spaced Bazaar.",
    "Dictation started, say stop dictation to finish.",
    "The weather in San Diego is sunny with a high of seventy two degrees.",
    "I opened the map for walking directions to the library.",
    "The library opens at nine in the morning.",
    "Blue is the colour you chose.",
    "Reading the poem aloud now.",
    "I will summarise our conversation.",
    "Break down painting the hallway into four steps.",
    "The calendar has one event this week.",
    "Cleaning up the text in the window in front.",
    "Maximizing yourself on the desktop.",
    "Volume down.",
    "What time is it?",
)

_OBJECTS = ("timer", "music", "note", "calendar", "window", "player", "dictionary", "file")
_VALUES = ("ready", "closed", "set for three minutes", "playing", "saved", "paused", "open", "quiet")


@dataclass
class Source:
    """One recorded utterance: samples plus its word onsets/offsets in source time."""

    key: str
    text: str
    voice: str
    engine: str
    samples: np.ndarray
    words: tuple[dict, ...]


def safe_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def tts_pool(need: int) -> list[str]:
    """The distinct Voxa replies a suite draws from, in a fixed order (curated first, then generated)."""
    pool = list(FST_TEXTS) + list(REPLY_TEXTS) + [p["voxa_text"] for p in PAIRS]
    unique: list[str] = []
    for text in pool:
        if text not in unique:
            unique.append(text)
    n = 0
    while len(unique) < need:
        obj = _OBJECTS[n % len(_OBJECTS)]
        value = _VALUES[(n // len(_OBJECTS)) % len(_VALUES)]
        unique.append(f"The {obj} is {value}, and I will tell you when it changes.")
        n += 1
    return unique[:need]


def write_wav(path: Path, samples: np.ndarray, rate: int = SR) -> None:
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
        if rate != SR:
            raise ValueError(f"expected {SR} Hz, found {rate} Hz")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def decode_audio(raw: bytes, container: str, rate: int = SR) -> np.ndarray:
    """Decode container bytes to float32 mono at SR with ffmpeg (already a system dependency)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", container, "-i", "-", "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"],
        input=raw, capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"ffmpeg could not decode {container}: {detail}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def edge_voice(text: str, voice: str, rate: int, tmp_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Voxa's own natural TTS path: Edge TTS with word boundaries, through voxa.speech."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    service = SpeechService(allow_online_fallback=False, allow_offline_fallback=False)
    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_dir)  # speech.py writes its MP3 through tempfile; keep it off /tmp
    try:
        path, ticks = _run_quietly(service._fetch_to_file(text, rate, voice, lambda: False))
        try:
            raw = Path(path).read_bytes()
        finally:
            Path(path).unlink(missing_ok=True)
    finally:
        tempfile.tempdir = previous
    samples = decode_audio(raw, "mp3")
    words = [
        {"word": str(word).strip(), "onset_s": float(start), "offset_s": float(start + duration)}
        for word, start, duration in ticks
        if str(word).strip() and duration > 0.0
    ]
    return samples, words


def espeak_words(text: str, voice: str, rate: int) -> tuple[np.ndarray, list[dict]]:
    """Offline fallback: eSpeak NG per word, so the word timings are exact by construction."""
    engine_voice = espeak_voice(voice)
    chunks: list[np.ndarray] = []
    words: list[dict] = []
    position = 0.0
    for token in re.findall(r"\S+", text):
        proc = subprocess.run(
            ["espeak-ng", "--stdout", "-v", engine_voice, "-s", str(rate), token],
            capture_output=True, check=True,
        )
        samples = decode_audio(proc.stdout, "wav")
        chunks.append(samples)
        words.append({"word": token.strip(",.;:!?") or token,
                      "onset_s": position, "offset_s": position + len(samples) / SR})
        position += len(samples) / SR
    return np.concatenate(chunks), words


def voice_source(text: str, voice: str, rate: int, data_root: Path, kind: str) -> Source:
    """One TTS source, rendered once and cached under ``sources/<kind>/``."""
    cache = data_root / "sources" / kind
    key = f"{safe_key(text)}-{safe_key(voice)}"
    wav, meta = cache / f"{key}.wav", cache / f"{key}.json"
    if wav.exists() and meta.exists():
        payload = json.loads(meta.read_text())
        return Source(key=key, text=payload["text"], voice=payload["voice"], engine=payload["engine"],
                      samples=read_wav(wav), words=tuple(payload["words"]))
    cache.mkdir(parents=True, exist_ok=True)
    engine = "edge_tts"
    try:
        samples, words = edge_voice(text, voice, rate, cache / "_tmp")
    except Exception as exc:  # noqa: BLE001 - offline or service failure falls back to eSpeak NG
        print(f"natural voice unavailable ({str(exc)[:120]}); using eSpeak NG for {text!r}", file=sys.stderr)
        samples, words = espeak_words(text, voice, rate)
        engine = "espeak-ng"
    write_wav(wav, samples)
    meta.write_text(json.dumps(
        {"text": text, "voice": voice, "rate": rate, "engine": engine, "sr": SR, "words": words}, indent=2) + "\n")
    # Re-read the 16-bit cache so a source built in this run is bit-for-bit what a later run reads.
    return Source(key=key, text=text, voice=voice, engine=engine, samples=read_wav(wav), words=tuple(words))


_WHISPER = None

# Bump when the labelling rules change, so an old cache is rebuilt instead of silently reused.
LABEL_CACHE_VERSION = 2


def whisper_model():
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel

        _WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
    return _WHISPER


def label_user_clip(samples: np.ndarray, key: str, cache_dir: Path) -> list[dict]:
    """Word onsets/offsets for a user clip: faster-whisper once per clip, then cached."""
    meta = cache_dir / f"{key}.json"
    if meta.exists():
        payload = json.loads(meta.read_text())
        if payload.get("version") == LABEL_CACHE_VERSION:
            return payload["words"]
    words: list[dict] = []
    segments, _info = whisper_model().transcribe(
        samples, word_timestamps=True, beam_size=1, language="en", vad_filter=False)
    for segment in segments:
        for word in segment.words:
            token = word.word.strip()
            if token and word.end > word.start:
                # A label whose end equals its start carries no timing information; keeping it would
                # put a zero-length word in the truth file.
                words.append({"word": token, "onset_s": float(word.start), "offset_s": float(word.end)})
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"key": key, "version": LABEL_CACHE_VERSION, "words": words}, indent=2) + "\n")
    return words


def librispeech_corpus(data_root: Path) -> list[Path]:
    root = data_root / "sources" / "librispeech" / "LibriSpeech" / "test-clean"
    if not root.is_dir():
        raise SystemExit(
            "LibriSpeech test-clean is missing under the bench data root. Download it once:\n"
            f"  curl -o {data_root}/librispeech.tar https://www.openslr.org/resources/12/test_clean.tar.gz\n"
            f"  mkdir -p {data_root}/sources/librispeech && tar -xz -C {data_root}/sources/librispeech "
            f"-f {data_root}/librispeech.tar")
    return sorted(root.rglob("*.flac"))


def probe_seconds(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def pick_librispeech(corpus: list[Path], need: int, seed: int) -> list[Path]:
    """Deterministic pick of ``need`` clips between 2 and 8 seconds, spread over the corpus."""
    chosen: list[Path] = []
    span = len(corpus) / max(1, need)
    for slot in range(need):
        start = int(slot * span)
        picked = None
        for index in range(start, min(len(corpus), start + 60)):
            if 2.0 <= probe_seconds(corpus[index]) <= 8.0:
                picked = corpus[index]
                break
        chosen.append(picked if picked is not None else corpus[start % len(corpus)])
    return chosen


def librispeech_source(path: Path, data_root: Path) -> Source:
    """Decode one LibriSpeech clip (cached) and return it with its cached word labels."""
    key = path.stem
    cache = data_root / "sources" / "user"
    wav = cache / f"{key}.wav"
    if wav.exists():
        samples = read_wav(wav)
    else:
        samples = decode_audio(path.read_bytes(), "flac")
        cache.mkdir(parents=True, exist_ok=True)
        write_wav(wav, samples)
        samples = read_wav(wav)  # the 16-bit cache is the single source of truth
    words = label_user_clip(samples, key, cache / "labels")
    text = " ".join(word["word"] for word in words)
    return Source(key=key, text=text, voice="librispeech", engine="librispeech", samples=samples,
                  words=tuple(words))


def make_rir(rng: np.random.Generator, rt60_s: float) -> np.ndarray:
    """Synthetic room impulse response: a dominant direct path, then reflections that decay to -60 dB.

    The first reflection cannot arrive before the direct path, and it stays quieter than it, which is
    what makes the injected delay measurable to within a sample from the delivered files.
    """
    n = max(2, int(round(rt60_s * SR)))
    impulse = np.zeros(n, dtype=np.float64)
    impulse[0] = 1.0
    first = min(n - 1, int(round(rng.uniform(0.004, 0.012) * SR)))
    t = (np.arange(n) - first) / SR
    envelope = 10.0 ** (-3.0 * np.clip(t, 0.0, None) / rt60_s)
    reflections = rng.standard_normal(n)
    peak = float(np.max(np.abs(reflections[first:]))) or 1.0
    impulse[first:] = (0.25 * reflections * envelope / peak)[first:]
    return impulse


def fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    size = 1
    while size < len(a) + len(b) - 1:
        size <<= 1
    return np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size))[: len(a) + len(b) - 1]


def apply_nonlinearity(x: np.ndarray, spec: dict | None) -> np.ndarray:
    """Speaker nonlinearity, applied to the played signal before the room (the reference stays exact)."""
    if spec is None:
        return x
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return x
    y = np.asarray(x, dtype=np.float64) / peak
    if spec["kind"] == "hard_clip":
        y = np.clip(y, -spec["param"], spec["param"])
    else:
        y = np.tanh(spec["param"] * y) / np.tanh(spec["param"])
    return (y * peak).astype(np.float32)


def resample_clock(x: np.ndarray, out_len: int, delay_samples: int, step_at: int | None,
                   step_samples: int, drift_ppm: float) -> np.ndarray:
    """Resample the ECHO only: bulk delay, an optional mid-clip delay step, and clock drift.

    The far-end reference keeps its own clock, which is what makes delay, a step and drift hard
    for a canceller that assumes one fixed offset.
    """
    positions = np.arange(out_len, dtype=np.float64)
    delay = np.full(out_len, float(delay_samples))
    if step_at is not None:
        delay[step_at:] += float(step_samples)
    rate = 1.0 + (drift_ppm * 1e-6) * (positions / max(1, out_len))
    source = (positions - delay) / rate
    grid = np.arange(len(x), dtype=np.float64)
    return np.interp(source, grid, x, left=0.0, right=0.0).astype(np.float32)


def noise_signal(rng: np.random.Generator, n: int) -> np.ndarray:
    """Room/fan/typing noise: broadband with a low-frequency tilt, like the ANC test signals."""
    if n <= 0:
        return np.zeros(0, np.float32)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spectrum *= 1.0 / np.sqrt(1.0 + (freqs / 1500.0) ** 2)
    x = np.fft.irfft(spectrum, n)
    peak = float(np.max(np.abs(x)))
    return (x / peak).astype(np.float32) if peak > 0 else x.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def ratio_db(signal: float, reference: float) -> float | None:
    if signal <= 1e-9 or reference <= 1e-9:
        return None
    return 20.0 * float(np.log10(signal / reference))


def place_user(user: Source, onset_s: float, pause_s: float | None,
               rng: np.random.Generator) -> tuple[np.ndarray, list[dict], float]:
    """Place the user's audio in clip time, optionally split around a mid-sentence pause."""
    words = list(user.words)
    if pause_s is None or len(words) < 3:
        head, tail, split_at = user.samples, np.zeros(0, np.float32), len(words)
    else:
        split_at = int(rng.integers(max(1, len(words) // 3), max(2, 2 * len(words) // 3)))
        boundary = words[split_at]["onset_s"]
        cut = int(round(boundary * SR))
        head, tail = user.samples[:cut], user.samples[cut:]
    parts = [head]
    if tail.size:
        parts.append(np.zeros(int(round(pause_s * SR)), np.float32))
        parts.append(tail)
    body = np.concatenate(parts)
    start = int(round(onset_s * SR))
    out = np.zeros(start + len(body) + SR // 4, np.float32)
    out[start:start + len(body)] = body
    shifted: list[dict] = []
    for index, word in enumerate(words):
        shift = pause_s if (pause_s is not None and index >= split_at and tail.size) else 0.0
        shifted.append({
            "word": word["word"],
            "onset_s": onset_s + word["onset_s"] + shift,
            "offset_s": onset_s + word["offset_s"] + shift,
        })
    return out, shifted, (start + len(body)) / SR


def user_onset(item: dict, tts_source: Source | None, rng: np.random.Generator) -> float:
    """Where the user starts, in clip time, for this clip's overlap position."""
    if tts_source is None:
        return float(item["jitter_s"])
    ticks = tts_source.words
    end = len(tts_source.samples) / SR
    position = item["position"]
    if position == "onset":
        base = 0.0
    elif position == "first_syllable":
        base = (ticks[0]["onset_s"] + 0.06) if ticks else 0.05
    elif position == "middle":
        base = float(item.get("dt_fraction", 0.5)) * end
    elif position == "last_word":
        base = (ticks[-1]["onset_s"] if ticks else 0.75 * end)
    elif position == "cold":
        base = max(0.0, end - float(rng.uniform(0.0, 0.2)))
    elif position == "echo_tail":
        base = end + float(rng.uniform(0.05, 0.4))
    else:
        base = float(rng.uniform(0.1, 0.5)) * end
    return max(0.0, base + float(item["jitter_s"]))


def impairments(rng: np.random.Generator, *, force_step: bool = False, force_drift: bool = False,
                force_clean: bool = False) -> dict:
    """Echo path and impairments, per the roadmap: RT60 0.2-0.8 s, delay 0-300 ms, drift +-200 ppm,
    a 20-80 ms mid-clip step, and a nonlinearity on 80 % of clips.

    ``force_clean`` gives one clip per suite with no drift, no step and no nonlinearity: the delay
    injected into that clip is what a correlating canceller must recover to the sample.
    """
    drift = float(rng.uniform(-200.0, 200.0))
    if not force_drift and rng.random() < 0.4:
        drift = 0.0
    step_ms = float(rng.uniform(20.0, 80.0)) if (force_step or rng.random() < 0.25) else None
    if rng.random() < 0.8:
        if rng.random() < 0.5:
            nonlinearity = {"kind": "hard_clip", "param": float(rng.uniform(0.5, 0.9))}
        else:
            nonlinearity = {"kind": "tanh", "param": float(rng.uniform(1.0, 4.0))}
    else:
        nonlinearity = None
    if force_clean:
        drift, step_ms, nonlinearity = 0.0, None, None
    return {
        "rt60_s": float(rng.uniform(0.2, 0.8)),
        "delay_ms": float(rng.uniform(0.0, 300.0)),
        "delay_step_ms": step_ms,
        "drift_ppm": drift,
        "nonlinearity": nonlinearity,
    }


def plan(suite: str, seed: int) -> list[dict]:
    """The seeded clip plan: scenario, sources, overlap position, levels and impairments.

    No audio is touched here, so the scenario coverage can be checked without rendering.
    """
    if suite not in SUITE_MULTIPLIER:
        raise SystemExit(f"unknown suite {suite!r}; expected one of {sorted(SUITE_MULTIPLIER)}")
    multiplier = SUITE_MULTIPLIER[suite]
    pool = tts_pool(TTS_POOL_SIZE[suite])
    rng = np.random.default_rng(seed)
    items: list[dict] = []
    for scenario, count in SCENARIO_COUNTS.items():
        for index in range(count * multiplier):
            item = {
                "id": f"{suite}-{len(items):04d}-{scenario}-{index:02d}",
                "scenario": scenario,
                "position": None,
                "jitter_s": float(rng.uniform(-0.05, 0.05)),
                "expect": "keep",
                "tts_text": None,
                "user_text": None,
                "user_source": None,
                "ser_db": None,
                "snr_db": None,
                "pause_s": None,
                "pair_id": None,
                "pair_side": None,
                "dt_fraction": None,
            }
            if scenario == "fst":
                item["tts_text"] = FST_TEXTS[index % len(FST_TEXTS)]
                item["expect"] = "none"
            elif scenario == "pair":
                pair = PAIRS[(index // 2) % len(PAIRS)]
                item["pair_id"] = pair["id"]
                item["pair_side"] = "voxa" if index % 2 == 0 else "user"
                item["tts_text"] = pair["voxa_text"]
                if item["pair_side"] == "voxa":
                    item["expect"] = "none"
                else:
                    item["user_text"] = pair["user_text"]
                    item["expect"] = expected_decision(pair["user_text"])
            elif scenario == "short":
                item["tts_text"] = pool[int(rng.integers(len(pool)))]
                item["user_text"] = SHORT_ANSWERS[index % len(SHORT_ANSWERS)]
                item["expect"] = expected_decision(item["user_text"])
            elif scenario == "quiet":
                item["tts_text"] = pool[int(rng.integers(len(pool)))]
                item["user_source"] = "librispeech"
                item["ser_db"] = -10.0
            elif scenario == "pause":
                item["tts_text"] = pool[int(rng.integers(len(pool)))]
                item["user_source"] = "librispeech"
                item["pause_s"] = float(rng.uniform(0.4, 1.5))
            elif scenario in {"cold", "path_change"}:
                item["tts_text"] = pool[int(rng.integers(len(pool)))]
                item["user_text"] = COMMAND_PHRASES[index % len(COMMAND_PHRASES)]
                item["position"] = "cold" if scenario == "cold" else "middle"
                if scenario == "path_change":
                    item["dt_fraction"] = (0.2, 0.5, 0.8)[index % 3]
            elif scenario == "nst":
                item["user_source"] = "librispeech"
                item["jitter_s"] = float(rng.uniform(0.0, 0.4))
            else:  # the DT positions
                item["tts_text"] = pool[int(rng.integers(len(pool)))]
                item["user_source"] = "librispeech"
                item["position"] = scenario.split("_", 1)[1]
                if scenario == "dt_middle":
                    item["dt_fraction"] = (0.2, 0.5, 0.8)[index % 3]
            if item["ser_db"] is None and scenario not in {"fst", "nst"}:
                item["ser_db"] = float(rng.choice(SER_LEVELS))
            item["snr_db"] = None if rng.random() < 0.25 else float(rng.choice(SNR_LEVELS))
            item.update(impairments(rng, force_step=scenario == "path_change",
                                    force_drift=scenario == "path_change",
                                    force_clean=(scenario == "dt_onset" and index == 0)))
            items.append(item)
    return items


def build_clip(item: dict, tts_source: Source | None, user_source: Source | None,
               rng: np.random.Generator) -> dict:
    """Build one clip's arrays and its truth record."""
    rt60_s = item["rt60_s"]
    delay_samples = int(round(item["delay_ms"] / 1000.0 * SR))
    step_samples = int(round(item["delay_step_ms"] / 1000.0 * SR)) if item["delay_step_ms"] else 0

    playback_onset = 0.0
    playback_end = (playback_onset + len(tts_source.samples) / SR) if tts_source else 0.0
    onset = user_onset(item, tts_source, rng)
    user_arr, user_words, user_end = (None, [], 0.0)
    if user_source is not None:
        user_arr, user_words, user_end = place_user(user_source, onset, item["pause_s"], rng)

    clip_seconds = max(playback_end, user_end) + rt60_s + 0.3
    n = max(SR // 10, int(round(clip_seconds * SR)))

    ref = np.zeros(n, np.float32)
    if tts_source is not None:
        start = int(round(playback_onset * SR))
        body = tts_source.samples[: n - start]
        ref[start:start + len(body)] = body

    played = apply_nonlinearity(ref, item["nonlinearity"])
    echo = fft_convolve(played, make_rir(rng, rt60_s))
    step_at = int(round(rng.uniform(0.2, 0.8) * n)) if step_samples else None
    d = resample_clock(echo, n, delay_samples, step_at, step_samples, item["drift_ppm"])

    s = np.zeros(n, np.float32)
    if user_arr is not None:
        limit = min(len(user_arr), n)
        s[:limit] = user_arr[:limit]

    noise = noise_signal(rng, n) if item["snr_db"] is not None else np.zeros(n, np.float32)

    # Levels: the echo sits at a nominal level, the user is set by SER against the echo, and the
    # noise by SNR against the user (against the echo when there is no user to reference).
    if rms(d) > 1e-9:
        d = d * (ECHO_RMS / rms(d))
    user_target = ECHO_RMS * 10.0 ** (item["ser_db"] / 20.0) if item["ser_db"] is not None else ECHO_RMS
    if rms(s) > 1e-9:
        s = s * (user_target / rms(s))
    noise_reference = rms(s) if rms(s) > 1e-9 else rms(d)
    if item["snr_db"] is not None and noise_reference > 1e-9 and rms(noise) > 1e-9:
        noise = noise * ((noise_reference / 10.0 ** (item["snr_db"] / 20.0)) / rms(noise))

    mix = s + d + noise
    peak = max(float(np.max(np.abs(x))) if x.size else 0.0 for x in (s, d, noise, mix))
    gain = 1.0
    if peak > MIX_PEAK:
        gain = MIX_PEAK / peak
        s, d, noise, mix = s * gain, d * gain, noise * gain, mix * gain

    truth = {
        "id": item["id"],
        "scenario": item["scenario"],
        "expect": item["expect"],
        "overlap_position": item["position"],
        "sr": SR,
        "frames": n,
        "seconds": n / SR,
        "playback": None if tts_source is None else {
            "text": tts_source.text,
            "voice": tts_source.voice,
            "engine": tts_source.engine,
            "onset_s": playback_onset,
            "offset_s": playback_end,
            "echo_tail_end_s": playback_end + rt60_s,
            "words": _shift_words(tts_source, playback_onset),
        },
        "user": None if user_source is None else {
            "text": user_source.text,
            "source": user_source.key,
            "voice": user_source.voice,
            "onset_s": onset,
            "offset_s": user_end,
            "pause_s": item["pause_s"],
            "words": user_words,
        },
        "echo": None if tts_source is None else {
            "rt60_s": rt60_s,
            "delay_ms": delay_samples * 1000.0 / SR,
            "delay_samples": delay_samples,
            "delay_step_ms": item["delay_step_ms"],
            # TASK-3 BUG FIX (f130), see tasks/notes-f130-voxa-hear-t3-delay-drift.md: the path
            # change lands at a RANDOM time (line 642) but the truth never recorded WHERE, so a
            # path_change clip could not reconstruct its own echo path, and a delay estimate made
            # after the step could not be judged against the truth. The audio is unchanged; only
            # the truth record gains the two fields the renderer already knew.
            "step_at_s": (step_at / SR) if step_at else None,
            "clip_seconds": n / SR,
            "drift_ppm": item["drift_ppm"],
            "nonlinearity": item["nonlinearity"],
        },
        "levels": {
            "ser_db_target": item["ser_db"],
            "snr_db_target": item["snr_db"],
            "snr_reference": "user" if user_source is not None else "echo",
            "noise": item["snr_db"] is not None,
            "echo_rms": ECHO_RMS,
            "mix_gain_db": 20.0 * float(np.log10(gain)) if gain != 1.0 else 0.0,
        },
        "pair": {"id": item["pair_id"], "side": item["pair_side"]} if item["pair_id"] else None,
    }
    arrays = {"s": s, "d": d, "n": noise, "mic": mix, "ref": ref}
    return {"truth": truth, "arrays": arrays}


def _shift_words(source: Source, onset_s: float) -> list[dict]:
    return [
        {"word": word["word"], "onset_s": onset_s + word["onset_s"], "offset_s": onset_s + word["offset_s"]}
        for word in source.words
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render_clip(item: dict, tts_source: Source | None, user_source: Source | None,
                rng: np.random.Generator, out_dir: Path) -> dict:
    """Render one clip: five WAVs plus truth.json, then hash them for the manifest."""
    built = build_clip(item, tts_source, user_source, rng)
    truth, arrays = built["truth"], built["arrays"]
    clip_dir = out_dir / item["id"]
    clip_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name in ("s", "d", "n", "mic", "ref"):
        path = clip_dir / f"{name}.wav"
        write_wav(path, arrays[name])
        written[f"{name}.wav"] = sha256_file(path)
    # Realised levels are measured from the delivered 16-bit files, not from the float arrays.
    s_done, d_done, n_done = read_wav(clip_dir / "s.wav"), read_wav(clip_dir / "d.wav"), read_wav(clip_dir / "n.wav")
    truth["levels"]["ser_db_realised"] = ratio_db(rms(s_done), rms(d_done))
    snr_reference = rms(s_done) if rms(s_done) > 1e-9 else rms(d_done)
    truth["levels"]["snr_db_realised"] = ratio_db(snr_reference, rms(n_done))
    truth["levels"]["echo_rms_realised"] = rms(d_done)
    truth["levels"]["user_rms_realised"] = rms(s_done)
    path = clip_dir / "truth.json"
    path.write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n")
    written["truth.json"] = sha256_file(path)
    return {"id": item["id"], "scenario": item["scenario"], "files": written}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voxa.bench render",
        description="render a seeded synthetic clip set for the Voxa hearing bench")
    parser.add_argument("--suite", choices=sorted(SUITE_MULTIPLIER), default="smoke",
                        help="smoke (48 clips) or full (480 clips)")
    parser.add_argument("--seed", type=int, default=0, help="numpy Generator seed; same seed gives the same bytes")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest path (default .voxa-bench/manifest-<suite>.json)")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("VOXA_BENCH_DATA", str(DEFAULT_DATA_ROOT))),
                        help="bench data root; clips go to <data-root>/<suite>/")
    args = parser.parse_args(argv)

    data_root: Path = args.data_root
    out_dir = data_root / args.suite
    items = plan(args.suite, args.seed)
    texts = sorted({item["tts_text"] for item in items if item["tts_text"]})
    print(f"sources: {len(texts)} Voxa replies ({VOXA_TTS_VOICE})")
    tts_sources = {text: voice_source(text, VOXA_TTS_VOICE, VOXA_TTS_RATE, data_root, "tts") for text in texts}
    phrases = sorted({item["user_text"] for item in items if item["user_text"]})
    phrase_sources = {text: voice_source(text, USER_TTS_VOICE, VOXA_TTS_RATE, data_root, "user_phrases")
                      for text in phrases}
    needed = sum(1 for item in items if item["user_source"] == "librispeech")
    corpus = librispeech_corpus(data_root) if needed else []
    picked = pick_librispeech(corpus, needed, args.seed)
    librispeech_sources = [librispeech_source(path, data_root) for path in picked]
    print(f"sources: {len(picked)} LibriSpeech clips, {len(phrases)} user phrases ({USER_TTS_VOICE})")

    librispeech_index = 0
    manifest_clips: list[dict] = []
    for index, item in enumerate(items):
        clip_rng = np.random.default_rng([args.seed, index])
        tts_source = tts_sources.get(item["tts_text"]) if item["tts_text"] else None
        if item["user_text"]:
            user_source = phrase_sources[item["user_text"]]
        elif item["user_source"] == "librispeech":
            user_source = librispeech_sources[librispeech_index]
            librispeech_index += 1
        else:
            user_source = None
        manifest_clips.append(render_clip(item, tts_source, user_source, clip_rng, out_dir))
        if (index + 1) % 10 == 0 or index + 1 == len(items):
            print(f"rendered {index + 1}/{len(items)}")

    manifest_path: Path = args.manifest or Path(f".voxa-bench/manifest-{args.suite}.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"suite": args.suite, "seed": args.seed, "sr": SR, "count": len(manifest_clips),
                "clips": manifest_clips}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(manifest_clips)} clips under {out_dir} and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
