#!/usr/bin/env python3
"""VoxaTest live-run volume pre-check (issue 130, task v07).

Every live run must call this BEFORE any case. Rules it enforces (rhY 2026-11-24):
- read and record the default sink volume/mute and the default source volume;
- unmute the sink if muted (never mute, never lower, never touch other apps);
- calibrate by measurement: play VoxaTest's own ~3 s test phrase through the
  speakers while the microphone records (Voxa must be closed), and require the
  mic capture of the phrase to peak at >= -20 dBFS and to sit >= 15 dB above the
  room noise floor (3 s of silence measured first);
- if below target, RAISE the sink volume in 5% steps (to the stream's own max)
  and re-measure, up to 100%. Never lower any volume.
- if 100% is still too quiet, exit 1: the run must not start at an inaudible level.

Exit 0 + a machine-readable summary on stdout. Run with the VoxaTest venv python
(so numpy is available).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "voxatest" / "data" / "samples" / "hello-sample.wav"
PHRASE = "Hello, this is the Voxa volume check."
RATE = 16000
TARGET_PEAK_DBFS = -20.0
TARGET_OVER_NOISE_DB = 15.0


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def pactl_get(*args: str) -> str:
    return run(["pactl", f"get-{args[0]}", *args[1:]])


def parse_volume(text: str) -> tuple[int, int]:
    """Return (percent, muted) from pactl get-*-volume/-mute style output."""
    pct = 0
    mute = 0
    for token in text.replace("\n", " ").split():
        if token.endswith("%"):
            pct = max(pct, int(token[:-1]))
    if "yes" in text:
        mute = 1
    return pct, mute


def _write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(rate)
        dst.writeframes(data.tobytes())
    return path


def _decode(raw: bytes, container: str) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", container, "-i", "-", "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
        input=raw, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def make_phrase(tmp: Path) -> Path:
    """VoxaTest's own ~3 s test phrase, spoken at full stream scale (its own 100%): Edge TTS
    if it answers, else the spoken sample we already ship, twice. Peak-normalized to 0.95
    so the speaker hears VoxaTest's stream at 100% and only the SINK gain is under test."""
    try:
        import asyncio

        import edge_tts

        mp3 = tmp / "phrase.mp3"

        async def synth() -> None:
            await edge_tts.Communicate(PHRASE, "en-US-GuyNeural").save(str(mp3))

        asyncio.run(synth())
        if mp3.stat().st_size > 1000:
            samples = _decode(mp3.read_bytes(), "mp3")
            samples *= min(1.0, 0.95 / max(float(np.max(np.abs(samples))), 1e-9))
            return _write_wav(tmp / "phrase.wav", samples)
    except Exception:  # noqa: BLE001 - offline or no network: the spoken sample we already ship
        pass
    with wave.open(str(SAMPLE), "rb") as src:
        frames, params = src.readframes(src.getnframes()), src.getparams()
    gap = b"\x00" * (params.framerate // 5 * params.sampwidth * params.nchannels)
    wav = tmp / "phrase.wav"
    with wave.open(str(wav), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
        dst.writeframes(gap)
        dst.writeframes(frames)
    samples = _decode(wav.read_bytes(), "wav")
    samples *= min(1.0, 0.95 / max(float(np.max(np.abs(samples))), 1e-9))
    return _write_wav(wav, samples)


class MicCapture:
    """16 kHz mono S16 capture from the default source via GStreamer (as Voxa does)."""

    def __init__(self) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self.gst = Gst
        self.chunks: list[np.ndarray] = []

    def record(self, seconds: float, after_start=None) -> np.ndarray:
        self.chunks = []
        gst = self.gst
        src = gst.ElementFactory.make("pulsesrc")
        if src is None:
            src = gst.ElementFactory.make("autoaudiosrc")
        conv, res, caps, sink = (gst.ElementFactory.make(n) for n in
                                 ("audioconvert", "audioresample", "capsfilter", "appsink"))
        caps.set_property("caps", gst.Caps.from_string("audio/x-raw,format=S16LE,channels=1,rate=16000"))
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)

        def on_sample(pad):
            sample = pad.emit("pull-sample")
            buf = sample.get_buffer()
            data = np.frombuffer(buf.extract_dup(0, buf.get_size()), dtype=np.int16)
            self.chunks.append(data.astype(np.float32) / 32768.0)
            return gst.FlowReturn.OK

        sink.connect("new-sample", on_sample)
        pipe = gst.Pipeline.new("volumecheck")
        for el in (src, conv, res, caps, sink):
            pipe.add(el)
        src.link(conv)
        conv.link(res)
        res.link(caps)
        caps.link(sink)
        if pipe.set_state(gst.State.PLAYING) == gst.StateChangeReturn.FAILURE:
            raise RuntimeError("could not open the microphone")
        if after_start is not None:
            time.sleep(0.3)
            after_start()
        time.sleep(seconds + 0.3)
        pipe.set_state(gst.State.NULL)
        if not self.chunks:
            raise RuntimeError("microphone produced no audio")
        return np.concatenate(self.chunks)


def dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    return 20.0 * math.log10(max(float(np.max(np.abs(samples))), 1e-9))


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(samples ** 2))), 1e-9))


def gated_rms_dbfs(samples: np.ndarray, noise_rms: float) -> float:
    """RMS of only the frames that are clearly the phrase (not the room), so the trailing
    silence does not drag the level down; the noise gate sits 9 dB above the noise floor."""
    gate = noise_rms * 3.0
    voiced = samples[np.abs(samples) > gate]
    if voiced.size == 0:
        return -120.0
    return rms_dbfs(voiced)


def main() -> int:
    out: dict = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    sink_pct, sink_mute = parse_volume(pactl_get("sink-volume", "@DEFAULT_SINK@") + pactl_get("sink-mute", "@DEFAULT_SINK@"))
    src_pct, _ = parse_volume(pactl_get("source-volume", "@DEFAULT_SOURCE@"))
    out["initial"] = {"sink_volume": sink_pct, "sink_mute": sink_mute, "source_volume": src_pct}

    if sink_mute:
        run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
        out["unmuted_sink"] = True

    capture = MicCapture()
    if shutil.which("paplay") is None:
        print(json.dumps({**out, "fail": "paplay missing"}))
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        phrase = make_phrase(Path(tmp))

        steps = []
        while True:
            noise = capture.record(3.0)
            thirds = [noise[i * RATE // 3:(i + 1) * RATE // 3] for i in range(3)]
            noise_db = min(rms_dbfs(t) for t in thirds)  # the room floor, not a transient
            proc: list = []

            def start_play() -> None:
                proc.append(subprocess.Popen(["paplay", str(phrase)],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

            phrase_audio = capture.record(5.0, after_start=start_play)
            if proc:
                proc[0].wait()
            peak = dbfs(phrase_audio)
            rms = gated_rms_dbfs(phrase_audio, 10.0 ** (noise_db / 20.0))
            sink_pct_now, _ = parse_volume(pactl_get("sink-volume", "@DEFAULT_SINK@"))
            steps.append({"sink_volume": sink_pct_now, "noise_rms_dbfs": round(noise_db, 1),
                          "phrase_peak_dbfs": round(peak, 1), "phrase_rms_dbfs": round(rms, 1)})
            ok = peak >= TARGET_PEAK_DBFS and rms >= noise_db + TARGET_OVER_NOISE_DB
            if ok:
                out["final"] = {"sink_volume": sink_pct_now,
                                "source_volume": parse_volume(pactl_get("source-volume", "@DEFAULT_SOURCE@"))[0]}
                out["noise_floor_dbfs"] = round(noise_db, 1)
                out["phrase_peak_dbfs"] = round(peak, 1)
                out["phrase_rms_dbfs"] = round(rms, 1)
                out["steps"] = steps
                out["pass"] = True
                print(json.dumps(out))
                return 0
            sink_pct_now, _ = parse_volume(pactl_get("sink-volume", "@DEFAULT_SINK@"))
            if sink_pct_now >= 100:
                out["steps"] = steps
                out["pass"] = False
                out["fail"] = ("default sink at 100% still below target: raise it or fix the mic by hand; "
                               "never lowered below the initial level")
                print(json.dumps(out))
                return 1
            run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+5%"])
            time.sleep(max(0.0, started - time.monotonic() + 0.2))


if __name__ == "__main__":
    sys.exit(main())
