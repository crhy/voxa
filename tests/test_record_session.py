"""r01 (crhy/voxa#72): the guided session recorder, tested with FAKE devices ONLY.

A mocked recorder plays the signal it "heard" back 40 ms late plus noise (the task's contract):
alignment has to land within 1 ms, every take's JSON must be complete, and --to-bench must produce
a suite that loads exactly under the bench's own clip contract. No test here opens a microphone,
plays a sound or touches the sound server.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import tools.record_session as rs

SR = rs.SR
SHIFT40 = int(round(0.040 * SR))  # the fake playback->mic delay: 40 ms
ONE_MS = SR // 1000  # 48 samples at 48 kHz


def tone(seconds: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def voiceish(seconds: float, seed: int = 7) -> np.ndarray:
    """A speech-shaped broadband stand-in for TTS/music (a pure tone has no correlation peak
    away from zero lag: sine autocorrelation sidelobes drown it)."""
    rng = np.random.default_rng(seed)
    return (0.3 * rng.standard_normal(int(seconds * SR))).astype(np.float32)


class FakePlayer:
    """Records what it was asked to play and WHEN (sample clock), without touching a sink."""

    def __init__(self):
        self.played: list[tuple[Path, np.ndarray, int]] = []

    def play(self, path: Path) -> None:
        x, rate = rs.read_wav_mono(path, SR)
        self.played.append((path, x, int(time.monotonic() * SR)))


class FakeRecorder:
    """Returns the signals this player played, shifted by `shift` samples (the fake device's
    latency), plus optional noise. stop() is always called; the test asserts it was."""

    def __init__(self, player: FakePlayer, shift: int = 0, noise: float = 0.0,
                 device: str = "fake") -> None:
        self.player = player
        self.shift = shift
        self.noise = noise
        self.device = device
        self.t0 = 0
        self.n0 = 0
        self.stopped = False

    def start(self) -> None:
        self.t0 = int(time.monotonic() * SR)
        self.n0 = len(self.player.played)
        self.stopped = False

    def stop(self) -> np.ndarray:
        self.stopped = True
        now = int(time.monotonic() * SR)
        entries = [(at - self.t0, x) for _p, x, at in self.player.played[self.n0:]]
        length = (max([now - self.t0] + [off + len(x) + self.shift for off, x in entries])
                  + SR // 8)
        out = np.zeros(length, dtype=np.float64)
        for off, x in entries:
            at = max(0, off + self.shift)
            out[at:at + len(x)] += x
        if self.noise:
            rng = np.random.default_rng(1234)
            out += rng.standard_normal(length) * self.noise
        return out.astype(np.float32)


def fake_assets(tmp_path: Path):
    """Monkeypatched asset builder: TTS and music become tones, no network, no eSpeak, no ffmpeg."""

    def ensure(spec: rs.Playback) -> Path:
        path = tmp_path / (rs.safe(spec.asset) + ".wav")
        if not path.exists():
            rs.write_wav(path, voiceish(2.0), SR)
        return path

    return ensure


def mini_script(tmp_path: Path) -> tuple[rs.Take, ...]:
    return (
        rs.Take("room-01", "room", "user", "read this", prompt="Remind me of the milk."),
        rs.Take("voxa-01", "voxa", "echo", "do not speak",
                playback=(rs.Playback("voxa:Hello! I am Voxa."),), tail_s=0.05),
        rs.Take("barge-01", "bargein", "bargein", "interrupt", prompt="stop",
                playback=(rs.Playback("voxa:Opening your calendar."),), tail_s=0.05),
        rs.Take("noise-01", "noise", "noise", "cough once", tail_s=0.05),
    )


def make_session(tmp_path, monkeypatch):
    """A whole mini session on fake devices: the exact run the task describes for the tests."""
    monkeypatch.setattr(rs, "ensure_asset", fake_assets(tmp_path))
    out = tmp_path / "session"
    player = FakePlayer()
    mic = FakeRecorder(player, shift=SHIFT40, noise=0.02, device="fake-mic")
    ref = FakeRecorder(player, shift=0, device="fake-ref")
    takes = mini_script(tmp_path)
    rs.run_session(takes, out, player, mic, ref, countdown_s=0, wake=False,
                   on_ready=lambda *_: None)
    return out, takes


def test_record_session_fake_delay_measures_40ms_within_one_ms(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ensure_asset", fake_assets(tmp_path))
    player = FakePlayer()
    mic = FakeRecorder(player, shift=SHIFT40, noise=0.02)
    ref = FakeRecorder(player, shift=0)
    path = tmp_path / "played.wav"
    rs.write_wav(path, voiceish(1.0), SR)
    mic.start()
    ref.start()
    player.play(path)
    time.sleep(0.1)
    mic_x, ref_x = mic.stop(), ref.stop()
    delay = rs.measure_delay(mic_x, ref_x, sr=SR)
    assert delay is not None
    assert abs(delay["samples"] - SHIFT40) <= ONE_MS, delay
    # the exact samples land at the same place in both streams, 40 ms apart in the mic
    at_ref = rs.asset_start(ref_x, voiceish(1.0), SR)
    at_mic = rs.asset_start(mic_x, voiceish(1.0), SR)
    assert at_mic - at_ref - SHIFT40 <= ONE_MS  # both streams share one clock; < 1 ms skew


def test_record_session_runs_with_fake_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ensure_asset", fake_assets(tmp_path))
    out = tmp_path / "s"
    player = FakePlayer()
    mic = FakeRecorder(player, shift=SHIFT40, noise=0.02, device="fake-mic")
    ref = FakeRecorder(player, shift=0, device="fake-ref")
    takes = mini_script(tmp_path)
    rs.run_session(takes, out, player, mic, ref, countdown_s=0, wake=False,
                   on_ready=lambda *_: None)
    assert mic.stopped and ref.stopped  # close everything the tool opened
    metas = {}
    for take in takes:
        d = out / rs.safe(take.id)
        metas[take.id] = json.loads((d / "take.json").read_text())
        mic_x, rate = rs.read_wav_mono(d / "mic.wav", SR)
        assert rate == SR
        assert len(mic_x) > 0
    room = metas["room-01"]
    assert room["expected_label"] == "user"
    assert room["prompt"] == "Remind me of the milk."
    assert room["playback"] == []
    assert room["delay"] is None  # nothing played: no delay to measure
    voxa = metas["voxa-01"]
    assert voxa["delay"] is not None and abs(voxa["delay"]["samples"] - SHIFT40) <= ONE_MS
    assert voxa["sample_rate"] == SR
    assert voxa["devices"]["default_sink"] == ""  # tests never reach the real sound server
    assert "get-sink-volume" in voxa["system_volume"]["tool"]
    for take_id in ("voxa-01", "barge-01"):
        pb = metas[take_id]["playback"]
        assert len(pb) == 1
        assert Path(pb[0]["file"]).exists()
        assert len(pb[0]["sha256"]) == 64
        assert pb[0]["start_sample_ref"] is not None and pb[0]["start_sample_mic"] is not None
    session = json.loads((out / "session.json").read_text())
    assert session["sample_rate"] == SR
    assert session["volume_was_never_set"] is True


def test_record_session_synthetic_reference_is_the_played_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ensure_asset", fake_assets(tmp_path))
    out = tmp_path / "s"
    player = FakePlayer()
    mic = FakeRecorder(player, shift=SHIFT40, noise=0.02, device="fake-mic")
    takes = mini_script(tmp_path)
    rs.run_session(takes, out, player, mic, None, synthetic_ref=True, countdown_s=0,
                   wake=False, on_ready=lambda *_: None)
    assert mic.stopped
    meta = json.loads((out / "voxa-01" / "take.json").read_text())
    assert meta["ref_source"].startswith("synthetic")
    assert meta["delay"] is not None


def test_record_session_to_bench_produces_loadable_suite(tmp_path, monkeypatch):
    out, _takes = make_session(tmp_path, monkeypatch)
    bench = tmp_path / "bench"
    suite_dir = rs.to_bench(out, bench, "real")
    assert suite_dir == bench / "real"
    manifest = json.loads((suite_dir / "suite.json").read_text())
    assert manifest["sr"] == rs.BENCH_SR
    assert set(manifest["clips"]) == {"real-room-01", "real-voxa-01", "real-barge-01",
                                      "real-noise-01"}
    expect = {"real-room-01": "keep", "real-voxa-01": "none", "real-barge-01": "yield",
              "real-noise-01": "none"}
    for name in manifest["clips"]:
        loaded = rs.load_bench_clip(suite_dir / name)  # the bench contract itself
        truth = loaded["truth"]
        assert truth["id"] == name
        assert truth["scenario"] == "real"
        assert truth["expect"] == expect[name]
        assert truth["recorded"]["system_volume"]["tool"].startswith("pactl get-sink-volume")
        assert len(loaded["mic"]) == len(loaded["ref"])
        assert 2000 < len(loaded["mic"]) < 100000
    assert json.loads((suite_dir / "real-barge-01" / "truth.json").read_text())["playback"]["text"]\
        == "Opening your calendar."


def test_record_session_resamples_48k_to_16k_exactly():
    x = tone(1.0)
    y = rs.resample_to_rate(x, SR, rs.BENCH_SR)
    assert len(y) == pytest.approx(len(x) / 3.0, abs=2)


def test_record_session_bench_script_shape():
    script = rs.SCRIPT
    assert 12 <= len(script) <= 20
    labels = {t.label for t in script}
    assert {"user", "echo", "noise", "bargein", "backchannel"} <= labels
    assert all(t.label in rs.EXPECT_BY_LABEL for t in script)
    total = rs.estimate_session_seconds(script)
    assert 7 * 60 <= total <= 13 * 60  # ~10 minutes of guided session
    kinds = {t.kind for t in script}
    assert {"bargein", "backchannel", "music", "far", "noise"} <= kinds


def test_record_session_never_sets_the_volume(rhY_rule_guard):
    assert rhY_rule_guard  # every pactl call in the module is a read (get/list), never a set


@pytest.fixture
def rhY_rule_guard():
    src = Path(rs.__file__).read_text()
    for call in src.split('_pactl("')[1:]:
        command = call.split('"')[0]
        assert command.split()[0].startswith(("get", "list")), command
    assert "set-sink-volume\", " not in src and "set-sink-mute\", " not in src
    return True


def test_record_session_help_headless():
    proc = subprocess.run([sys.executable, str(Path(rs.__file__)), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    assert "record_session.py" in proc.stdout
