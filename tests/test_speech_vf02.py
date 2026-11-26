"""vf02 #72: the whole speaking chain can be rendered to a WAV with no speakers, and
spoken replies never carry emoji or Markdown."""

from __future__ import annotations

import os

import pytest

from voxa.speech import _audio_sink_spec, _audio_sink_tail, _speakable_text


# --- step 2d: strip emoji + Markdown at the speech boundary -------------------

def test_speakable_strips_emoji() -> None:
    assert _speakable_text("See you soon 😊🎉") == "See you soon"
    assert _speakable_text("2024 is a common year, so there are 365 days ✅") == (
        "2024 is a common year, so there are 365 days"
    )


def test_speakable_strips_markdown() -> None:
    spoken = _speakable_text("**bold** _em_ `code` #heading | table -bullet")
    for mark in ("*", "_", "`", "#", "|"):
        assert mark not in spoken
    assert "bold" in spoken and "heading" in spoken


def test_speakable_keeps_link_text_drops_url() -> None:
    assert _speakable_text("Paris [the capital](https://example.com/paris)") == "Paris the capital"


def test_speakable_plain_reply_unchanged() -> None:
    assert _speakable_text("One hour is 60 minutes.") == "One hour is 60 minutes."


# --- step 2c: VOXA_AUDIO_SINK routes the sink to a WAV file -------------------

def test_sink_defaults_to_autoaudiosink(monkeypatch) -> None:
    monkeypatch.delenv("VOXA_AUDIO_SINK", raising=False)
    assert _audio_sink_spec() == ""
    assert _audio_sink_tail() == "autoaudiosink"


def test_sink_file_env_selects_wav_sink(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_AUDIO_SINK", "file:/tmp/voxa-off.wav")
    tail = _audio_sink_tail()
    assert "wavenc" in tail and "filesink" in tail and "/tmp/voxa-off.wav" in tail


def test_audio_sink_env_switch_writes_nonsilent_wav(tmp_path) -> None:
    """Push a 1.5 s tone through the real sink chain and assert the WAV is >1 s and not silent."""
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)

    out = str(tmp_path / "sink.wav")
    os.environ["VOXA_AUDIO_SINK"] = f"file:{out}"
    try:
        pipeline = Gst.parse_launch(
            "appsrc name=src format=bytes block=true is-live=false ! "
            "audioconvert ! audioresample ! volume name=voxa_volume ! " + _audio_sink_tail()
        )
        assert pipeline is not None
        appsrc = pipeline.get_by_name("src")
        assert appsrc is not None

        # 1.5 s of 16 kHz mono s16le, a 440 Hz tone, so the WAV is audible (not silence).
        import math
        import struct

        rate, seconds = 16000, 1500 // 10
        frames = bytearray()
        for i in range(int(1.5 * rate)):
            sample = int(12000 * math.sin(2 * math.pi * 440 * i / rate))
            frames += struct.pack("<h", sample)
        buf = Gst.Buffer.new_allocate(None, len(frames), None)
        buf.fill(0, bytes(frames))
        appsrc.emit("push-buffer", buf)
        appsrc.emit("end-of-stream")

        assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
        bus = pipeline.get_bus()
        msg = bus.poll(Gst.MessageType.EOS | Gst.MessageType.ERROR, Gst.CLOCK_TIME_NONE)
        pipeline.set_state(Gst.State.NULL)
    finally:
        del os.environ["VOXA_AUDIO_SINK"]

    assert msg is not None and msg.type == Gst.MessageType.EOS

    import wave

    import numpy as np

    assert os.path.getsize(out) > 0
    with wave.open(out) as w:
        n, sr = w.getnframes(), w.getframerate()
        assert n / sr > 1.0  # more than a second of audio
        data = np.frombuffer(w.readframes(n), dtype="<i2")
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    assert rms > 500.0  # not silent
