"""Offline proof of Voxa's speaking chain writing to a WAV (task vf02 step 2c).

No microphone, no speakers, no network: VOXA_AUDIO_SINK=file:<path> makes the real
speech pipeline render to a WAV instead of the user's default output. These tests
assert the env switch and that the WAV sink chain produces a real, non-silent file.
"""

from __future__ import annotations

import array
import math
import struct
import wave

import pytest

from voxa.speech import _file_sink_path


def test_file_sink_path_reads_the_env_switch(monkeypatch, tmp_path) -> None:
    target = str(tmp_path / "out.wav")
    monkeypatch.setenv("VOXA_AUDIO_SINK", f"file:{target}")
    assert _file_sink_path() == target


def test_file_sink_path_empty_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("VOXA_AUDIO_SINK", raising=False)
    assert _file_sink_path() == ""
    monkeypatch.setenv("VOXA_AUDIO_SINK", "autoaudiosink")
    assert _file_sink_path() == ""


def _sine_pcm(seconds: float, freq: float, rate: int = 48000) -> bytes:
    frames = int(seconds * rate)
    peak = 12000
    values = array.array(
        "h",
        [int(peak * math.sin(2 * math.pi * freq * i / rate)) for i in range(frames)],
    )
    return values.tobytes()


def test_wav_sink_chain_writes_a_non_silent_second_of_audio(tmp_path) -> None:
    # The tail of the real speech chain (decodebin -> audioconvert -> audioresample
    # -> wavenc -> filesink) must write a WAV over one second long and not silent.
    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)

    encoded = tmp_path / "tone.wav"
    out = tmp_path / "reply.wav"
    pcm = _sine_pcm(1.4, 440.0)

    # A real encoded file to feed the tail chain (the same decodebin -> ... ->
    # wavenc -> filesink that VOXA_AUDIO_SINK selects). A Python-written WAV keeps
    # the fixture deterministic; the pipeline under test is Voxa's own.
    with wave.open(str(encoded), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(pcm)

    decode = Gst.parse_launch(
        f'filesrc location="{encoded}" ! decodebin ! '
        "audioconvert ! audioresample ! wavenc ! filesink name=voxa_sink"
    )
    decode.get_by_name("voxa_sink").set_property("location", str(out))
    assert decode.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
    bus = decode.get_bus()
    eos = None
    for _ in range(200):  # bounded: 0.5s polls up to 100s
        eos = bus.poll(Gst.MessageType.EOS, 500_000_000)
        if eos is not None:
            break
    assert eos is not None

    assert out.is_file() and out.stat().st_size > 44
    with wave.open(str(out), "rb") as wav:
        frames = wav.getnframes()
        seconds = frames / wav.getframerate()
        samples = struct.unpack(f"<{frames}h", wav.readframes(frames))
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    assert seconds > 1.0
    assert rms > 1000  # not silent


def test_start_natural_stream_uses_file_sink(monkeypatch, tmp_path) -> None:
    # The real streaming entry point must select the wavenc/filesink tail when the
    # env switch is set, and autoaudiosink otherwise.
    pytest.importorskip("gi")
    import voxa.speech as speech_mod

    # Load the real GStreamer bindings first so _start_natural_stream's own
    # _ensure_gstreamer() is a no-op and parse_launch can be stubbed.
    speech_mod._ensure_gstreamer()

    service = speech_mod.SpeechService()
    launched: list[str] = []

    target = tmp_path / "reply.wav"
    monkeypatch.setenv("VOXA_AUDIO_SINK", f"file:{target}")

    class _Sink:
        def set_property(self, name, value):  # noqa: ANN001, D102
            assert name == "location" and value == str(target)

    class _Pipeline:
        def get_by_name(self, name):  # noqa: ANN001, D102
            if name == "source":
                return object()
            return _Sink()

        def set_state(self, _state):  # noqa: ANN001, D102
            return speech_mod.Gst.StateChangeReturn.SUCCESS

        def get_bus(self):  # noqa: D102
            class _Bus:
                def add_signal_watch(self): ...  # noqa: D102, E704

                def connect(self, *_a): ...  # noqa: D102, E704

            return _Bus()

    monkeypatch.setattr(speech_mod.Gst, "parse_launch", lambda s: launched.append(s) or _Pipeline())
    monkeypatch.setattr(service, "_natural_worker", lambda *a: None)

    import threading

    service._start_natural_stream("hello", 180, "en-US-AriaNeural", threading.Event())

    assert launched and "wavenc ! filesink" in launched[0]
    assert "autoaudiosink" not in launched[0]

    launched.clear()
    monkeypatch.delenv("VOXA_AUDIO_SINK")
    service._start_natural_stream("hello", 180, "en-US-AriaNeural", threading.Event())
    assert launched and "autoaudiosink" in launched[0]