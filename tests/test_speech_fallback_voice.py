from __future__ import annotations

import subprocess
import threading

import voxa.speech as speech_mod
from voxa.speech import SpeechService, espeak_voice, google_voice_params


def test_google_voice_params_for_known_voices():
    assert google_voice_params("en-IE-EmilyNeural") == ("en", "ie")
    assert google_voice_params("en-US-GuyNeural") == ("en", "com")
    assert google_voice_params("fr-FR-DeniseNeural") == ("fr", "com")
    assert google_voice_params("es-MX-DaliaNeural") == ("es", "com.mx")


def test_google_voice_params_unknown_and_empty():
    assert google_voice_params("") == ("en", "com")
    assert google_voice_params("xx-YY-ZedNeural") == ("xx", "com")


def test_espeak_voice_keeps_accent_and_gender():
    assert espeak_voice("en-IE-EmilyNeural") == "en-gb+f3"
    assert espeak_voice("en-US-GuyNeural") == "en-us"
    assert espeak_voice("fr-FR-DeniseNeural") == "fr-fr+f3"
    assert espeak_voice("es-MX-DaliaNeural") == "es-419+f3"


def test_espeak_voice_unknown_and_empty():
    assert espeak_voice("") == "en-us"
    assert espeak_voice("xx-YY-ZedNeural") == "en-us"


def test_offline_fallback_uses_character_voice(monkeypatch):
    service = SpeechService()
    service._voice = "en-IE-EmilyNeural"
    cancel_event = threading.Event()
    service.cancel_event = cancel_event

    recorded = []

    def fake_run(argv, *args, **kwargs):
        recorded.append(argv)
        raise subprocess.CalledProcessError(1, "espeak-ng", stderr=b"")

    monkeypatch.setattr(speech_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(speech_mod, "_ensure_gstreamer", lambda: None)
    monkeypatch.setattr(speech_mod, "_ui_once", lambda *args: None)

    service._synthesize_offline("hello", 180, cancel_event, "Edge HTTP 403")

    assert recorded
    argv = recorded[0]
    assert argv[0] == "espeak-ng"
    assert "-v" in argv
    assert argv[argv.index("-v") + 1] == "en-gb+f3"


def _pcm16_wav(amplitude: int, n: int = 4000) -> bytes:
    import io
    import math
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(
            b"".join(
                int(amplitude * math.sin(2 * math.pi * 40 * i / 16000)).to_bytes(2, "little", signed=True)
                for i in range(n)
            )
        )
    return buf.getvalue()


def _wav_peak(data: bytes) -> int:
    import wave

    with wave.open(__import__("io").BytesIO(data), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return max(int.from_bytes(frames[i:i + 2], "little", signed=True) for i in range(0, len(frames), 2))


def test_cap_wav_peaks_loud_input_hits_ceiling():
    from voxa.speech import CEILING_DBFS, cap_wav_peaks

    loud = _pcm16_wav(30000)
    capped = cap_wav_peaks(loud)
    peak = _wav_peak(capped)
    ceiling = 32767.0 * 10.0 ** (CEILING_DBFS / 20.0)
    assert peak <= ceiling + 1


def test_cap_wav_peaks_quiet_input_untouched():
    from voxa.speech import cap_wav_peaks

    quiet = _pcm16_wav(1000)
    assert cap_wav_peaks(quiet) == quiet


def test_synthesize_offline_caps_espeak_output(monkeypatch, tmp_path):
    import subprocess
    import threading

    from voxa.speech import SpeechService, cap_wav_peaks

    service = SpeechService()
    cancel_event = threading.Event()
    service.cancel_event = cancel_event

    loud = _pcm16_wav(32000)
    result = subprocess.CompletedProcess(["espeak-ng"], 0, stdout=loud, stderr=b"")
    monkeypatch.setattr(speech_mod.subprocess, "run", lambda *a, **k: result)
    monkeypatch.setattr(speech_mod, "_ensure_gstreamer", lambda: None)
    monkeypatch.setattr(speech_mod, "_ui_once", lambda fn, *args: fn(*args))
    played = []
    monkeypatch.setattr(service, "_play_file", lambda path, event=None: played.append(path))

    service._synthesize_offline("hello", 180, cancel_event, "edge failed")

    assert played
    with open(played[0], "rb") as handle:
        written = handle.read()
    assert _wav_peak(written) <= _wav_peak(cap_wav_peaks(loud)) + 1
