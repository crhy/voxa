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
