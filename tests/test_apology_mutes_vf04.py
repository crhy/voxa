"""vf04 (#72): the spoken apology must go through the normal reply path.

_speak_apology used to call ``speech.speak`` directly: the conversation was
not muted and own_speech never heard the sentence, so in an open follow-up
window Voxa heard "Sorry, I could not get an answer" as a new request and
apologised forever.  Now the apology is spoken like any reply: muted while
it plays, held by own_speech, unmuted when it ends.
"""

from __future__ import annotations

import time

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")
Adw.init()

from voxa.ui.state import AssistantState  # noqa: E402
from voxa.window import AI_APOLOGY, MainWindow  # noqa: E402


def _pump(iterations: int = 60) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


def _wait_for(condition, timeout: float = 5.0) -> bool:
    ctx = GLib.MainContext.default()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        while ctx.iteration(False):
            pass
        if condition():
            return True
        time.sleep(0.01)
    return False


class FakeAudio:
    def __init__(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, device_id, on_audio, on_level, on_error) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False


class CapturingSpeech:
    """Keeps the reply's lifecycle callbacks so the test can end the speech."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.on_started = None
        self.on_done = None
        self.on_error = None

    def speak(self, text, rate, voice, on_started=None, on_done=None, on_error=None, on_words=None) -> None:
        self.spoken.append(text)
        self.on_started = on_started
        self.on_done = on_done
        self.on_error = on_error

    def position(self) -> float:
        return 0.0

    def stop(self) -> None:
        pass


class FailingClient:
    def generate_stream(self, *, model, prompt, cancel_event, on_chunk, messages=None, **_kwargs) -> str:
        raise ConnectionResetError("Connection reset by peer")


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.apologymutetest")
    app.register(None)
    return app


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                     ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VOXA_ACTION_LOG", str(tmp_path / "state" / "actions.jsonl"))
    for name in (
        "_load_whisper",
        "_load_wake_whisper",
        "_detect_hardware_async",
        "_refresh_ollama_models",
        "_refresh_devices",
        "_start_ai_server_async",
        "_restart_ai_server_async",
        "_stop_ai_server_async",
    ):
        monkeypatch.setattr(MainWindow, name, lambda *args, **kwargs: None)
    win = MainWindow(application)
    win.audio = FakeAudio()
    win.speech = CapturingSpeech()
    win.whisper._model = object()
    win.devices = [object()]
    win.present()
    _pump()
    yield win
    win._closing = True
    win.assistant.go_offline()
    win.destroy()
    _pump(20)


def test_apology_mutes_the_conversation_and_holds_it_as_own_speech(window, monkeypatch) -> None:
    monkeypatch.setattr(window, "_ai_client", lambda: FailingClient())
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    window._on_conversation_prompt("what is the capital of france")

    # The request fails; the apology is what the user hears.
    assert _wait_for(lambda: AI_APOLOGY in window.speech.spoken)
    assert window.speech.on_done is not None, "the apology was not spoken through the reply path"

    # While it plays: the microphone is muted against echo and own_speech holds
    # the apology, so its own words can never become a new request.
    assert window.conversation is not None
    assert window.conversation.muted is True
    assert window.own_speech.speaking is True
    assert "could" in window.own_speech.texts[-1] and "answer" in window.own_speech.texts[-1]

    # When the speech ends the conversation opens up again.
    window.speech.on_done()
    assert _wait_for(lambda: window.conversation is not None and window.conversation.muted is False)
    assert window.own_speech.speaking is False
