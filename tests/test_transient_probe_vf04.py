"""vf04 (#72): a backend that drops a probe mid-handshake must not crash the query thread.

The vf02 gate run died because http.client.RemoteDisconnected escaped from
OllamaClient.is_model_loaded (urllib does not wrap errors raised while reading
the response status) and was caught by the query worker's generic
``except Exception`` -> "the AI query thread crashed".  Here the opener drops
the connection, the real probe returns its "unknown" value, and the worker
still answers the heard question.
"""

from __future__ import annotations

import http.client
import time

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")
Adw.init()

import voxa.window as window_module  # noqa: E402
from voxa.ollama import OllamaClient  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402
from voxa.window import MainWindow  # noqa: E402


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


class FakeSpeech:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text, rate, voice, on_started=None, on_done=None, on_error=None, on_words=None) -> None:
        self.spoken.append(text)

    def position(self) -> float:
        return 0.0

    def stop(self) -> None:
        pass


def _dropped(request, timeout):
    raise http.client.RemoteDisconnected("Remote end closed connection without response")


class DroppedProbeClient:
    """A real Ollama probe over a socket that closes before the status line,
    on a client that then still answers the request."""

    def __init__(self, text: str) -> None:
        self.text = text

    def is_model_loaded(self, model: str):
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr("voxa.ollama.open_url", _dropped)
            return OllamaClient().is_model_loaded(model)

    def generate_stream(self, *, model, prompt, cancel_event, on_chunk, messages=None, **_kwargs) -> str:
        return self.text


class StubServer:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> bool:
        self.started = True
        return True


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.transientprobevftest")
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
    # The window installs the Voxa stylesheet on the real display; that CSS then
    # sized every later GTK test (vf04 left it installed and the suite noticed).
    monkeypatch.setattr(window_module, "install_styles", lambda *args: None)
    win = MainWindow(application)
    win.audio = FakeAudio()
    win.speech = FakeSpeech()
    win.whisper._model = object()
    win.devices = [object()]
    win.present()
    _pump()
    yield win
    win._closing = True
    win.assistant.go_offline()
    win.destroy()
    _pump(20)


def _state(win) -> AssistantState:
    return win.assistant_model.state


def test_dropped_probe_still_answers_and_never_reports_a_crash(window, monkeypatch, caplog) -> None:
    client = DroppedProbeClient("The capital of France is Paris.")
    monkeypatch.setattr(window, "_ai_client", lambda: client)
    monkeypatch.setattr(window, "ai_server", StubServer())
    window.settings.ai_backend = "strata"
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    window._on_conversation_prompt("what is the capital of france")

    assert _wait_for(lambda: _state(window) is AssistantState.SPEAKING)
    assert "the AI query thread crashed" not in caplog.text
    finals = [r for r in window.action_log.read() if r["route"] == "model" and r["ok"] is not None]
    assert len(finals) == 1
    assert finals[0]["ok"] is True
    assert "Paris" in finals[0]["speech"]
