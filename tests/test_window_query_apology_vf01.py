"""Issue #72: a heard request must never end silent.

The real window, with fake audio/speech/Whisper and a fake AI client, drives a
hands-free turn through the broken shapes: a query thread that dies on a
non-Ollama exception (a dropped backend connection), and a backend that is
stopped before the request.  Before the fix the action log only ever held the
pre-call record (ok null, empty speech, ms 0) and the user heard nothing at
all; every request must now produce a final record and, hands-free, an
apology out loud.
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

from voxa.ollama import OllamaError  # noqa: E402
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


class FakeSpeech:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text, rate, voice, on_started=None, on_done=None, on_error=None, on_words=None) -> None:
        self.spoken.append(text)

    def position(self) -> float:
        return 0.0

    def stop(self) -> None:
        pass


class FakeClient:
    def __init__(self, text: str = "The capital of France is Paris.", error: Exception | None = None,
                 delay: float = 0.0) -> None:
        self.text, self.error, self.delay = text, error, delay

    def generate_stream(self, *, model, prompt, cancel_event, on_chunk, messages=None, **_kwargs) -> str:
        if self.error is not None:
            raise self.error
        if self.delay:
            time.sleep(self.delay)
        return self.text


class OfflineClient(FakeClient):
    """Like a real backend client while ``ollama serve`` is down."""

    def __init__(self, text: str, server) -> None:
        super().__init__(text)
        self.server = server
        self.probes = 0

    def is_model_loaded(self, model: str):
        self.probes += 1
        if self.server.started:
            return True
        return None

    def start_called(self) -> bool:
        return self.server.started


class StubServer:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> bool:
        self.started = True
        return True


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.windowqueryapologytest")
    app.register(None)
    return app


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                     ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.setenv("HOME", str(tmp_path))
    # never append request records to a developer's real action log
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


def _final_records(window) -> list[dict]:
    return [r for r in window.action_log.read() if r["route"] == "model" and r["ok"] is not None]


def test_a_crashed_query_thread_still_apologises_and_records(window, monkeypatch) -> None:
    # A dropped backend connection is not an OllamaError: before the fix it
    # killed the query thread silently, with no record and no apology.
    monkeypatch.setattr(window, "_ai_client", lambda: FakeClient(error=ConnectionResetError("Connection reset by peer")))
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    window._on_conversation_prompt("what is the capital of france")

    assert _wait_for(lambda: _state(window) is AssistantState.ERROR)
    assert AI_APOLOGY in window.speech.spoken
    finals = _final_records(window)
    assert len(finals) == 1
    assert finals[0]["ok"] is False
    assert "ConnectionResetError" in finals[0]["detail"]
    assert finals[0]["speech"] == ""
    window._recover_assistant(window.assistant.token())


def test_a_finished_request_writes_its_final_record(window, monkeypatch) -> None:
    monkeypatch.setattr(window, "_ai_client", lambda: FakeClient("The capital of France is Paris.", delay=0.01))
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    window._on_conversation_prompt("what is the capital of france")

    assert _wait_for(lambda: _state(window) is AssistantState.SPEAKING)
    finals = _final_records(window)
    assert len(finals) == 1
    assert finals[0]["ok"] is True
    assert "Paris" in finals[0]["speech"]
    assert finals[0]["ms"] > 0
    assert finals[0]["timings"]


def test_a_stopped_managed_backend_is_started_before_the_request(window, monkeypatch) -> None:
    server = StubServer()
    client = OfflineClient("The capital of France is Paris.", server)
    monkeypatch.setattr(window, "_ai_client", lambda: client)
    monkeypatch.setattr(window, "ai_server", server)
    window.settings.ai_backend = "ollama"
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    window._on_conversation_prompt("what is the capital of france")

    assert _wait_for(lambda: _state(window) is AssistantState.SPEAKING)
    assert server.started
    assert client.probes >= 1
    finals = _final_records(window)
    assert len(finals) == 1 and finals[0]["ok"] is True


def _state(win) -> AssistantState:
    return win.assistant_model.state
