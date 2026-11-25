"""The scripted listening path in the real window (issue #130).

These drive the real MainWindow (fake audio/speech/Whisper, like test_window_offline):
the scripted start is exactly the Active button path, and listening.json says the truth
after both the scripted change and the button change. Skipped where PyGObject is not
installed (the release gate's venv), which is why the pure parts live in
test_voxa_listening_state.py and test_voxa_application_actions.py.
"""

from __future__ import annotations

import json

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")
Adw.init()

from voxa.audio import AudioDevice  # noqa: E402
from voxa.listening import default_path as listening_default_path  # noqa: E402
from voxa.listening import read_state  # noqa: E402
from voxa.window import MainWindow  # noqa: E402


def _pump(iterations: int = 60) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


class FakeAudio:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, device_id, on_audio, on_level, on_error) -> None:
        self.calls.append("start")
        self._active = True

    def start_device_watch(self) -> None:
        pass

    def stop(self) -> None:
        self.calls.append("stop")
        self._active = False


class FakeSpeech:
    def stop(self) -> None:
        pass


@pytest.fixture
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.listeningtest")
    app.register(None)
    return app


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    for var, sub in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_STATE_HOME", "state"),
    ):
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.delenv("VOXA_LISTENING_STATE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
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
    win.whisper._model = object()  # WhisperService.ready
    win.devices = [AudioDevice(id="mic", name="Microphone")] if "id" in AudioDevice.__dataclass_fields__ else [object()]
    win.present()
    _pump()
    yield win
    win._closing = True
    win.assistant.go_offline()
    win.destroy()
    _pump(20)


def _state() -> dict | None:
    return read_state(listening_default_path())


def test_scripted_start_listens_like_the_active_button_and_reports_it(window) -> None:
    window.scripted_start_listening()
    _pump()
    assert window.conversation_active and window.conversation is not None
    assert window.audio.is_active
    state = _state()
    assert state is not None and state["active"] is True and state["reason"] == ""


def test_scripted_stop_takes_listening_down_and_reports_it(window) -> None:
    window.scripted_start_listening()
    _pump()
    window.scripted_stop_listening()
    _pump()
    assert not window.conversation_active and window.conversation is None
    assert not window.audio.is_active
    state = _state()
    assert state is not None and state["active"] is False


def test_the_button_path_writes_the_same_file(window) -> None:
    window.assistant.activate()  # exactly what pressing Active does
    _pump()
    assert window.conversation_active
    assert _state() is not None and _state()["active"] is True
    window.assistant.go_offline()
    _pump()
    assert _state() is not None and _state()["active"] is False


def test_a_deaf_machine_reports_the_reason_instead_of_faking_active(window) -> None:
    window.devices = []
    window.scripted_start_listening()
    _pump()
    assert not window.conversation_active
    state = _state()
    assert state is not None and state["active"] is False
    assert "microphone" in state["reason"].lower()
