"""Regression tests for the production MainWindow (issue #7, phase A).

Each test drives the real window with fake audio and speech, and each one is
independent: it checks one user-visible guarantee that must never regress.
"""

from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")
Adw.init()

from voxa.audio import AudioDevice  # noqa: E402
from voxa.config import ConfigStore  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402
from voxa.window import MainWindow  # noqa: E402

_REAL_REFRESH = MainWindow._refresh_ollama_models  # the fixture stubs the class attribute


def _pump(iterations: int = 150) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


class FakeAudio:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._active = False
        self.stuck = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, device_id, on_audio, on_level, on_error) -> None:
        self.calls.append("start")
        self._active = True

    def stop(self) -> None:
        self.calls.append("stop")
        if not self.stuck:
            self._active = False


class FakeSpeech:
    def __init__(self) -> None:
        self.stops = 0
        self.callbacks: dict[str, object] = {}

    def speak(self, text, rate, voice, on_started=None, on_done=None, on_error=None) -> None:
        self.callbacks = {"started": on_started, "done": on_done, "error": on_error}

    def stop(self) -> None:
        self.stops += 1


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.regressiontest")
    app.register(None)
    return app


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("_load_whisper", "_load_wake_whisper", "_detect_hardware_async", "_refresh_ollama_models", "_refresh_devices"):
        monkeypatch.setattr(MainWindow, name, lambda *args, **kwargs: None)
    win = MainWindow(application)
    win.audio = FakeAudio()
    win.speech = FakeSpeech()
    win.whisper._model = object()  # WhisperService.ready
    win.devices = [AudioDevice(identifier="mic", name="Microphone", device=None)]
    win.present()
    _pump()
    yield win
    win._closing = True
    win.assistant.go_offline()
    win.destroy()
    _pump(20)


def _state(win) -> AssistantState:
    return win.assistant_model.state


def test_window_starts_offline_with_the_microphone_off(window) -> None:
    # Privacy: Voxa never listens until the user presses ACTIVE.
    assert _state(window) is AssistantState.OFFLINE
    assert window.audio.is_active is False
    assert not window.assistant_model.tasks
    controls = window.shell.status_controls
    assert controls.offline_button.has_css_class("selected")
    assert not controls.active_button.has_css_class("selected")


def test_closing_the_window_while_active_stops_listening(window) -> None:
    assert window.assistant.activate() is True
    assert window.audio.is_active
    assert window.do_close_request() is False  # False means "let the window close"
    assert window.audio.is_active is False
    assert _state(window) is AssistantState.OFFLINE


def test_preferences_and_shortcuts_dialogs_open(window) -> None:
    window.show_preferences()
    _pump()
    window.show_shortcuts()
    _pump()


def test_dictation_from_the_transcript_window_still_works(window) -> None:
    window.start_recording()
    _pump()
    assert window.listening is True
    assert window.audio.is_active
    assert _state(window) is AssistantState.OFFLINE  # dictation never turns the assistant on
    window.stop_recording()
    _pump()
    assert window.listening is False
    assert window.audio.is_active is False
    assert _state(window) is AssistantState.OFFLINE


@pytest.mark.parametrize("appearance", ["system", "light", "dark"])
def test_every_appearance_setting_applies(window, appearance) -> None:
    window.settings.appearance = appearance
    window._apply_appearance()


def test_choosing_a_model_in_the_shell_is_saved_to_config(window) -> None:
    window._apply_ollama_models(["a", "b"])
    window.shell.on_model_selected("b")
    assert window.settings.ollama_model == "b"
    assert ConfigStore().load().ollama_model == "b"


def test_choosing_a_backend_in_the_shell_is_saved_and_pushed_back(window) -> None:
    window.shell.on_backend_selected("ollama")
    assert window.settings.ai_backend == "ollama"
    assert ConfigStore().load().ai_backend == "ollama"
    assert window.shell.model_selector.get_backend() == "ollama"
    window._apply_model_combo(["m1"])
    assert window.shell.model_selector.get_backend() == "ollama"
    # And the other way round: whatever Preferences saved (settings.ai_backend)
    # is what the shell picker is shown, via _apply_model_combo.
    window.settings.ai_backend = "llamacpp"
    window._apply_model_combo(["m1"])
    assert window.shell.model_selector.get_backend() == "llamacpp"


@pytest.mark.parametrize(
    ("backend", "label"),
    [("llamacpp", "llama.cpp"), ("ollama", "Ollama")],
)
def test_model_picker_lists_models_with_the_backend_named(window, backend, label) -> None:
    class FakeClient:
        def list_models(self) -> list[str]:
            return ["m1", "m2"]

    window.settings.ai_backend = backend
    window._ai_client = lambda: FakeClient()
    _REAL_REFRESH(window)
    _pump(250)

    selector = window.shell.model_selector
    items = [selector._items.get_string(i) for i in range(selector._items.get_n_items())]
    assert items == ["m1", "m2"]
    assert selector.get_backend() == backend
    assert selector._backend_dropdown.get_selected_item().get_string() == label
    assert selector._backend_dropdown.get_selected() == (1 if backend == "ollama" else 0)


def test_activation_is_refused_with_a_reason_when_no_microphone_or_model_is_ready(window) -> None:
    window.whisper._model = None
    assert window.assistant.activate() is False
    assert _state(window) is AssistantState.OFFLINE
    assert "Whisper" in window._start_failure

    window.whisper._model = object()
    window.devices = []
    assert window.assistant.activate() is False
    assert _state(window) is AssistantState.OFFLINE
    assert "microphone" in window._start_failure.lower()
