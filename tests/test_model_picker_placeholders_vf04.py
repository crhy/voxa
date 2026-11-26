"""vf04 (#72): the model picker's placeholders must never become a model.

vf02 stored "Ollama not running" in ``self.ollama_models`` so the legacy
combo, the model manager and the Preferences save all read it as a real
model: opening Preferences (or picking the only entry) while Ollama was down
saved "Ollama not running" into ``ollama_model``.  The placeholder is display
text only now; choosing the "<model> (not installed)" entry saves "<model>".
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

from voxa.window import MainWindow  # noqa: E402


def _pump(iterations: int = 60) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


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


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.pickerplaceholderstest")
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
        "_warm_up_model",
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


def _offline_ollama(window) -> None:
    window.settings.ai_backend = "ollama"
    window.settings.ollama_model = "qwen3.5-0.8b-q4km:latest"
    MainWindow._show_no_models(window, "Ollama")


def test_placeholder_stays_out_of_the_real_model_list(window) -> None:
    _offline_ollama(window)
    assert window.ollama_models == []
    assert window._models_unreachable is True
    # The picker still DISPLAYS the placeholder instead of "(None)".
    assert window.shell.model_selector.get_selected() == "Ollama not running"


def test_choosing_the_placeholder_saves_nothing(window) -> None:
    _offline_ollama(window)
    combo_entry = window.shell.model_selector.get_selected()
    assert combo_entry == "Ollama not running"
    window._on_shell_model_selected(combo_entry)
    assert window.settings.ollama_model == "qwen3.5-0.8b-q4km:latest"
    # The legacy combo shows the placeholder too; choosing it saves nothing.
    assert window.model_combo.get_model().get_item(0).get_string() == "Ollama not running"
    window._on_model_selected(window.model_combo, "selected")
    assert window.settings.ollama_model == "qwen3.5-0.8b-q4km:latest"


def test_choosing_the_not_installed_entry_saves_the_bare_model(window) -> None:
    window.settings.ai_backend = "ollama"
    window.settings.ollama_model = "keep-me"
    window._models_unreachable = False
    # A reachable backend missing the configured model shows it suffixed.
    MainWindow._apply_ollama_models(window, ["other-model"])
    assert window.settings.ollama_model == "other-model"
    window.settings.ollama_model = "keep-me"
    MainWindow._apply_model_combo(window, ["other-model"], reachable=True)
    entry = window.shell.model_selector.get_selected()
    assert entry == "keep-me (not installed)"
    window._on_shell_model_selected(entry)
    assert window.settings.ollama_model == "keep-me"
    assert "not installed" not in window.settings.ollama_model


def test_preferences_save_while_unreachable_leaves_model_unchanged(window, monkeypatch) -> None:
    _offline_ollama(window)
    window.devices = []  # the preferences device rows need real AudioDevice fields

    captured: list[Adw.PreferencesDialog] = []
    original_present = Adw.PreferencesDialog.present

    def spy_present(self, parent=None):
        captured.append(self)
        return original_present(self, parent)

    monkeypatch.setattr(Adw.PreferencesDialog, "present", spy_present)
    window.show_preferences()
    _pump()
    assert captured, "the Preferences dialog did not open"
    captured[0].close()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        if not captured[0].get_visible():
            break
        time.sleep(0.01)
    _pump(20)
    assert window.settings.ollama_model == "qwen3.5-0.8b-q4km:latest"
    assert window.ollama_models == []
