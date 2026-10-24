"""The window UI must reflect the llama.cpp engine (the default backend).

Covers the three sub-tasks of u1-engine-ui: the status dot follows the engine's
health (not the GGUF-path list rule), the model picker shows brain labels instead
of GGUF paths, and picking a label sets ``llamacpp_model`` and restarts the server.
No real windows are built: the pure helpers are called directly and the window
methods are driven through ``types.SimpleNamespace`` fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voxa import engine, smallbrain
from voxa.server import SERVER_FAILED, SERVER_READY, SERVER_STARTING

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.model_selector import ai_status_for_backend  # noqa: E402


def test_status_dot_follows_engine_health_for_llamacpp():
    # GGUF paths never contain spaces, so the old list rule would read "online"
    # even with the server down; the engine's health decides instead.
    gguf_path = str(smallbrain.path_for(smallbrain.SMALL_BRAIN))
    assert ai_status_for_backend("llamacpp", [gguf_path], SERVER_READY) == "online"
    assert ai_status_for_backend("llamacpp", [gguf_path], SERVER_STARTING) == "starting"
    assert ai_status_for_backend("llamacpp", [gguf_path], SERVER_FAILED) == "offline"
    assert ai_status_for_backend("llamacpp", [], SERVER_READY) == "online"


def test_other_backends_keep_the_list_rule():
    assert ai_status_for_backend("ollama", ["qwen2.5:0.5b"], SERVER_FAILED) == "online"
    assert ai_status_for_backend("ollama", [], SERVER_READY) == "offline"
    assert ai_status_for_backend("strata", ["Strata is not running"], SERVER_READY) == "offline"


def test_available_labels_offers_installed_or_selected_brains(monkeypatch):
    monkeypatch.setattr(smallbrain, "is_installed", lambda model, directory=None: False)
    assert smallbrain.available_labels("") == []
    # A selected brain is offered even when it is not installed on disk.
    selected = str(smallbrain.path_for(smallbrain.LIGHTEST_BRAIN))
    assert smallbrain.available_labels(selected) == [smallbrain.LIGHTEST_BRAIN.label]
    # An installed brain is offered even when nothing is selected.
    monkeypatch.setattr(smallbrain, "is_installed", lambda model, directory=None: model is smallbrain.SMALL_BRAIN)
    assert smallbrain.available_labels("") == [smallbrain.SMALL_BRAIN.label]


def test_label_and_path_round_trip():
    for brain in smallbrain.BRAINS:
        path = smallbrain.path_for_label(brain.label)
        assert path == str(smallbrain.path_for(brain))
        assert smallbrain.label_for_path(path) == brain.label
    assert smallbrain.path_for_label("no such brain") == ""
    assert smallbrain.label_for_path("/nowhere/none.gguf") == ""


SERVER_IDLE_SENTINEL = "IDLE"


def _fake_window(backend, *, llamacpp_model="", ollama_model="", server_status=SERVER_IDLE_SENTINEL, welcome=False):
    settings = SimpleNamespace(ai_backend=backend, llamacpp_model=llamacpp_model, ollama_model=ollama_model)
    saved: list[object] = []
    restarted: list[str] = []

    class Selector:
        status = None

        def set_status(self, value):
            self.status = value

    class Shell:
        def __init__(self):
            self.model_selector = Selector()

        def set_models(self, *args):
            pass

    class Server:
        def __init__(self, status):
            self._status = status

        def health(self):
            return SimpleNamespace(status=self._status)

    return SimpleNamespace(
        settings=settings,
        _welcome_followup=welcome,
        _say_welcome_how_to=lambda: None,
        shell=Shell(),
        ai_server=Server(server_status),
        ollama_models=[],
        config_store=SimpleNamespace(save=lambda s: saved.append(s)),
        _apply_model_combo=lambda models: None,
        _restart_ai_server_async=lambda: restarted.append("restart"),
    ), saved, restarted


def test_apply_ollama_models_shows_labels_and_uses_health_for_llamacpp(monkeypatch, tmp_path):
    from voxa.window import MainWindow

    monkeypatch.setattr(engine, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(smallbrain, "is_installed", lambda model, directory=None: model is smallbrain.SMALL_BRAIN)
    selected = str(tmp_path / smallbrain.SMALL_BRAIN.filename)
    (tmp_path / smallbrain.SMALL_BRAIN.filename).write_bytes(b"x")
    win, saved, _ = _fake_window("llamacpp", llamacpp_model=selected, server_status=SERVER_FAILED)

    # The raw list from the engine is GGUF paths; the picker must show the label.
    MainWindow._apply_ollama_models(win, [selected])

    assert win.ollama_models == [smallbrain.SMALL_BRAIN.label]
    assert win.shell.model_selector.status == "offline"  # server down, despite a path present
    assert saved == []  # a selected brain whose file exists is left alone


def test_apply_ollama_models_default_picks_a_brain_for_llamacpp(monkeypatch):
    from voxa.window import MainWindow

    monkeypatch.setattr(smallbrain, "is_installed", lambda model, directory=None: model is smallbrain.SMALL_BRAIN)
    win, saved, _ = _fake_window("llamacpp", llamacpp_model="", server_status=SERVER_READY)

    MainWindow._apply_ollama_models(win, [])

    assert win.ollama_models == [smallbrain.SMALL_BRAIN.label]
    assert win.settings.llamacpp_model == str(smallbrain.path_for(smallbrain.BRAINS[0]))
    assert win.settings.ollama_model == ""  # the Ollama setting is never touched
    assert len(saved) == 1
    assert win.shell.model_selector.status == "online"


def test_shell_model_pick_sets_llamacpp_path_and_restarts():
    from voxa.window import MainWindow

    win, saved, restarted = _fake_window("llamacpp", llamacpp_model="", server_status=SERVER_READY)
    win.ollama_models = [smallbrain.SMALL_BRAIN.label]

    MainWindow._on_shell_model_selected(win, smallbrain.SMALL_BRAIN.label)

    assert win.settings.llamacpp_model == str(smallbrain.path_for(smallbrain.SMALL_BRAIN))
    assert len(saved) == 1
    assert restarted == ["restart"]


def test_shell_model_pick_leaves_ollama_backend_untouched():
    from voxa.window import MainWindow

    win, saved, restarted = _fake_window("ollama", ollama_model="keep-me", server_status=SERVER_READY)
    win.ollama_models = ["keep-me", "other"]
    win._warm_up_model = lambda name: None

    MainWindow._on_shell_model_selected(win, "other")

    assert win.settings.llamacpp_model == ""  # llama.cpp path never set for Ollama users
    assert win.settings.ollama_model == "other"
    assert restarted == []
