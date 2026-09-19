"""The production MainWindow must be the new assistant shell, with the old UI reachable but not primary.

Regression tests for issue #7 section 1: a second full main-window implementation must not
quietly come back. They build the real MainWindow on a virtual display with throwaway config
and stub the slow or networked start-up work (Whisper, hardware probe, model list).
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

from voxa.ollama import OllamaError  # noqa: E402
from voxa.ui.shell import AssistantShell  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402
from voxa.window import MainWindow  # noqa: E402

_REAL_REFRESH = MainWindow._refresh_ollama_models  # the fixture stubs the class attribute


def _pump(iterations: int = 150) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


@pytest.fixture(scope="module")
def application():
    """One registered Adw.Application for the module (an application id can register once per process)."""
    app = Adw.Application(application_id="io.github.crhy.voxa.windowtest")
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
    win.present()
    _pump()
    yield win
    win._closing = True
    win.destroy()
    _pump(20)


def _is_descendant(widget: Gtk.Widget, ancestor: Gtk.Widget) -> bool:
    node = widget
    while node is not None:
        if node is ancestor:
            return True
        node = node.get_parent()
    return False


def test_production_window_is_the_assistant_shell(window) -> None:
    assert isinstance(window.shell, AssistantShell)
    assert _is_descendant(window.shell, window)
    assert window.shell.get_mapped()
    assert window.assistant_model.state is AssistantState.OFFLINE


def test_assistant_is_centered_in_the_real_window(window) -> None:
    ok, rect = window.shell.assistant_view.compute_bounds(window)
    assert ok
    assert abs((rect.get_x() + rect.get_width() / 2) - window.get_width() / 2) <= 2


def test_no_transcript_editor_or_chat_box_in_the_main_view(window) -> None:
    for old in (window.transcript_view, window.response_view, window.record_button, window.ask_button):
        assert not _is_descendant(old, window.shell)
        assert not old.get_mapped()
    assert not _is_descendant(window.transcript_view, window)


def test_legacy_dictation_view_is_reachable_from_the_menu_and_shortcuts(window) -> None:
    assert window.lookup_action("transcript") is not None
    for shortcut_action in ("record", "conversation", "ask", "copy", "clear", "preferences"):
        assert window.lookup_action(shortcut_action) is not None

    window.show_transcript_window()
    _pump()
    secondary = window._legacy_window
    assert secondary is not None and secondary.get_visible()
    assert _is_descendant(window.transcript_view, secondary)
    secondary.close()  # hide-on-close: it can be shown again
    _pump()
    assert not secondary.get_visible()
    window.show_transcript_window()
    _pump()
    assert secondary.get_visible()
    secondary.destroy()


def test_late_hardware_callback_after_close_is_harmless(window) -> None:
    # MainWindow used to call the nonexistent Gtk.Window.is_destroyed() here and raise.
    window._closing = True
    window._apply_hardware_summary(8.0, "GPU VRAM", [])
    assert "is_destroyed" not in open(MainWindow.__module__.replace(".", "/") + ".py", encoding="utf-8").read()


def test_shell_controls_drive_the_assistant_model(window) -> None:
    window.shell.on_offline()
    assert window.assistant_model.state is AssistantState.OFFLINE


def test_paperclip_is_honest_until_attachments_exist(window) -> None:
    button = window.shell.attachment_button
    assert not button.get_sensitive()
    assert "later" in button.get_tooltip_text()


def test_user_is_told_once_when_the_speech_model_is_ready(window, monkeypatch) -> None:
    toasts: list[str] = []
    monkeypatch.setattr(window, "_toast", toasts.append)
    window._on_whisper_ready("base", "CPU")
    window._on_whisper_ready("small", "CPU")  # switching models later must not repeat the hint
    assert [t for t in toasts if "ACTIVE" in t] == ["Voxa is ready. Press ACTIVE to start listening."]


def test_an_offline_ai_server_is_explained_in_the_model_picker(window, monkeypatch) -> None:
    class Offline:
        def list_models(self):
            raise OllamaError("Could not connect")

    monkeypatch.setattr(window, "_ai_client", lambda: Offline())
    window.settings.ollama_model = "keep-me"
    _REAL_REFRESH(window)

    selector = window.shell.model_selector
    deadline_ok = False
    for _ in range(200):
        _pump(5)
        if selector._items.get_n_items() == 1 and selector._items.get_item(0).get_string() == "No models available":
            deadline_ok = True
            break
    assert deadline_ok
    assert window.settings.ollama_model == "keep-me"  # an outage must not erase the saved choice
