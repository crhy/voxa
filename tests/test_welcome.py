"""Welcome copy tests: pure logic first, then the GTK dialog when a display exists."""

import pytest

from voxa.config import Settings
from voxa.welcome import greeting, next_step, spoken, step_text


def test_next_step_table():
    assert next_step("llamacpp", True, 0) == "ready"
    assert next_step("ollama", False, 0) == "install"
    assert next_step("ollama", True, 0) == "model"
    assert next_step("ollama", True, 2) == "ready"


def test_step_text_install():
    headline, explanation, label = step_text("install")
    assert headline == "One thing first"
    assert "Ollama" in explanation
    assert label == "Install Ollama"


def test_step_text_model():
    headline, explanation, label = step_text("model")
    assert headline == "One thing first"
    assert "qwen2.5:0.5b" in explanation
    assert label == "Download a model"


def test_step_text_ready():
    headline, explanation, label = step_text("ready")
    assert headline == "You're all set"
    assert explanation.startswith("Press ACTIVE")
    assert label == "Get started"


def test_spoken_install():
    text = spoken("install", "Aoife")
    assert text.startswith("Hello, I'm Aoife")
    assert "Ollama" in text


def test_greeting_falls_back_to_voxa():
    assert "Voxa" in greeting("")


def test_welcomed_defaults_false():
    assert Settings().welcomed is False


def _gtk():
    gi = pytest.importorskip("gi")
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Gtk

    if not Gtk.init_check():
        pytest.skip("no display for GTK")
    return Gtk


def test_dialog_primary_labels():
    _gtk()
    from voxa.ui.welcome_dialog import WelcomeDialog
    for step in ("install", "model", "ready"):
        dialog = WelcomeDialog("Voxa", step, lambda got: None)
        assert dialog.primary_button.get_label() == step_text(step)[2]


def test_dialog_primary_click():
    _gtk()
    from voxa.ui.welcome_dialog import WelcomeDialog
    calls = []
    dialog = WelcomeDialog("Voxa", "install", calls.append)
    dialog.primary_button.emit("clicked")
    assert calls == ["install"]


def test_dialog_later_click():
    _gtk()
    from voxa.ui.welcome_dialog import WelcomeDialog
    calls = []
    dialog = WelcomeDialog("Voxa", "install", calls.append)
    dialog.later_button.emit("clicked")
    assert calls == []


def test_dialog_no_tutorial_button():
    _gtk()
    from voxa import welcome
    from voxa.ui.welcome_dialog import WelcomeDialog
    dialog = WelcomeDialog("Voxa", "ready", lambda got: None, lambda: None)
    assert welcome.TUTORIAL_URL == ""
    assert dialog.tutorial_button is None
