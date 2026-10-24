"""The llama.cpp welcome brain: pure copy first, then the primary click on a fake window."""

import types

import pytest

from voxa.welcome import brain_status, brain_step, step_text


def test_brain_step():
    assert brain_step(True) == "ready"
    assert brain_step(False) == "brain"


def test_step_text_brain():
    headline, explanation, label = step_text("brain")
    assert headline == "One thing first"
    assert "530 MB" in explanation
    assert label == "Download my brain"


def test_brain_status_table():
    assert brain_status(0, 0) == "Downloading my brain…"
    assert brain_status(0, 100) == "Downloading my brain… 0%"
    assert brain_status(50, 100) == "Downloading my brain… 50%"
    assert brain_status(100, 100) == "Downloading my brain… 100%"


def test_on_welcome_primary_brain():
    gi = pytest.importorskip("gi")
    gi.require_version("Adw", "1")
    gi.require_version("Gdk", "4.0")
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    if not Gtk.init_check():
        pytest.skip("no GTK display available")

    from voxa.window import MainWindow

    calls: list[str] = []
    ns = types.SimpleNamespace(
        speech=types.SimpleNamespace(stop=lambda: calls.append("stop")),
        _download_small_brain=lambda: calls.append("brain"),
    )
    MainWindow._on_welcome_primary(ns, "brain")
    assert calls == ["stop", "brain"]
