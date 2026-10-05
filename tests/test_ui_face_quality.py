"""The Facial Quality switch: Low / Medium / High on the main screen."""

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
from voxa.ui.character_picker import PortraitPicker  # noqa: E402
from voxa.ui.face_quality import FaceQualitySwitch, mode_for_label, quality_label  # noqa: E402
from voxa.window import MainWindow  # noqa: E402


def _pump(iterations: int = 150) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


def test_quality_label_table() -> None:
    assert quality_label("still") == "Low"
    assert quality_label("prerendered") == "Medium"
    assert quality_label("live") == "High"
    assert quality_label("bogus") == "Medium"


def test_mode_for_label_table() -> None:
    assert mode_for_label("Low") == "still"
    assert mode_for_label("Medium") == "prerendered"
    assert mode_for_label("High") == "live"
    assert mode_for_label("bogus") == "prerendered"


def test_set_mode_does_not_fire() -> None:
    fired: list[str] = []
    switch = FaceQualitySwitch(on_selected=fired.append)
    switch.set_mode("still")
    assert switch.get_mode() == "still"
    assert fired == []


def test_clicking_low_fires_still_once() -> None:
    fired: list[str] = []
    switch = FaceQualitySwitch(on_selected=fired.append)
    switch._buttons["still"].emit("clicked")
    assert fired == ["still"]
    assert switch.get_mode() == "still"


def test_clicking_active_button_does_not_fire() -> None:
    fired: list[str] = []
    switch = FaceQualitySwitch(on_selected=fired.append)
    switch._buttons["prerendered"].emit("clicked")
    assert fired == []
    assert switch.get_mode() == "prerendered"


def test_set_live_available_disables_high_and_falls_back() -> None:
    fired: list[str] = []
    switch = FaceQualitySwitch(on_selected=fired.append)
    switch.set_mode("live")
    assert fired == []
    switch.set_live_available(False)
    assert switch._buttons["live"].has_css_class("linked")
    assert not switch._buttons["live"].get_sensitive()
    assert fired == ["prerendered"]
    assert switch.get_mode() == "prerendered"


def test_popover_has_no_face_dropdown() -> None:
    picker = PortraitPicker()
    picker.refresh([])
    assert not hasattr(picker, "set_face_mode")
    assert not hasattr(picker, "get_face_mode")


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.facetest")
    app.register(None)
    return app


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    for var, sub in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        monkeypatch.setenv(var, str(tmp_path / sub))
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
    win.audio = None
    win.speech = None
    win.devices = [AudioDevice(identifier="mic", name="Microphone", device=None)]
    win.present()
    _pump()
    yield win
    win._closing = True
    win.destroy()
    _pump(20)


def test_bottom_controls_hold_the_switch(window) -> None:
    win = window
    controls = win.shell._bottom_controls
    found = False
    child = controls.get_first_child()
    while child is not None:
        if child is win.shell.face_quality:
            found = True
        child = child.get_next_sibling()
    assert found
    assert win.shell.face_quality.get_mode() == win.settings.face_mode


def test_choosing_low_through_the_window_saves_still(window) -> None:
    win = window
    win.shell.face_quality._buttons["still"].emit("clicked")
    _pump(5)
    assert win.settings.face_mode == "still"
