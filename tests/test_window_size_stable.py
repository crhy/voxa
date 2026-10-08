"""The main window's natural size must not change when the user clicks Active.

Regression test for issue N08: clicking Active runs the activation code path, which
pushes a long "getting ready" caption into the assistant view. A GTK window grows when a
child's natural size exceeds it, so an unconstrained caption label made the window jump.
These tests build the real MainWindow on a virtual display, measure the window's natural
size while idle, then run the same code path the Active button takes (with listening faked)
and assert the natural size is unchanged.
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

from voxa.ui.state import AssistantState  # noqa: E402
from voxa.window import MainWindow  # noqa: E402


def _pump(iterations: int = 150) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


def _natural(widget: Gtk.Widget) -> tuple[int, int]:
    width = widget.measure(Gtk.Orientation.HORIZONTAL, -1)[1]
    height = widget.measure(Gtk.Orientation.VERTICAL, -1)[1]
    return (width, height)


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.sizetest")
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
    win.present()
    _pump()
    yield win
    win._closing = True
    win.destroy()
    _pump(20)


GETTING_READY = (
    "Getting ready\u2026 \u2014 Loading the speech model \u2014 "
    "I'll start listening as soon as it's ready"
)


def test_idle_natural_size_is_the_minimum(window) -> None:
    width, height = _natural(window)
    assert width == 850
    assert height == 607


def test_active_getting_ready_does_not_change_natural_width(window) -> None:
    before = _natural(window)
    window.assistant_model.set_state(AssistantState.OFFLINE, GETTING_READY)
    _pump()
    assert _natural(window)[0] == before[0]


def test_listening_does_not_change_natural_size(window) -> None:
    before = _natural(window)
    window.assistant_model.set_state(AssistantState.LISTENING, "")
    _pump()
    assert _natural(window) == before
