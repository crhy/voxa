"""The scripted listening actions on the application (issue #130), with a fake window.

The actions do not need a window until they run, and the window is built late (in
do_activate), so a stub gi module is enough to check the wiring in the real
VoxaApplication: the named actions exist, and activating them drives the window's
scripted start/stop - the same methods the Active button path ends up in.
"""

from __future__ import annotations

import importlib
import sys
import types

APP_ID = "io.github.crhy.voxa"


class FakeAction:
    def __init__(self, name: str, parameter_type) -> None:
        self.name = name
        self.parameter_type = parameter_type
        self.on_activate = None

    @staticmethod
    def new(name: str, parameter_type):
        return FakeAction(name, parameter_type)

    def connect(self, signal: str, handler) -> None:
        assert signal == "activate"
        self.on_activate = handler


class FakeApplication:
    def __init__(self, application_id=None, flags=None) -> None:
        self.application_id = application_id
        self.actions: dict[str, FakeAction] = {}
        self.window = None
        self.activated = 0

    def add_action(self, action: FakeAction) -> None:
        self.actions[action.name] = action

    def set_accels_for_action(self, action: str, accels) -> None:
        pass

    def activate(self) -> None:
        self.activated += 1


class FakeWindow:
    def __init__(self) -> None:
        self.log: list[str] = []

    def scripted_start_listening(self) -> None:
        self.log.append("start")

    def scripted_stop_listening(self) -> None:
        self.log.append("stop")

    def present(self) -> None:
        self.log.append("present")


def _load_application_with_fake_gi(monkeypatch):
    gi_module = types.ModuleType("gi")
    gi_module.require_version = lambda *_args: None
    repository = types.ModuleType("gi.repository")

    adw = types.ModuleType("gi.repository.Adw")
    adw.Application = FakeApplication
    adw.AboutDialog = object

    gio = types.ModuleType("gi.repository.Gio")
    gio.SimpleAction = FakeAction

    class _ApplicationFlags:
        DEFAULT_FLAGS = 0

    gio.ApplicationFlags = _ApplicationFlags

    glib = types.ModuleType("gi.repository.GLib")
    glib.VariantType = types.SimpleNamespace(new=lambda *_args: None)

    gtk = types.ModuleType("gi.repository.Gtk")
    gtk.License = types.SimpleNamespace(MIT_X11=0)

    repository.Adw = adw
    repository.Gio = gio
    repository.GLib = glib
    repository.Gtk = gtk
    gi_module.repository = repository

    for name in [
        "voxa.application",
        "gi",
        "gi.repository",
        "gi.repository.Adw",
        "gi.repository.Gio",
        "gi.repository.GLib",
        "gi.repository.Gtk",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "gi", gi_module)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)
    try:
        return importlib.import_module("voxa.application")
    finally:
        monkeypatch.delitem(sys.modules, "voxa.application", raising=False)


def test_the_scripted_listening_actions_drive_the_window(monkeypatch):
    module = _load_application_with_fake_gi(monkeypatch)
    app = module.VoxaApplication()
    assert {"start-listening", "stop-listening"} <= set(app.actions)
    assert app.actions["start-listening"].parameter_type is None
    assert app.actions["stop-listening"].parameter_type is None

    app.window = FakeWindow()
    app.actions["start-listening"].on_activate()
    app.actions["stop-listening"].on_activate()
    assert app.window.log == ["start", "stop"]


def test_start_listening_wakes_the_app_when_no_window_is_open(monkeypatch):
    module = _load_application_with_fake_gi(monkeypatch)
    app = module.VoxaApplication()
    assert app.window is None
    app.actions["start-listening"].on_activate()  # FakeApplication.activate does not build a window
    assert app.activated == 1  # do_activate is what builds the real window, then the action runs
    assert app.window is None  # no window: nothing was driven and nothing crashed
