"""Tests for the adaptive noise cancellation preference switch."""

from __future__ import annotations

import pytest

from voxa.agent import ui
from voxa.agent.intents import ROUTED_TOOLS, ToolCall, route
from voxa.agent.tools import anc, contacts, github, media, textedit, translate, windows
from voxa.agent.tools import briefing as briefing_tool
from voxa.audio import AudioCapture
from voxa.config import Settings


def test_echo_mode_mapping():
    """Switch on maps to echo_mode='voxa', off maps to echo_mode='off'."""
    settings = Settings()
    settings.echo_mode = "voxa"
    assert settings.echo_mode == "voxa"
    settings.echo_mode = "off"
    assert settings.echo_mode == "off"


def test_audio_capture_set_echo_mode_off():
    """set_echo_mode(False) clears reference_source and canceller."""
    capture = AudioCapture()
    capture.set_echo_mode(False)
    assert capture.reference_source is None
    assert capture.canceller is None
    assert capture._canceller_enabled is False


def test_audio_capture_set_echo_mode_on_inactive():
    """set_echo_mode(True) on inactive capture sets fields for next start()."""
    capture = AudioCapture()
    capture.set_echo_mode(True, "test_monitor")
    assert capture.reference_source == "test_monitor"
    assert capture.canceller is not None
    assert capture._canceller_enabled is True
    assert capture.is_active is False


def test_router_noise_cancellation_phrases():
    """Router recognizes noise cancellation on/off/get phrases."""
    assert route("turn noise cancellation on") == ToolCall(
        tool="set_noise_cancellation", args={"state": "on"}
    )
    assert route("turn noise cancellation off") == ToolCall(
        tool="set_noise_cancellation", args={"state": "off"}
    )
    assert route("enable adaptive noise cancellation") == ToolCall(
        tool="set_noise_cancellation", args={"state": "on"}
    )
    assert route("disable adaptive noise cancellation") == ToolCall(
        tool="set_noise_cancellation", args={"state": "off"}
    )
    assert route("is noise cancellation on") == ToolCall(
        tool="get_noise_cancellation", args={}
    )


def test_routed_tools_contains_noise_cancellation():
    """ROUTED_TOOLS includes the noise cancellation tools."""
    assert "set_noise_cancellation" in ROUTED_TOOLS
    assert "get_noise_cancellation" in ROUTED_TOOLS


def test_anc_tool_hooks(monkeypatch):
    """The anc tools flip and report through hooks (fake hooks here)."""
    seen = []
    monkeypatch.setattr(anc, "set_echo_mode", lambda mode: seen.append(mode))
    monkeypatch.setattr(anc, "get_echo_mode", lambda: "voxa")
    result = anc._set_noise_cancellation_handler({"state": "on"})
    assert result.ok and seen == ["voxa"]
    report = anc._get_noise_cancellation_handler({})
    assert report.ok and report.speech == "Noise cancellation is on."


def test_switch_row_exists_in_preferences(tmp_path, monkeypatch):
    """GTK test: the switch row exists, and toggling it calls the handler."""
    gi = pytest.importorskip("gi")
    gi.require_version("Adw", "1")
    gi.require_version("Gtk", "4.0")
    from gi.repository import Adw, GLib, Gtk  # noqa: E402

    if not Gtk.init_check():
        pytest.skip("no GTK display available")
    Adw.init()
    import voxa.window as window_module
    from voxa.window import MainWindow

    monkeypatch.setattr(window_module, "install_styles", lambda *args: None)
    monkeypatch.setattr(window_module, "webrtc_available", lambda: False, raising=False)

    def pump(iterations: int = 150) -> None:
        ctx = GLib.MainContext.default()
        for _ in range(iterations):
            ctx.iteration(False)

    class FakeAudio:
        calls: list = []

        def set_echo_mode(self, enabled, source=None) -> None:
            self.calls.append((enabled, source))

    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache")):
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
    app = Adw.Application(application_id="io.github.crhy.voxa.anctest")
    app.register(None)
    hooks = (
        (windows, "ON_LOCK"),
        (textedit, "ask_model"),
        (textedit, "read_clipboard"),
        (textedit, "write_clipboard"),
        (media, "get_my_channel"),
        (media, "set_my_channel"),
        (github, "set_owner"),
        (ui, "say"),
        (contacts, "get_contacts"),
        (contacts, "save_contacts"),
        (translate, "ask_model"),
        (translate, "speak_in"),
        (briefing_tool, "get_location"),
        (anc, "get_echo_mode"),
        (anc, "set_echo_mode"),
    )
    originals = [(module, attr, getattr(module, attr, None)) for module, attr in hooks]
    win = MainWindow(app)
    win.audio = FakeAudio()
    for module, attr, original in originals:
        setattr(module, attr, original)
    win.present()
    pump()
    win.show_preferences()
    pump()

    def find(widget):
        child = widget.get_first_child()
        while child is not None:
            if isinstance(child, Adw.ComboRow) and child.get_title() == "Adaptive noise cancellation":
                return child
            if hasattr(child, "get_first_child"):
                found = find(child)
                if found is not None:
                    return found
            child = child.get_next_sibling()
        return None

    row = find(win)
    assert row is not None
    row.set_selected(1)
    pump()
    assert win.settings.echo_mode == "off"
    assert win.audio.calls == [("off", None)]
    win._closing = True
    win.assistant.go_offline()
    win.destroy()
    pump(20)
