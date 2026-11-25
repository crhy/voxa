"""Tests must never reach the real sound server.

A test run that started the real window once loaded an echo-cancel device into the developer's PulseAudio and
switched the default audio output. Pointing PULSE_SERVER at nothing makes every pactl/parec call fail harmlessly.
"""

import os

# NumPy 2 huge-page madvise stalls khugepaged on this machine (see voxa/__init__.py);
# the gate box has tens of GB pinned by the local model. Set before anything imports
# numpy: conftest is the first file pytest loads.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import pytest

os.environ["PULSE_SERVER"] = "unix:/nonexistent-voxa-tests"
os.environ["VOXA_NO_WELCOME"] = "1"
# GTK's software renderer: under xvfb the GPU renderer took several GB outside the test process and grew through
# the suite (on top of a local AI server's locked memory this locked up the PC on 2026-10-23), then crashed in
# window creation after a reboot. With cairo the whole suite uses about 1 GB and runs in 75 s instead of 2-4 min.
os.environ["GSK_RENDERER"] = "cairo"


@pytest.fixture(autouse=True)
def _host_environment(monkeypatch):
    """Keep command assertions independent of whether pytest itself runs in Flatpak."""
    import voxa.agent.host as host
    import voxa.agent.jsruntime as jsruntime
    import voxa.agent.tools.typing as typing
    import voxa.apps as apps
    import voxa.documents as documents
    import voxa.mail as mail

    monkeypatch.delenv("FLATPAK_ID", raising=False)
    for module in (host, jsruntime, typing, apps, documents, mail):
        monkeypatch.setattr(module, "IN_FLATPAK", False)


@pytest.fixture(autouse=True)
def _no_playback_delay(monkeypatch):
    """Stations and channels start at once in tests instead of after her spoken confirmation."""
    import voxa.agent.player as player

    monkeypatch.setattr(player, "START_DELAY_SECONDS", 0.0)


@pytest.fixture(autouse=True)
def _no_real_desktop_control(monkeypatch):
    """No test may touch the desktop's accessibility or send real input to the screen.

    A test that reached the real a11y bus would switch the developer's desktop accessibility on and could leave it
    there; a real screenshot or click would drive the desktop behind rhY's back. Both layers take their D-Bus client,
    screenshot, OCR, window lookup and input sink from module attributes, so every test gets harmless ones.
    """
    import subprocess

    from voxa.agent import a11y_scope, ui, visual_control

    class DeadBus:
        def is_enabled(self):
            return False

        def set_enabled(self, value):
            return False

    class RefusedSink:
        def click(self, x, y):
            return False

        def type_text(self, text):
            return False

    monkeypatch.setattr(a11y_scope, "GdbusBus", DeadBus)
    monkeypatch.setattr(ui, "a11y_bus", DeadBus())
    monkeypatch.setattr(visual_control, "capture_screen", lambda: "")
    monkeypatch.setattr(visual_control, "ocr_words", lambda path: [])
    monkeypatch.setattr(visual_control, "window_region", lambda app, runner=subprocess.run: None)
    monkeypatch.setattr(visual_control, "XTestSink", lambda: RefusedSink())


@pytest.fixture(autouse=True)
def _no_real_media_player(monkeypatch):
    """A player that happens to be running on the developer's desktop must not steer a test."""
    import sys

    window = sys.modules.get("voxa.window")
    if window is not None and hasattr(window, "active_player"):
        monkeypatch.setattr(window, "active_player", lambda: None)
