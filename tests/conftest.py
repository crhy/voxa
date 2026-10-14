"""Tests must never reach the real sound server.

A test run that started the real window once loaded an echo-cancel device into the developer's PulseAudio and
switched the default audio output. Pointing PULSE_SERVER at nothing makes every pactl/parec call fail harmlessly.
"""

import os

import pytest

os.environ["PULSE_SERVER"] = "unix:/nonexistent-voxa-tests"
os.environ["VOXA_NO_WELCOME"] = "1"


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
def _no_real_media_player(monkeypatch):
    """A player that happens to be running on the developer's desktop must not steer a test."""
    import sys

    window = sys.modules.get("voxa.window")
    if window is not None and hasattr(window, "active_player"):
        monkeypatch.setattr(window, "active_player", lambda: None)
