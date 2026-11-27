"""Tests must never reach the real sound server.

A test run that started the real window once loaded an echo-cancel device into the developer's PulseAudio and
switched the default audio output. Pointing PULSE_SERVER at nothing makes every pactl/parec call fail harmlessly.
"""

import os

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
def _no_real_media_player(monkeypatch):
    """A player that happens to be running on the developer's desktop must not steer a test."""
    import sys

    window = sys.modules.get("voxa.window")
    if window is not None and hasattr(window, "active_player"):
        monkeypatch.setattr(window, "active_player", lambda: None)


@pytest.fixture(autouse=True)
def _hermetic_state_dir(monkeypatch, tmp_path):
    """Never read or write the developer's own state dir.

    The window's coaching state (suggestions.json) and the default action log live
    under XDG_STATE_HOME, which this shell may point somewhere real: a run that
    offered a coaching suggestion used to leave the next run convinced it had
    already been offered twice.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


# Every attribute MainWindow wires on a tool module when it is built.
_WINDOW_HOOKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("voxa.agent.tools.windows", ("ON_LOCK", "QUIT")),
    ("voxa.agent.tools.textedit", ("ask_model", "read_clipboard", "write_clipboard")),
    ("voxa.agent.tools.media", ("get_my_channel", "set_my_channel")),
    ("voxa.agent.tools.github", ("set_owner",)),
    ("voxa.agent.ui", ("say",)),
    ("voxa.agent.tools.contacts", ("get_contacts", "save_contacts")),
    ("voxa.agent.tools.translate", ("ask_model", "speak_in")),
    ("voxa.agent.tools.events", ("ask_model",)),
    ("voxa.agent.tools.convosummary", ("ask_model", "get_history")),
    ("voxa.agent.tools.projectsteps", ("ask_model",)),
    ("voxa.agent.tools.trips", ("ask_model",)),
    ("voxa.agent.tools.extract", ("ask_model",)),
    ("voxa.agent.tools.rebook", ("ask_model",)),
    ("voxa.agent.tools.slides", ("ask_model",)),
    ("voxa.agent.tools.briefing", ("get_location",)),
    ("voxa.agent.tools.anc", ("get_echo_mode", "set_echo_mode")),
    ("voxa.agent.tools.transcribe_file", ("transcriber",)),
    ("voxa.agent.tools.typing", ("LAST_REPLY",)),
)


@pytest.fixture(autouse=True)
def _window_hooks_are_restored():
    """A test that builds a MainWindow hands its own lambdas to the tool modules.

    Those are module globals, so they outlived the test: a later test that asked
    whether the clipboard hook is installed found the dead window's lambda and
    answered about a window that no longer existed.  Put the old values back.
    """
    import sys

    saved = []
    for name, attrs in _WINDOW_HOOKS:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr in attrs:
            saved.append((module, attr, getattr(module, attr, None)))
    yield
    for module, attr, value in reversed(saved):
        setattr(module, attr, value)
