from pathlib import Path

import pytest

from voxa.apps import DesktopApp, launch
from voxa.documents import open_in_libreoffice
from voxa.mail import compose
from voxa.simulation import actions_simulated


@pytest.fixture
def simulated(monkeypatch):
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    return True


def booms(*args, **kwargs):
    raise AssertionError(f"real action attempted: {args} {kwargs}")


def test_actions_simulated_reads_env(monkeypatch):
    monkeypatch.delenv("VOXA_SIMULATE_ACTIONS", raising=False)
    assert actions_simulated() is False
    for value in ("1", "true", "YES", " on ", "On"):
        monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", value)
        assert actions_simulated() is True
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "")
    assert actions_simulated() is False


def test_launch_simulated(simulated, monkeypatch):
    monkeypatch.setattr("voxa.apps.subprocess.Popen", booms)
    launch(DesktopApp("Firefox", "/usr/share/applications/firefox.desktop"))


def test_compose_simulated(simulated, monkeypatch):
    monkeypatch.setattr("voxa.mail.subprocess.Popen", booms)
    compose("mom@example.com", "Dinner", "Hi Mom!")


def test_open_in_libreoffice_simulated(simulated, monkeypatch):
    monkeypatch.setattr("voxa.documents.subprocess.Popen", booms)
    open_in_libreoffice(Path("/tmp/doc.odt"))


def test_real_actions_still_run(monkeypatch):
    monkeypatch.delenv("VOXA_SIMULATE_ACTIONS", raising=False)
    launched = []
    monkeypatch.setattr("voxa.apps.subprocess.Popen", lambda command, **kwargs: launched.append(command))
    monkeypatch.setattr("voxa.mail.subprocess.Popen", lambda command, **kwargs: launched.append(command))
    monkeypatch.setattr("voxa.documents.subprocess.Popen", lambda command, **kwargs: launched.append(command))
    launch(DesktopApp("Firefox", "/usr/share/applications/firefox.desktop"))
    compose("mom@example.com", "Dinner", "Hi Mom!")
    open_in_libreoffice(Path("/tmp/doc.odt"))
    assert len(launched) == 3
