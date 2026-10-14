from __future__ import annotations

import pytest

from voxa import apps
from voxa.agent import host
from voxa.agent.tools.applications import open_app

DUMP = """@@@/usr/share/applications/io.github.crhy.Gigolo.desktop
[Desktop Entry]
Type=Application
Name=Gigolo
@@@/usr/share/applications/io.github.crhy.BrutalChess.desktop
[Desktop Entry]
Type=Application
Name=Brutal Chess
@@@/usr/share/applications/com.example.Pluma.desktop
[Desktop Entry]
Type=Application
Name=Pluma
@@@/usr/share/applications/org.libreoffice.LibreOffice.writer.desktop
[Desktop Entry]
Type=Application
Name=LibreOffice Writer
"""

APPS = apps.parse_desktop_dump(DUMP)


def _name(query: str) -> str | None:
    app = apps.match_app(query, APPS)
    return app.name if app else None


def test_match_app_rejects_weak_fuzzy():
    assert _name("google") is None
    assert _name("brutal chest") == "Brutal Chess"
    assert _name("plume") == "Pluma"
    assert _name("the office") == "LibreOffice Writer"


@pytest.fixture
def opened(monkeypatch):
    commands: list[list[str]] = []

    def fake_spawn(command):
        commands.append(list(command))

    monkeypatch.setattr(host, "spawn", fake_spawn)
    monkeypatch.setattr(apps, "list_apps", lambda: APPS)
    monkeypatch.setattr(apps, "launch", lambda app: None)
    return commands


def test_known_site_opens_directly(opened):
    result = open_app({"name": "cnn"})
    assert result.ok
    assert result.detail == "https://www.cnn.com/"
    assert opened == [["xdg-open", "https://www.cnn.com/"]]


def test_generic_email_client_falls_back_to_gmail(opened):
    result = open_app({"name": "email client"})
    assert result.ok
    assert result.detail == "https://mail.google.com/"
    assert opened == [["xdg-open", "https://mail.google.com/"]]


def test_unknown_name_suggests_website(opened):
    result = open_app({"name": "frobnicate"})
    assert not result.ok
    assert result.speech == (
        'I couldn\'t find an app called frobnicate. Say "open frobnicate dot com" for the website.'
    )
    assert opened == []


def test_file_extension_keeps_original_failure(opened):
    result = open_app({"name": "report.pdf"})
    assert not result.ok
    assert result.speech == "I couldn't find an app called report.pdf."
    assert opened == []
