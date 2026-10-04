from __future__ import annotations

from pathlib import Path

from voxa.theme import HOST_THEME_ENV, host_theme_is_dark, release_forced_theme


def test_release_forced_theme_moves_value():
    env = {"GTK_THEME": "Spaced-Win11-Dark", "HOME": "/root"}
    value = release_forced_theme(env)
    assert value == "Spaced-Win11-Dark"
    assert "GTK_THEME" not in env
    assert env[HOST_THEME_ENV] == "Spaced-Win11-Dark"
    assert env["HOME"] == "/root"


def test_release_forced_theme_noop_when_unset():
    env = {"HOME": "/root"}
    assert release_forced_theme(env) == ""
    assert env == {"HOME": "/root"}


def test_release_forced_theme_noop_when_empty():
    env = {"GTK_THEME": ""}
    assert release_forced_theme(env) == ""
    assert env == {"GTK_THEME": ""}


def test_host_theme_is_dark():
    assert host_theme_is_dark({HOST_THEME_ENV: "Spaced-Win11-Dark"})
    assert host_theme_is_dark({HOST_THEME_ENV: "Adwaita:dark"})
    assert host_theme_is_dark({HOST_THEME_ENV: "adwaita-DARK"})
    assert not host_theme_is_dark({HOST_THEME_ENV: "Spaced-Win11"})
    assert not host_theme_is_dark({})
    assert not host_theme_is_dark({HOST_THEME_ENV: ""})


def test_application_releases_theme_before_importing_gi():
    source = (Path(__file__).resolve().parent.parent / "voxa" / "application.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    gi_line = next(i for i, line in enumerate(lines) if "import gi" in line)
    call_line = next(i for i, line in enumerate(lines) if "release_forced_theme()" in line)
    assert call_line < gi_line
