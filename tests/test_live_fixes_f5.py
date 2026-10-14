from __future__ import annotations

import subprocess

from voxa.agent.intents import route
from voxa.agent.tools import filemanage


def test_focus_start_at_and_a():
    for phrase in ("start a 25 minute focus session", "start at 25 minute focus session"):
        call = route(phrase)
        assert call is not None
        assert call.tool == "focus_start"
        assert call.args["duration"] == "25 minute"


def test_glued_mail_address_routes():
    call = route("Sam's email is samatgmail.com")
    assert call is not None
    assert call.tool == "set_contact_email"
    assert call.args == {"name": "Sam", "address": "sam@gmail.com"}


def test_glued_an_variant():
    call = route("Sam's email is samangmail.com")
    assert call is not None
    assert call.tool == "set_contact_email"
    assert call.args == {"name": "Sam", "address": "sam@gmail.com"}


def test_spoken_mail_address_still_works():
    call = route("Sam's email is sam at gmail dot com")
    assert call is not None
    assert call.tool == "set_contact_email"
    assert call.args == {"name": "Sam", "address": "sam at gmail dot com"}


def test_make_folder_already_exists(monkeypatch):
    calls: list[list[str]] = []

    def run(command, stdin=None, check=True):
        argv = list(command)
        calls.append(argv)
        if argv[0] == "sh":
            return subprocess.CompletedProcess(argv, 0, "/home/u", "")
        if argv[0] == "xdg-user-dir":
            return subprocess.CompletedProcess(argv, 0, "/home/u", "")
        if argv[0] == "test":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(filemanage, "_run", run)
    result = filemanage.make_folder({"name": "taxes", "folder": "documents"})
    assert result.ok
    assert result.speech == "You already have a folder called Taxes in documents."
    assert ["mkdir", "--", "/home/u/Documents/Taxes"] not in calls


def test_make_folder_failure_hides_raw_text(monkeypatch):
    def run(command, stdin=None, check=True):
        argv = list(command)
        if argv[0] == "sh":
            return subprocess.CompletedProcess(argv, 0, "/home/u", "")
        if argv[0] == "xdg-user-dir":
            return subprocess.CompletedProcess(argv, 0, "/home/u", "")
        if argv[0] == "test":
            return subprocess.CompletedProcess(argv, 1, "", "")
        if argv[0] == "mkdir":
            raise subprocess.CalledProcessError(1, argv, stderr=b"mkdir: cannot create directory: nope")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(filemanage, "_run", run)
    result = filemanage.make_folder({"name": "taxes", "folder": "documents"})
    assert not result.ok
    assert result.speech == "I couldn't make that folder."
    assert "cannot create directory" in result.detail


def test_routing_preserved():
    assert route("what's on my to-do list").tool == "list_read"
    assert route("what time is it").tool == "tell_time"
    assert route("what is the capital of France") is None
    assert route("open the calculator").tool == "open_app"
