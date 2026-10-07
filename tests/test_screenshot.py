import subprocess

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import filemanage, screenshot

STAMP = "2026-10-07 10-45-12"
PICTURES = "/home/u/Pictures"
PATH = f"{PICTURES}/Screenshot {STAMP}.png"
NO_GRABBER = "I could not take a screenshot on this computer."


def _fake_run(installed, exits=None, empty=None, calls=None):
    exits = exits or {}
    empty = empty or set()

    def run(command, stdin=None, check=True):
        if command[0] == "date":
            return subprocess.CompletedProcess(command, 0, STAMP, "")
        if command[:4] == ["sh", "-c", 'command -v "$1"', "sh"]:
            program = command[4]
            return subprocess.CompletedProcess(command, 0 if program in installed else 1, "", "")
        if command[:2] == ["test", "-s"]:
            return subprocess.CompletedProcess(command, 0 if command[2] not in empty else 1, "", "")
        if calls is not None:
            calls.append(list(command))
        return subprocess.CompletedProcess(command, exits.get(command[0], 0), "", "")

    return run


def test_only_import_installed(monkeypatch):
    monkeypatch.setattr(filemanage, "_folder", lambda spoken: PICTURES)
    calls = []
    monkeypatch.setattr(screenshot, "_run", _fake_run({"import"}, calls=calls))
    result = screenshot.take_screenshot()
    assert calls == [["import", "-window", "root", PATH]]
    assert result.ok
    assert result.detail == PATH


def test_scrot_used_import_not_tried(monkeypatch):
    monkeypatch.setattr(filemanage, "_folder", lambda spoken: PICTURES)
    calls = []
    monkeypatch.setattr(screenshot, "_run", _fake_run({"scrot", "import"}, calls=calls))
    result = screenshot.take_screenshot()
    assert calls == [["scrot", PATH]]
    assert result.ok
    assert result.detail == PATH


def test_grabber_exits_one_next_tried(monkeypatch):
    monkeypatch.setattr(filemanage, "_folder", lambda spoken: PICTURES)
    calls = []
    monkeypatch.setattr(screenshot, "_run", _fake_run({"scrot", "import"}, exits={"scrot": 1}, calls=calls))
    result = screenshot.take_screenshot()
    assert calls == [["scrot", PATH], ["import", "-window", "root", PATH]]
    assert result.ok
    assert result.detail == PATH


def test_empty_file_next_tried(monkeypatch):
    monkeypatch.setattr(filemanage, "_folder", lambda spoken: PICTURES)
    calls = []
    monkeypatch.setattr(screenshot, "_run", _fake_run({"scrot", "import"}, empty={PATH}, calls=calls))
    result = screenshot.take_screenshot()
    assert ["import", "-window", "root", PATH] in calls
    assert not result.ok
    assert result.speech == NO_GRABBER


def test_nothing_installed(monkeypatch):
    monkeypatch.setattr(filemanage, "_folder", lambda spoken: PICTURES)
    monkeypatch.setattr(screenshot, "_run", _fake_run(set()))
    result = screenshot.take_screenshot()
    assert not result.ok
    assert result.speech == NO_GRABBER


def test_router_matches():
    assert route("take a screenshot") == ToolCall("take_screenshot", {})
    assert route("screenshot") == ToolCall("take_screenshot", {})
    assert route("grab a screen shot please") == ToolCall("take_screenshot", {})
    assert route("take a picture of my screen") == ToolCall("take_screenshot", {})


def test_router_rejects():
    assert route("show me pictures of screens") != ToolCall("take_screenshot", {})
