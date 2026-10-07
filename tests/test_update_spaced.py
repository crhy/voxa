"""Tests for the Spaced Update flow: pure logic, the tool, and the router."""
from __future__ import annotations

import subprocess
import sys

from voxa.agent.intents import route
from voxa.agent.sysupdate import (
    OS_DONE_WARN,
    parse_latest,
    parse_os_release,
    verdict,
    version_key,
)


def _module():
    return sys.modules["voxa.agent.tools.sysupdate"]


def test_version_key():
    assert version_key("v10.26.1") == (26, 10, 1)
    assert version_key("10.26.1") == (26, 10, 1)
    assert version_key("12.26") < version_key("1.27")
    assert version_key("10.26.0") < version_key("10.26.1")


def test_parse_os_release():
    assert parse_os_release('NAME="Spaced"\nVERSION_ID="10.26.1"\n') == "10.26.1"
    assert parse_os_release("VERSION_ID=10.26.1") == "10.26.1"
    assert parse_os_release("NAME=x\n") == ""


def test_parse_latest():
    assert parse_latest('{"tag_name": "v10.26.1"}') == "10.26.1"
    assert parse_latest('{"tag_name": "10.26.1"}') == "10.26.1"
    assert parse_latest("not json") == ""
    assert parse_latest("") == ""


def test_verdict_branches():
    ok, msg = verdict("Authentication cancelled", "", "10.26.1", "10.26.1")
    assert not ok and msg.startswith("The update was cancelled at the password prompt.")

    ok, msg = verdict("System update failed", "", "10.26.1", "10.26.1")
    assert not ok and msg.startswith("The system update failed.")

    ok, msg = verdict("", "", "10.26.1", "10.26.1")
    assert not ok and msg.startswith("The system update did not finish while I was watching.")

    ok, msg = verdict("Updated to Spaced Linux 10.26.1", "Updates installed", "10.26.0", "10.26.1")
    assert not ok and msg.startswith("The update ran, but this computer is on 10.26.0")

    ok, msg = verdict(OS_DONE_WARN[0], "Updates installed", "10.26.1", "10.26.1")
    assert not ok and msg.startswith("The update finished, but some packages are still pending.")

    ok, msg = verdict("Updated to Spaced Linux 10.26.1", "Update failed", "10.26.1", "10.26.1")
    assert not ok and msg.startswith("Spaced Linux is on 10.26.1, the latest release, but the app updates did not complete.")

    ok, msg = verdict("Updated to Spaced Linux 10.26.1", "You\u2019re up to date", "10.26.1", "10.26.1")
    assert ok and msg.startswith("Spaced Linux is up to date: version 10.26.1, the latest release.")

    ok, msg = verdict("Updated to Spaced Linux 10.26.1", "You\u2019re up to date", "10.26.1", "")
    assert ok and msg.startswith("The updates finished. I could not check the latest release number online")


class FakeUI:
    def __init__(self, answers):
        self.pressed = []
        self.answers = list(answers)
        self.wait_for_any_calls = 0

    def press(self, app, label, role=""):
        self.pressed.append(label)
        return {"ok": True, "pressed": {label: True}}

    def wait(self, app, label, seconds):
        return {"ok": True}

    def wait_for_any(self, app, labels, seconds, poll=3.0):
        self.wait_for_any_calls += 1
        return self.answers.pop(0) if self.answers else ""


def _fake_run(os_release, present=True):
    def run(command, timeout, check=False):
        if command[:2] == ["sh", "-c"]:
            return subprocess.CompletedProcess(command, 0 if present else 1, "", "")
        return subprocess.CompletedProcess(command, 0, os_release, "")

    return run


def _patch(monkeypatch, module, *, ui, run, spawn, fetch, sleep):
    monkeypatch.setattr(module, "ui", ui)
    monkeypatch.setattr(module, "_run", run)
    monkeypatch.setattr(module, "spawn", spawn)
    monkeypatch.setattr(module, "fetch", fetch)
    monkeypatch.setattr(module, "_sleep", sleep)


def test_tool_happy_path(monkeypatch):
    module = _module()
    ui = FakeUI(["Updated to Spaced Linux", "You\u2019re up to date"])
    _patch(
        monkeypatch,
        module,
        ui=ui,
        run=_fake_run('VERSION_ID="10.26.1"\n'),
        spawn=lambda cmd: None,
        fetch=lambda url: '{"tag_name": "v10.26.1"}',
        sleep=lambda s: None,
    )
    result = module.update_system({})
    assert result.ok
    assert ui.pressed == ["OS Release", "Check Release", "Update System", "Updates", "Check for Updates"]


def test_tool_with_app_updates(monkeypatch):
    module = _module()
    ui = FakeUI(["Updated to Spaced Linux", "available", "Updates installed"])
    _patch(
        monkeypatch,
        module,
        ui=ui,
        run=_fake_run('VERSION_ID="10.26.1"\n'),
        spawn=lambda cmd: None,
        fetch=lambda url: '{"tag_name": "v10.26.1"}',
        sleep=lambda s: None,
    )
    result = module.update_system({})
    assert result.ok
    assert ui.pressed == [
        "OS Release",
        "Check Release",
        "Update System",
        "Updates",
        "Check for Updates",
        "Select all",
        "Install Selected",
    ]


def test_tool_cancelled_at_password(monkeypatch):
    module = _module()
    ui = FakeUI(["Authentication cancelled"])
    _patch(
        monkeypatch,
        module,
        ui=ui,
        run=_fake_run('VERSION_ID="10.26.1"\n'),
        spawn=lambda cmd: None,
        fetch=lambda url: '{"tag_name": "v10.26.1"}',
        sleep=lambda s: None,
    )
    result = module.update_system({})
    assert not result.ok
    assert result.speech.startswith("The update was cancelled at the password prompt.")
    assert "Updates" not in ui.pressed


def test_tool_installed_behind(monkeypatch):
    module = _module()
    ui = FakeUI(["Updated to Spaced Linux 10.26.0", "You\u2019re up to date"])
    _patch(
        monkeypatch,
        module,
        ui=ui,
        run=_fake_run('VERSION_ID="10.26.0"\n'),
        spawn=lambda cmd: None,
        fetch=lambda url: '{"tag_name": "v10.26.1"}',
        sleep=lambda s: None,
    )
    result = module.update_system({})
    assert not result.ok


def test_tool_fetch_raises(monkeypatch):
    module = _module()

    def boom(url):
        raise OSError("offline")

    ui = FakeUI(["Updated to Spaced Linux", "You\u2019re up to date"])
    _patch(
        monkeypatch,
        module,
        ui=ui,
        run=_fake_run('VERSION_ID="10.26.1"\n'),
        spawn=lambda cmd: None,
        fetch=boom,
        sleep=lambda s: None,
    )
    result = module.update_system({})
    assert result.ok
    assert result.speech.startswith("The updates finished. I could not check the latest release number online")


def test_tool_update_system_press_fails(monkeypatch):
    module = _module()

    class PressFail(FakeUI):
        def press(self, app, label, role=""):
            self.pressed.append(label)
            if label == "Update System":
                return {"ok": False, "error": "disabled"}
            return {"ok": True, "pressed": {label: True}}

    ui = PressFail([])
    _patch(
        monkeypatch,
        module,
        ui=ui,
        run=_fake_run('VERSION_ID="10.26.1"\n'),
        spawn=lambda cmd: None,
        fetch=lambda url: '{"tag_name": "v10.26.1"}',
        sleep=lambda s: None,
    )
    result = module.update_system({})
    assert not result.ok
    assert result.speech == "I could not press Update System in Spaced Update."


def test_tool_missing(monkeypatch):
    module = _module()
    ui = FakeUI([])
    _patch(
        monkeypatch,
        module,
        ui=ui,
        run=_fake_run("", present=False),
        spawn=lambda cmd: None,
        fetch=lambda url: "{}",
        sleep=lambda s: None,
    )
    result = module.update_system({})
    assert not result.ok
    assert result.speech == "Spaced Update is not installed on this computer."


def test_waits_use_wait_for_any(monkeypatch):
    module = _module()
    ui = FakeUI(["Updated to Spaced Linux", "You\u2019re up to date"])
    _patch(
        monkeypatch,
        module,
        ui=ui,
        run=_fake_run('VERSION_ID="10.26.1"\n'),
        spawn=lambda cmd: None,
        fetch=lambda url: '{"tag_name": "v10.26.1"}',
        sleep=lambda s: None,
    )
    module.update_system({})
    assert ui.wait_for_any_calls >= 2


def test_router_phrases():
    for phrase in (
        "update spaced linux",
        "upgrade the system",
        "install all the latest updates",
        "open spaced update and install all the latest updates",
    ):
        call = route(phrase)
        assert call is not None and call.tool == "update_system"
    for phrase in ("am i up to date", "what version of spaced linux am i on"):
        call = route(phrase)
        assert call is not None and call.tool == "check_system_version"
    for phrase in ("open spaced update", "update the document"):
        call = route(phrase)
        assert call is None or call.tool not in {"update_system", "check_system_version"}
