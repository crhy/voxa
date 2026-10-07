from __future__ import annotations

import subprocess

import pytest

from voxa.agent import ui
from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import default_registry
from voxa.agent.tools.uicontrol import list_buttons, press_button, read_window, uicontrol_tools


def _fake_proc(stdout):
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _runner_ok(stdout, seen):
    def runner(cmd, **kw):
        seen.append(cmd)
        return _fake_proc(stdout)

    return runner


def _runner_timeout(seen):
    def runner(cmd, **kw):
        seen.append(cmd)
        raise subprocess.TimeoutExpired(cmd, 0)

    return runner


def test_call_parses_last_json_line():
    seen = []
    runner = _runner_ok('noise\n{"ok": true, "pressed": {"name": "Save"}}', seen)
    result = ui.call("press", "GIMP", "Save", runner=runner)
    assert result == {"ok": True, "pressed": {"name": "Save"}}
    assert seen[0] == ["python3", "-", "press", "GIMP", "Save"]


def test_call_empty_output():
    seen = []
    runner = _runner_ok("   \n\n", seen)
    result = ui.call("apps", runner=runner)
    assert result == {"ok": False, "error": "no output"}


def test_call_bad_json():
    seen = []
    runner = _runner_ok("not json at all", seen)
    result = ui.call("apps", runner=runner)
    assert result == {"ok": False, "error": "bad JSON"}


def test_call_timeout():
    seen = []
    runner = _runner_timeout(seen)
    result = ui.call("apps", runner=runner)
    assert result == {"ok": False, "error": "helper failed"}


def test_call_simulated(monkeypatch):
    monkeypatch.setattr("voxa.simulation.actions_simulated", lambda: True)
    result = ui.call("apps")
    assert result == {"ok": True, "simulated": True}


def test_wait_for_any_finds_label(monkeypatch):
    lines = iter([["nothing here"], ["please SAVE the file"], ["x"]])
    monkeypatch.setattr(ui, "text", lambda app, **kw: next(lines))
    monkeypatch.setattr(ui, "apps", lambda **kw: [])
    clock = iter(range(0, 100)).__next__
    result = ui.wait_for_any("GIMP", ("save", "open"), seconds=10, poll=1, sleep=lambda s: None, clock=clock)
    assert result == "save"


def test_wait_for_any_times_out(monkeypatch):
    monkeypatch.setattr(ui, "text", lambda app, **kw: ["nothing"])
    monkeypatch.setattr(ui, "apps", lambda **kw: [])
    clock = iter(range(0, 100)).__next__
    result = ui.wait_for_any("GIMP", ("save",), seconds=3, poll=1, sleep=lambda s: None, clock=clock)
    assert result == ""


def test_wait_for_any_announces_password(monkeypatch):
    spoken = []
    monkeypatch.setattr(ui, "say", lambda text: spoken.append(text))
    monkeypatch.setattr(ui, "apps", lambda **kw: [{"name": "polkit-gtk", "windows": ["Authentication"]}])
    monkeypatch.setattr(ui, "text", lambda app, **kw: ["nothing"])
    clock = iter(range(0, 100)).__next__
    result = ui.wait_for_any("GIMP", ("save",), seconds=3, poll=1, sleep=lambda s: None, clock=clock)
    assert result == ""
    assert spoken == [ui.PASSWORD_NOTICE]


def test_press_button_refused():
    result = press_button({"label": "delete"})
    assert not result.ok
    assert result.speech == "I will not press that by voice. Please do it yourself."


def test_press_button_refused_case(monkeypatch):
    result = press_button({"label": "  DELETE ALL "})
    assert not result.ok
    assert result.speech == "I will not press that by voice. Please do it yourself."


def test_press_button_ok(monkeypatch):
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "GIMP")
    monkeypatch.setattr(ui, "press", lambda app, label, **kw: {"ok": True, "pressed": {"name": "Save"}})
    result = press_button({"label": "Save"})
    assert result.ok
    assert result.speech == "Pressed Save."


def test_press_button_ok_falls_back_to_label(monkeypatch):
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "GIMP")
    monkeypatch.setattr(ui, "press", lambda app, label, **kw: {"ok": True})
    result = press_button({"label": "Open"})
    assert result.ok
    assert result.speech == "Pressed Open."


def test_press_button_disabled(monkeypatch):
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "GIMP")
    monkeypatch.setattr(ui, "press", lambda app, label, **kw: {"ok": False, "error": "disabled"})
    result = press_button({"label": "Save"})
    assert not result.ok
    assert result.speech == "Save is greyed out right now."


def test_press_button_program_not_found(monkeypatch):
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "GIMP")
    monkeypatch.setattr(ui, "press", lambda app, label, **kw: {"ok": False, "error": "program not found"})
    result = press_button({"label": "Save"})
    assert not result.ok
    assert result.speech == "I could not find the program GIMP."


def test_press_button_not_found_lists(monkeypatch):
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "GIMP")
    monkeypatch.setattr(ui, "press", lambda app, label, **kw: {"ok": False, "error": "not found"})
    monkeypatch.setattr(
        ui,
        "items",
        lambda app, *roles, **kw: [
            {"name": "Save", "showing": True, "enabled": True},
            {"name": "Cancel", "showing": True, "enabled": True},
            {"name": "Hidden", "showing": False, "enabled": True},
        ],
    )
    result = press_button({"label": "Nope"})
    assert not result.ok
    assert result.speech == "I could not find a button called Nope. I can see: Save, Cancel."


def test_press_button_no_app(monkeypatch):
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "")
    result = press_button({"label": "Save"})
    assert not result.ok
    assert result.speech == "I cannot tell which window is in front."


def test_press_button_other_error(monkeypatch):
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "GIMP")
    monkeypatch.setattr(ui, "press", lambda app, label, **kw: {"ok": False, "error": "weird"})
    result = press_button({"label": "Save"})
    assert not result.ok
    assert result.speech == "I could not reach that program's buttons."


def test_list_buttons_empty(monkeypatch):
    monkeypatch.setattr(ui, "active_window_title", lambda **kw: "GIMP")
    monkeypatch.setattr(ui, "items", lambda app, *roles, **kw: [])
    result = list_buttons({})
    assert not result.ok
    assert result.speech == "I do not see any buttons there."


def test_list_buttons_one(monkeypatch):
    monkeypatch.setattr(
        ui,
        "items",
        lambda app, *roles, **kw: [{"name": "Save", "role": "push button", "showing": True, "enabled": True}],
    )
    result = list_buttons({"app": "GIMP"})
    assert result.ok
    assert result.speech == "I can see 1 buttons: Save."


def test_list_buttons_three(monkeypatch):
    monkeypatch.setattr(
        ui,
        "items",
        lambda app, *roles, **kw: [
            {"name": "Save", "role": "push button", "showing": True, "enabled": True},
            {"name": "Cancel", "role": "push button", "showing": True, "enabled": True},
            {"name": "Open", "role": "push button", "showing": True, "enabled": True},
            {"name": "Close", "role": "push button", "showing": True, "enabled": True},
            {"name": "NotAButton", "role": "button", "showing": True, "enabled": True},
            {"name": "Off", "role": "push button", "showing": False, "enabled": True},
        ],
    )
    result = list_buttons({"app": "GIMP"})
    assert result.ok
    assert result.speech == "I can see 3 buttons: Save, Cancel and Open."


def test_list_buttons_many(monkeypatch):
    items = [{"name": f"B{i}", "role": "push button", "showing": True, "enabled": True} for i in range(12)]
    monkeypatch.setattr(ui, "items", lambda app, *roles, **kw: items)
    result = list_buttons({"app": "GIMP"})
    assert result.ok
    assert result.speech == (
        "I can see 12 buttons: B0, B1, B2, B3, B4, B5, B6, B7, B8, B9 and 2 more."
    )


def test_read_window_empty(monkeypatch):
    monkeypatch.setattr(ui, "text", lambda app, **kw: [])
    result = read_window({})
    assert not result.ok
    assert result.speech == "I cannot read anything in that window."


def test_read_window_joins(monkeypatch):
    monkeypatch.setattr(ui, "text", lambda app, **kw: ["Hello", "x", "World"])
    result = read_window({"app": "GIMP"})
    assert result.ok
    assert result.speech == "Hello. World"


@pytest.mark.parametrize(
    "apps_value, expected",
    [
        ([{"name": "polkit-gtk", "windows": ["Authentication"]}], True),
        ([{"name": "GIMP", "windows": ["Authentication required"]}], True),
        ([{"name": "GIMP", "windows": ["Foo"]}], False),
        ([], False),
    ],
)
def test_password_prompt_showing(apps_value, expected, monkeypatch):
    monkeypatch.setattr(ui, "apps", lambda **kw: apps_value)
    assert ui.password_prompt_showing() is expected


def test_announce_password_once(monkeypatch):
    spoken = []
    monkeypatch.setattr(ui, "say", lambda text: spoken.append(text))
    monkeypatch.setattr(ui, "apps", lambda **kw: [{"name": "pkexec", "windows": ["Confirm"]}])
    state = {}
    assert ui.announce_password_once(state) is True
    assert spoken == [ui.PASSWORD_NOTICE]
    assert ui.announce_password_once(state) is False


def test_announce_password_no_prompt(monkeypatch):
    monkeypatch.setattr(ui, "apps", lambda **kw: [])
    state = {}
    assert ui.announce_password_once(state) is False


@pytest.mark.parametrize(
    "text_value, expected",
    [
        ("press the Save button", ToolCall("click_on", {"text": "Save"})),
        ("press the OK button", ToolCall("click_on", {"text": "OK"})),
        ("press enter", ToolCall("press_key", {"key": "enter"})),
        ("press space", ToolCall("press_key", {"key": "space"})),
        ("press control s", None),
        ("click History", ToolCall("click_on", {"text": "History"})),
        ("click on Search", ToolCall("click_on", {"text": "Search"})),
        ("press Save button", ToolCall("press_button", {"label": "Save"})),
        ("press Cancel in the editor", ToolCall("press_button", {"label": "Cancel", "app": "editor"})),
        ("what buttons can I press", ToolCall("list_buttons", {"app": ""})),
        ("list the buttons in Firefox", ToolCall("list_buttons", {"app": "Firefox"})),
        ("read the window", ToolCall("read_window", {})),
        ("what does this window say", ToolCall("read_window", {})),
    ],
)
def test_routing(text_value, expected):
    assert route(text_value) == expected


def test_uicontrol_tools_in_registry():
    names = default_registry().names()
    for tool in uicontrol_tools():
        assert tool.name in names
