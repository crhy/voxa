"""Compiz effects are driven by pressing the user's real key bindings."""

from __future__ import annotations

from voxa.agent.tools import compiz

INI = """[core]
as_active_plugins = core;cube;rotate;ezoom

[ezoom]
as_zoom_in_button = <Shift><Super>Button4
as_zoom_in_key = <Shift><Super>Up
as_zoom_out_key = <Shift><Super>Down
"""


def test_binding_notation_is_translated() -> None:
    assert compiz.to_xdotool("<Shift><Super>Up") == "shift+super+Up"
    assert compiz.to_xdotool("<Control><Alt>Left") == "ctrl+alt+Left"
    assert compiz.to_xdotool("<Shift><Super>Button4") is None  # mouse bindings cannot be pressed as keys
    assert compiz.to_xdotool("Disabled") is None and compiz.to_xdotool("") is None


def test_bindings_come_from_the_users_config_then_defaults(tmp_path) -> None:
    config = tmp_path / "Default.ini"
    config.write_text(INI)
    keys = compiz.bindings((config,))
    assert keys["zoom_in"] == "shift+super+Up" and keys["zoom_out"] == "shift+super+Down"
    assert keys["cube_right"] == "ctrl+alt+Right"  # not in the file: Compiz's default
    assert compiz.bindings((tmp_path / "missing.ini",))["zoom_in"] == "super+equal"


def test_actions_press_the_right_keys(monkeypatch, tmp_path) -> None:
    config = tmp_path / "Default.ini"
    config.write_text(INI)
    monkeypatch.setattr(compiz, "CONFIG_FILES", (config,))
    sent: list[list[str]] = []
    monkeypatch.setattr(compiz, "_run", lambda command: sent.append(command) or True)

    assert compiz.compiz_control({"action": "cube_left"}).ok
    assert sent[-1] == ["xdotool", "key", "--clearmodifiers", "ctrl+alt+Left"]
    sent.clear()
    assert compiz.compiz_control({"action": "zoom_in_more"}).ok
    assert sent == [["xdotool", "key", "--clearmodifiers", "shift+super+Up"]] * 2
    sent.clear()
    assert compiz.compiz_control({"action": "zoom_reset"}).ok and len(sent) == 8
    sent.clear()
    assert compiz.compiz_control({"action": "zoom_left"}).ok
    assert sent == [["xdotool", "mousemove_relative", "--", "-500", "0"]]


def test_failure_is_reported_not_claimed(monkeypatch) -> None:
    monkeypatch.setattr(compiz, "_run", lambda command: False)
    result = compiz.compiz_control({"action": "zoom_in"})
    assert not result.ok and "could not" in result.speech
    assert not compiz.compiz_control({"action": "spin"}).ok
