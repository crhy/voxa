from __future__ import annotations

import pytest

from voxa.agent.policy import RiskLevel
from voxa.agent.tools import default_registry
from voxa.agent.tools import typing as typing_mod


@pytest.fixture
def argv(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return None

    monkeypatch.setattr(typing_mod.subprocess, "run", fake_run)
    # The build machines have no xdotool; these tests are about the commands Voxa would send.
    monkeypatch.setattr(typing_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    return calls


def test_registry_has_typing_tools() -> None:
    names = default_registry().names()
    assert {"type_text", "press_key", "send_gmail"} <= set(names)


def test_type_text_types_formatted_dictation(argv) -> None:
    result = default_registry().call("type_text", {"text": "hello comma world period new line thanks"})
    assert result.ok
    assert result.speech == ""
    assert argv == [["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", "hello, world.\nthanks "]]


def test_press_key_maps_names(argv) -> None:
    registry = default_registry()
    assert registry.call("press_key", {"key": "Enter"}).ok
    assert registry.call("press_key", {"key": "select all"}).ok
    assert registry.call("press_key", {"key": "volume up"}).ok
    assert argv == [
        ["xdotool", "key", "--clearmodifiers", "Return"],
        ["xdotool", "key", "--clearmodifiers", "ctrl+a"],
        ["xdotool", "key", "--clearmodifiers", "XF86AudioRaiseVolume"],
    ]


def test_press_key_rejects_unknown_key(argv) -> None:
    result = default_registry().call("press_key", {"key": "rm -rf"})
    assert not result.ok
    assert result.speech == "I don't know that key."
    assert argv == []


@pytest.mark.parametrize(
    ("spoken", "combo"),
    [
        ("f", "f"),
        ("fullscreen", "f"),
        ("full-screen", "f"),
        ("return", "Return"),
        ("esc", "Escape"),
        ("play", "XF86AudioPlay"),
        ("playpause", "XF86AudioPlay"),
        ("next", "XF86AudioNext"),
        ("prev", "XF86AudioPrev"),
        ("vol up", "XF86AudioRaiseVolume"),
        ("volume_up", "XF86AudioRaiseVolume"),
        ("vol down", "XF86AudioLowerVolume"),
        ("space", "space"),
        ("page down", "Next"),
        ("page up", "Prior"),
        ("up", "Up"),
        ("down", "Down"),
        ("left", "Left"),
        ("right", "Right"),
        ("refresh", "F5"),
        ("back", "alt+Left"),
        ("forward", "alt+Right"),
        ("find", "ctrl+f"),
        ("redo", "ctrl+shift+z"),
        ("cut", "ctrl+x"),
        ("zoom in", "ctrl+plus"),
        ("zoom out", "ctrl+minus"),
    ],
)
def test_press_key_accepts_forgiving_names(argv, spoken: str, combo: str) -> None:
    assert default_registry().call("press_key", {"key": spoken}).ok
    assert argv == [["xdotool", "key", "--clearmodifiers", combo]]


def test_press_keys_rejects_bad_combo(argv) -> None:
    with pytest.raises(ValueError):
        typing_mod.press_keys("rm -rf")
    assert argv == []


def test_send_gmail_is_external_write(argv) -> None:
    tool = default_registry().get("send_gmail")
    assert tool.risk == RiskLevel.EXTERNAL_WRITE
    result = default_registry().call("send_gmail", {})
    assert result.ok
    assert result.speech == "Sent."
    assert argv == [["xdotool", "key", "--clearmodifiers", "ctrl+Return"]]


def test_simulation_runs_nothing(argv, monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    typing_mod.type_text("hello")
    typing_mod.press_keys("ctrl+s")
    assert argv == []


def test_missing_xdotool_fails(argv, monkeypatch) -> None:
    monkeypatch.setattr(typing_mod, "IN_FLATPAK", False)
    monkeypatch.setattr(typing_mod.shutil, "which", lambda name: None)
    assert not typing_mod.xdotool_available()
    result = default_registry().call("type_text", {"text": "hello"})
    assert not result.ok
    assert result.speech == "I can't type here."
    assert argv == []


def test_xdotool_available_inside_flatpak(monkeypatch) -> None:
    monkeypatch.setattr(typing_mod, "IN_FLATPAK", True)
    monkeypatch.setattr(typing_mod.shutil, "which", lambda name: None)
    assert typing_mod.xdotool_available()


def test_erase_repeats_backspace(argv) -> None:
    typing_mod.erase(3)
    assert argv == [
        ["xdotool", "key", "--clearmodifiers", "--delay", "4", "--repeat", "3", "BackSpace"]
    ]


def test_erase_with_zero_or_negative_count_runs_nothing(argv) -> None:
    typing_mod.erase(0)
    typing_mod.erase(-4)
    assert argv == []


def test_erase_is_simulated_like_other_actions(argv, monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    typing_mod.erase(12)
    assert argv == []
