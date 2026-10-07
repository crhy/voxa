from __future__ import annotations

import logging
import re
import shutil
import subprocess

import voxa.simulation
from voxa.agent.host import IN_FLATPAK, host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.spoken_text import format_dictation

log = logging.getLogger("voxa.agent.tools.typing")

LAST_REPLY: str = ""

_KEY_COMBO = re.compile(r"^[A-Za-z0-9_+]+$")

KEYS: dict[str, str] = {
    "enter": "Return",
    "tab": "Tab",
    "escape": "Escape",
    "backspace": "BackSpace",
    "select all": "ctrl+a",
    "copy": "ctrl+c",
    "paste": "ctrl+v",
    "undo": "ctrl+z",
    "save": "ctrl+s",
    "new tab": "ctrl+t",
    "close tab": "ctrl+w",
    "play pause": "XF86AudioPlay",
    "next track": "XF86AudioNext",
    "previous track": "XF86AudioPrev",
    "volume up": "XF86AudioRaiseVolume",
    "volume down": "XF86AudioLowerVolume",
    "mute": "XF86AudioMute",
    "full screen": "f",
    "space": "space",
    "page down": "Next",
    "page up": "Prior",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "refresh": "F5",
    "back": "alt+Left",
    "forward": "alt+Right",
    "find": "ctrl+f",
    "redo": "ctrl+shift+z",
    "cut": "ctrl+x",
    "zoom in": "ctrl+plus",
    "zoom out": "ctrl+minus",
}

_ALIASES: dict[str, str] = {
    "f": "full screen",
    "fullscreen": "full screen",
    "full-screen": "full screen",
    "return": "enter",
    "esc": "escape",
    "play": "play pause",
    "pause": "play pause",
    "play/pause": "play pause",
    "playpause": "play pause",
    "next": "next track",
    "previous": "previous track",
    "prev": "previous track",
    "vol up": "volume up",
    "volume up": "volume up",
    "vol down": "volume down",
    "volume down": "volume down",
}


def xdotool_available() -> bool:
    """True when the host can type with xdotool (it is always there for Flatpak)."""
    if IN_FLATPAK:
        return True
    return shutil.which("xdotool") is not None


def type_text(text: str) -> None:
    """Type text into the window that currently has focus, waiting for it to land."""
    if voxa.simulation.actions_simulated():
        log.info("simulated action: would type %r", text)
        return
    subprocess.run(
        host_command(["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text]),
        timeout=30,
        check=True,
    )


def erase(count: int) -> None:
    """Delete ``count`` characters with BackSpace in the focused window."""
    if count <= 0:
        return
    if voxa.simulation.actions_simulated():
        log.info("simulated action: would erase %d characters", count)
        return
    subprocess.run(
        host_command(["xdotool", "key", "--clearmodifiers", "--delay", "4", "--repeat", str(count), "BackSpace"]),
        timeout=30,
        check=True,
    )


def press_keys(combo: str) -> None:
    """Press a key combo such as ``ctrl+Return`` in the focused window."""
    if voxa.simulation.actions_simulated():
        log.info("simulated action: would press keys %s", combo)
        return
    if not _KEY_COMBO.fullmatch(combo):
        raise ValueError(f"invalid key combo: {combo}")
    subprocess.run(
        host_command(["xdotool", "key", "--clearmodifiers", combo]),
        timeout=30,
        check=True,
    )


def _type_text_handler(args: dict[str, str]) -> ToolResult:
    if not xdotool_available():
        return ToolResult.failure("I can't type here.")
    try:
        type_text(format_dictation(args["text"]))
    except Exception as exc:
        return ToolResult.failure("I can't type here.", detail=str(exc))
    return ToolResult.success("", detail=args["text"])


def _say_text_handler(args: dict[str, str]) -> ToolResult:
    return ToolResult.success(args["text"])


def _repeat_last_handler(args: dict[str, str]) -> ToolResult:
    if not LAST_REPLY:
        return ToolResult.success("I have not said anything yet.")
    return ToolResult.success(LAST_REPLY)


def _key_combo(key: str) -> str | None:
    """Resolve a spoken key name to an xdotool combo, accepting common aliases."""
    name = re.sub(r"\s+", " ", key.strip().casefold().replace("_", " "))
    return KEYS.get(_ALIASES.get(name, name))


def _press_key_handler(args: dict[str, str]) -> ToolResult:
    combo = _key_combo(args["key"])
    if combo is None:
        return ToolResult.failure("I don't know that key.")
    if not xdotool_available():
        return ToolResult.failure("I can't type here.")
    try:
        press_keys(combo)
    except Exception as exc:
        return ToolResult.failure("I can't type here.", detail=str(exc))
    return ToolResult.success("", detail=combo)


def _send_gmail_handler(args: dict[str, str]) -> ToolResult:
    if not xdotool_available():
        return ToolResult.failure("I can't type here.")
    try:
        press_keys("ctrl+Return")
    except Exception as exc:
        return ToolResult.failure("I can't type here.", detail=str(exc))
    return ToolResult.success("Sent.")


def typing_tools() -> list[Tool]:
    return [
        Tool(
            name="type_text",
            description="Type dictated text into the window that has focus.",
            parameters={"text": "the utterance to type"},
            risk=RiskLevel.REVERSIBLE,
            handler=_type_text_handler,
            required=("text",),
        ),
        Tool(
            name="say_text",
            description="Speak the given text aloud without typing anything.",
            parameters={"text": "the text to speak"},
            risk=RiskLevel.READ_ONLY,
            handler=_say_text_handler,
            required=("text",),
        ),
        Tool(
            name="repeat_last",
            description="Repeat the last thing Voxa said aloud.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_repeat_last_handler,
        ),
        Tool(
            name="press_key",
            description="Press a key in the focused window.",
            parameters={"key": "key name such as enter, copy, or volume up"},
            risk=RiskLevel.REVERSIBLE,
            handler=_press_key_handler,
            required=("key",),
        ),
        Tool(
            name="send_gmail",
            description="Send the drafted Gmail message with ctrl+Return.",
            parameters={},
            risk=RiskLevel.EXTERNAL_WRITE,
            handler=_send_gmail_handler,
        ),
    ]
