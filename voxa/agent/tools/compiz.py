"""Compiz desktop effects by voice: rotate the cube, zoom the desktop.

Compiz has no command for "rotate the cube"; it reacts to key bindings. This tool presses the same keys the user
would, with ``xdotool``, using the bindings from the user's Compiz configuration when they are set there and
Compiz's own defaults otherwise.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.compiz")

# action -> (ini section, option, Compiz's default binding in Compiz's own notation)
DEFAULT_BINDINGS: dict[str, tuple[str, str, str]] = {
    "cube_left": ("rotate", "rotate_left_key", "<Control><Alt>Left"),
    "cube_right": ("rotate", "rotate_right_key", "<Control><Alt>Right"),
    "zoom_in": ("ezoom", "zoom_in_key", "<Super>equal"),
    "zoom_out": ("ezoom", "zoom_out_key", "<Super>minus"),
}
CONFIG_FILES = (
    Path.home() / ".config" / "compiz" / "compizconfig" / "Default.ini",
    Path.home() / ".config" / "compiz-1" / "compizconfig" / "Default.ini",
)
_MODIFIERS = {"control": "ctrl", "ctrl": "ctrl", "alt": "alt", "shift": "shift", "super": "super", "primary": "ctrl"}
PAN_PIXELS = 500  # the zoomed view follows the pointer: moving the pointer pans it


def to_xdotool(binding: str) -> str | None:
    """``<Shift><Super>Up`` -> ``shift+super+Up``; None for an empty/disabled or mouse-button binding."""
    binding = binding.strip()
    if not binding or binding.lower() == "disabled" or "button" in binding.lower():
        return None
    modifiers = [_MODIFIERS.get(name.lower(), name.lower()) for name in re.findall(r"<([^>]+)>", binding)]
    key = re.sub(r"<[^>]+>", "", binding).strip()
    return "+".join([*modifiers, key]) if key else None


def read_bindings(text: str) -> dict[str, str]:
    """Bindings found in a compizconfig ini file, as action -> xdotool combo."""
    found: dict[str, str] = {}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        name = re.sub(r"^(?:as|s\d+)_", "", name)  # the ini prefixes options with "as_" or "s0_"
        for action, (wanted_section, option, _default) in DEFAULT_BINDINGS.items():
            if section == wanted_section and name == option:
                combo = to_xdotool(value)
                if combo:
                    found[action] = combo
    return found


def bindings(config_files=CONFIG_FILES) -> dict[str, str]:
    result = {action: to_xdotool(default) for action, (_s, _o, default) in DEFAULT_BINDINGS.items()}
    for path in config_files:
        try:
            result.update(read_bindings(Path(path).read_text()))
            break
        except OSError:
            continue
    return result


def _run(command: list[str]) -> bool:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return True
    try:
        done = subprocess.run(host_command(command), capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _press(combo: str, times: int = 1) -> bool:
    ok = True
    for _ in range(times):
        ok = _run(["xdotool", "key", "--clearmodifiers", combo]) and ok
    return ok


def compiz_control(args: dict[str, str]) -> ToolResult:
    action = args["action"]
    keys = bindings()
    plan: dict[str, tuple[str, int, str]] = {
        "cube_right": ("cube_right", 1, "Rotating right."),
        "cube_left": ("cube_left", 1, "Rotating left."),
        "zoom_in": ("zoom_in", 1, "Zooming in."),
        "zoom_in_more": ("zoom_in", 2, "Zooming in more."),
        "zoom_out": ("zoom_out", 1, "Zooming out."),
        "zoom_reset": ("zoom_out", 8, "Back to normal size."),
    }
    if action in plan:
        key, times, speech = plan[action]
        if _press(keys[key], times):
            return ToolResult.success(speech, detail=f"{keys[key]} x{times}")
        return ToolResult.failure("I could not send that to the desktop.", detail="xdotool failed")
    if action in ("zoom_left", "zoom_right"):
        distance = -PAN_PIXELS if action == "zoom_left" else PAN_PIXELS
        if _run(["xdotool", "mousemove_relative", "--", str(distance), "0"]):
            return ToolResult.success("Moving left." if distance < 0 else "Moving right.")
        return ToolResult.failure("I could not send that to the desktop.", detail="xdotool failed")
    return ToolResult.failure("I don't know that desktop effect.")


def compiz_tools() -> list[Tool]:
    return [
        Tool(
            name="compiz_control",
            description=(
                "Compiz desktop effects: rotate the desktop cube left or right, zoom the desktop in or out, "
                "and move the zoomed view left or right."
            ),
            parameters={
                "action": "one of cube_right, cube_left, zoom_in, zoom_in_more, zoom_out, zoom_reset, zoom_left, zoom_right",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=compiz_control,
            required=("action",),
        ),
    ]
