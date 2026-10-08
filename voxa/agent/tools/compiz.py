"""Compiz desktop effects by voice: rotate the cube, zoom the desktop.

Compiz has no command for "rotate the cube"; it reacts to key bindings. This tool presses the same keys the user
would, with ``xdotool``, using the bindings from the user's Compiz configuration when they are set there and
Compiz's own defaults otherwise.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
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
            text = _host_read(path)  # inside the Flatpak the user's Compiz settings are only visible on the host
            if text:
                result.update(read_bindings(text))
                break
    return result


def _host_read(path) -> str:
    if simulation.actions_simulated():
        return ""
    try:
        done = subprocess.run(host_command(["cat", str(path)]), capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


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


SPIN_SECONDS = 1.2   # one face of the cube every 1.2 s: a medium rate (fallback spin)
SPIN_MAX_FACES = 200
SPIN_GRAB = True  # the grab spin below; False falls back to pressing the rotate keys
SPIN_STEP_MS = 40  # one pointer step every 40 ms
SPIN_TURN_SECONDS = 20.0  # one full turn of the cube takes this long
SPIN_MAX_SECONDS = 300.0  # five minutes, then the spin stops itself
SPIN_EDGE_MARGIN = 150  # let go and grab again when the pointer gets this close to a screen edge
_spin_stop = threading.Event()
_spin_thread: threading.Thread | None = None


def faces_from_wmctrl(listing: str) -> int:
    """How many faces the cube has, from ``wmctrl -d`` ("DG: 7680x1080 ... WA: 0,0 1920x1080" -> 4)."""
    match = re.search(r"DG:\s*(\d+)x\d+.*?WA:\s*\S+\s+(\d+)x\d+", listing)
    if match and int(match.group(2)) > 0:
        faces = round(int(match.group(1)) / int(match.group(2)))
        if faces >= 2:
            return faces
    lines = [line for line in listing.splitlines() if line.strip()]
    return len(lines) if len(lines) >= 2 else 4


def desktop_size(listing: str) -> tuple[int, int, int]:
    """(desktop width, screen width, screen height) from ``wmctrl -d``; zeros when unknown."""
    match = re.search(r"DG:\s*(\d+)x\d+.*?WA:\s*\S+\s+(\d+)x(\d+)", listing)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return (0, 0, 0)


def spin_dx(desktop_width: int, turn_seconds: float = SPIN_TURN_SECONDS, step_ms: int = SPIN_STEP_MS) -> int:
    """Pixels per pointer step so the pointer crosses the whole desktop once per turn."""
    steps = max(1, int(turn_seconds * 1000 // step_ms))
    return max(1, round(desktop_width / steps))


def way_back(turned: int, faces: int) -> tuple[str, int]:
    """The shortest way home after ``turned`` right turns on a cube of ``faces``: (direction, presses)."""
    offset = turned % max(faces, 1)
    if offset == 0:
        return ("cube_left", 0)
    return ("cube_left", offset) if offset <= faces - offset else ("cube_right", faces - offset)


def spinning() -> bool:
    return _spin_thread is not None and _spin_thread.is_alive()


def stop_spin(wait: float = 0.0) -> bool:
    """Stop a running spin (it turns back to where it started). True when one was running."""
    if not spinning():
        return False
    _spin_stop.set()
    if wait:
        _spin_thread.join(wait)
    return True


def _spin(keys: dict[str, str], faces: int, sleep=None) -> None:
    turned = 0
    while turned < SPIN_MAX_FACES and not _spin_stop.is_set():
        if not _press(keys["cube_right"]):
            break
        turned += 1
        if _spin_stop.wait(SPIN_SECONDS):
            break
    direction, presses = way_back(turned, faces)
    if presses:
        _press(keys[direction], presses)


def _mouse_position() -> tuple[int, int] | None:
    if simulation.actions_simulated():
        return None
    try:
        done = subprocess.run(host_command(["xdotool", "getmouseposition"]), capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    parts = done.stdout.split()
    if len(parts) >= 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return None
    return None


def _spin_grab(start: tuple[int, int], desktop_w: int, screen_w: int, screen_h: int, direction: int) -> None:
    """Grab the cube the way the user does: hold ctrl+alt, drag with mouse button 1, step the pointer sideways."""
    dx = spin_dx(desktop_w) * direction
    x, y = start
    keys_down = False
    button_down = False

    def release() -> None:
        nonlocal keys_down, button_down
        if button_down:
            _run(["xdotool", "mouseup", "1"])
            button_down = False
        if keys_down:
            _run(["xdotool", "keyup", "ctrl+alt"])
            keys_down = False

    def grab() -> bool:
        nonlocal keys_down, button_down
        if _run(["xdotool", "keydown", "ctrl+alt"]):
            keys_down = True
            if _run(["xdotool", "mousedown", "1"]):
                button_down = True
                return True
            release()
        return False

    try:
        if not grab():
            return
        elapsed = 0.0
        while elapsed < SPIN_MAX_SECONDS and not _spin_stop.is_set():
            if (dx > 0 and x + SPIN_EDGE_MARGIN >= screen_w) or (dx < 0 and x - SPIN_EDGE_MARGIN <= 0):
                release()
                x, y = screen_w // 2, screen_h // 2
                if not _run(["xdotool", "mousemove", str(x), str(y)]):
                    break
                if not grab():
                    break
            if not _run(["xdotool", "mousemove_relative", "--", str(dx), "0"]):
                break
            x += dx
            if _spin_stop.wait(SPIN_STEP_MS / 1000):
                break
            elapsed += SPIN_STEP_MS / 1000
    finally:
        release()
        _run(["xdotool", "mousemove", str(start[0]), str(start[1])])


def start_spin(direction: str = "right") -> ToolResult:
    global _spin_thread
    if spinning():
        return ToolResult.success("Already spinning. Say stop to finish.")
    keys = bindings()
    try:
        listing = subprocess.run(
            host_command(["wmctrl", "-d"]), capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        listing = ""
    _spin_stop.clear()
    if SPIN_GRAB:
        desktop_w, screen_w, screen_h = desktop_size(listing)
        start = _mouse_position()
        if desktop_w and screen_w and start:
            _spin_thread = threading.Thread(
                target=_spin_grab,
                args=(start, desktop_w, screen_w, screen_h, -1 if direction == "left" else 1),
                name="cube-spin",
                daemon=True,
            )
            _spin_thread.start()
            return ToolResult.success("Spinning. Say stop when you have had enough.")
    _spin_thread = threading.Thread(
        target=_spin, args=(keys, faces_from_wmctrl(listing)), name="cube-spin", daemon=True
    )
    _spin_thread.start()
    return ToolResult.success("Spinning. Say stop when you have had enough.")


def compiz_control(args: dict[str, str]) -> ToolResult:
    action = args["action"]
    if action == "spin":
        return start_spin(args.get("direction", "right"))
    if action == "spin_stop":
        return ToolResult.success("Stopping." if stop_spin() else "The cube is not spinning.")
    keys = bindings()
    plan: dict[str, tuple[str, int, str]] = {
        "cube_right": ("cube_right", 1, ""),  # the cube turning is the answer
        "cube_left": ("cube_left", 1, ""),
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
                "Compiz desktop effects: rotate the desktop cube left or right, spin it slowly while zoomed out "
                "(like holding ctrl+alt and dragging), zoom the desktop in or out, and move the zoomed view left or right."
            ),
            parameters={
                "action": "one of cube_right, cube_left, spin, spin_stop, zoom_in, zoom_in_more, zoom_out, zoom_reset, zoom_left, zoom_right",
                "direction": "for spin: left or right (default right)",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=compiz_control,
            required=("action",),
        ),
    ]
