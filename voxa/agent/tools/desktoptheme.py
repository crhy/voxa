from __future__ import annotations

import json
import logging

from voxa.agent.desktoptheme import find_theme, metatheme_keys
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.desktoptheme")

_THEMES_PATH = "/usr/share/spaced-themes/themes.json"
_SPACED_ONLY = "Changing the desktop theme only works on Spaced Linux."


def change_theme(args: dict[str, str]) -> ToolResult:
    from voxa.agent.tools.windows import _run

    name = args["name"]
    proc = _run(["cat", _THEMES_PATH], check=False)
    if proc.returncode != 0:
        return ToolResult.failure(_SPACED_ONLY)
    try:
        themes = json.loads(proc.stdout).get("themes") or []
    except (ValueError, TypeError, AttributeError):
        return ToolResult.failure(_SPACED_ONLY)

    theme = find_theme(name, themes)
    if theme is None:
        known = ", ".join(t["name"] for t in themes)
        return ToolResult.failure(f"I don't know a theme called {name}. I know: {known}")

    idx = _run(["cat", f"/usr/share/themes/{theme['gtk_theme']}/index.theme"], check=False)
    keys = metatheme_keys(idx.stdout) if idx.returncode == 0 else {}
    gtk = keys.get("GtkTheme") or theme["gtk_theme"]

    commands: list[list[str]] = []
    if keys.get("IconTheme"):
        commands.append(["gsettings", "set", "org.mate.interface", "icon-theme", keys["IconTheme"]])
    if keys.get("MetacityTheme"):
        commands.append(["gsettings", "set", "org.mate.Marco.general", "theme", keys["MetacityTheme"]])
    if keys.get("CursorTheme"):
        commands.append(
            ["gsettings", "set", "org.mate.peripherals-mouse", "cursor-theme", keys["CursorTheme"]]
        )
    commands.append(["gsettings", "set", "org.mate.interface", "gtk-theme", gtk])

    for command in commands:
        if _run(command, check=False).returncode != 0:
            return ToolResult.failure(f"I couldn't switch to {theme['name']}.")

    _run(["/usr/local/bin/spaced-switch-theme", theme["id"]], check=False)
    return ToolResult.success(f"Switching to {theme['name']}.")


def desktoptheme_tools() -> list[Tool]:
    return [
        Tool(
            name="change_theme",
            description="Switch the Spaced Linux desktop theme.",
            parameters={"name": "the theme name"},
            risk=RiskLevel.REVERSIBLE,
            handler=change_theme,
            required=("name",),
        )
    ]
