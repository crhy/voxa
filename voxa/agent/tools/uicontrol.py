from __future__ import annotations

import logging

from voxa.agent import ui
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.uicontrol")

_REFUSED = frozenset(
    {
        "delete",
        "delete all",
        "erase",
        "format",
        "wipe",
        "factory reset",
        "pay",
        "pay now",
        "buy",
        "buy now",
        "place order",
        "confirm purchase",
        "send money",
        "transfer",
    }
)

_FRAME_BUTTONS = frozenset({"minimize", "maximize", "close", "restore"})


def press_button(args: dict[str, str]) -> ToolResult:
    label = args["label"]
    app = args.get("app", "")
    if label.strip().casefold() in _REFUSED:
        return ToolResult.failure("I will not press that by voice. Please do it yourself.")
    if not app:
        app = ui.active_window_title()
    if not app:
        return ToolResult.failure("I cannot tell which window is in front.")
    result = ui.press(app, label)
    if result.get("ok"):
        pressed = (result.get("pressed") or {}).get("name") or label
        return ToolResult.success(f"Pressed {pressed}.")
    error = result.get("error", "")
    if error == "disabled":
        return ToolResult.failure(f"{label} is greyed out right now.")
    if error == "program not found":
        return ToolResult.failure(f"I could not find the program {app}.")
    if error == "not found":
        message = f"I could not find a button called {label}."
        names = [
            item.get("name")
            for item in ui.items(app)
            if item.get("showing") and item.get("enabled") and item.get("name")
        ]
        if names:
            shown = ", ".join(names[:6])
            message += f" I can see: {shown}."
        return ToolResult.failure(message)
    return ToolResult.failure("I could not reach that program's buttons.")


def list_buttons(args: dict[str, str]) -> ToolResult:
    app = args.get("app", "")
    if not app:
        app = ui.active_window_title()
    names = [
        item.get("name")
        for item in ui.items(app)
        if item.get("role") == "push button"
        and item.get("showing")
        and item.get("enabled")
        and item.get("name")
        and len(item["name"]) <= 40
        and item["name"].casefold() not in _FRAME_BUTTONS
    ]
    if not names:
        return ToolResult.failure("I do not see any buttons there.")
    if len(names) <= 10:
        if len(names) == 1:
            listed = names[0]
        else:
            listed = ", ".join(names[:-1]) + " and " + names[-1]
        return ToolResult.success(f"I can see {len(names)} buttons: {listed}.")
    listed = ", ".join(names[:10])
    more = len(names) - 10
    return ToolResult.success(f"I can see {len(names)} buttons: {listed} and {more} more.")


def read_window(args: dict[str, str]) -> ToolResult:
    app = args.get("app", "")
    if not app:
        app = ui.active_window_title()
    lines = [line for line in ui.text(app) if len(line) >= 2]
    if not lines:
        return ToolResult.failure("I cannot read anything in that window.")
    joined = ". ".join(lines)
    if len(joined) > 700:
        cut = joined[:700]
        end = max(cut.rfind(". "), cut.rfind("."))
        if end > 0:
            cut = cut[: end + 1]
        joined = cut
    return ToolResult.success(joined)


def uicontrol_tools() -> list[Tool]:
    return [
        Tool(
            name="press_button",
            description="Press a named button in another program's window.",
            parameters={"label": "name of the button to press", "app": "name of the program (optional)"},
            risk=RiskLevel.REVERSIBLE,
            handler=press_button,
            required=("label",),
        ),
        Tool(
            name="list_buttons",
            description="List the buttons you can press in another program's window.",
            parameters={"app": "name of the program (optional)"},
            risk=RiskLevel.READ_ONLY,
            handler=list_buttons,
        ),
        Tool(
            name="read_window",
            description="Read the text shown in another program's window.",
            parameters={"app": "name of the program (optional)"},
            risk=RiskLevel.READ_ONLY,
            handler=read_window,
        ),
    ]
