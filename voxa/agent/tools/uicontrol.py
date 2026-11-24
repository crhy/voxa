from __future__ import annotations

import logging
from datetime import datetime

from voxa.agent import ui
from voxa.agent.actionlog import ActionLog, ActionRecord
from voxa.agent.policy import (
    DEFAULT_PER_DAY_CAP,
    DEFAULT_PER_ORDER_CAP,
    IntendedSpend,
    RiskLevel,
    day_spend_total,
    spend_guard,
)
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.config import ConfigStore

log = logging.getLogger("voxa.agent.tools.uicontrol")

# Buttons that destroy something: never pressed by voice, whatever is said.
_DESTRUCTIVE_REFUSED = frozenset(
    {
        "delete",
        "delete all",
        "erase",
        "format",
        "wipe",
        "factory reset",
    }
)

# Buttons that spend money: refused unless the reusable spend guard (issue #43)
# returns allow — a read-back, a "confirm" within 20 seconds, and the caps.
_SPEND_REFUSED = frozenset(
    {
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

_REFUSED = _DESTRUCTIVE_REFUSED | _SPEND_REFUSED

_FRAME_BUTTONS = frozenset({"minimize", "maximize", "close", "restore"})


def _press_and_report(app: str, label: str) -> ToolResult:
    """Press ``label`` in ``app`` and speak the outcome, unchanged from before."""
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


def _price(args: dict[str, str]) -> float:
    try:
        return max(0.0, float(args.get("price", "")))
    except (TypeError, ValueError):
        return 0.0


def _confirm_seconds(args: dict[str, str]) -> float | None:
    raw = args.get("confirm_seconds", "")
    if raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _spend_press(label: str, app: str, args: dict[str, str]) -> ToolResult:
    """Route a money-spending button through the one reusable spend guard, and
    log every allowed spend exactly once with amount, payee and reference."""
    settings = ConfigStore().load()
    action_log = ActionLog()
    intended = IntendedSpend(
        what=label,
        price=_price(args),
        source=args.get("source", ""),
        payee=args.get("payee", ""),
        address=args.get("address", ""),
        reference=args.get("reference", ""),
    )
    decision = spend_guard(
        intended,
        confirm_word=args.get("confirm", ""),
        confirm_seconds=_confirm_seconds(args),
        per_order_cap=getattr(settings, "spend_per_order_cap", DEFAULT_PER_ORDER_CAP),
        per_day_cap=getattr(settings, "spend_per_day_cap", DEFAULT_PER_DAY_CAP),
        spent_today=day_spend_total(action_log),
    )
    if not decision.allowed:
        return ToolResult.failure(decision.reason)
    result = _press_and_report(app, label)
    if result.ok:
        action_log.append(
            ActionRecord(
                time=datetime.now().isoformat(timespec="seconds"),
                heard=label,
                route="spend",
                tool="press_button",
                args={
                    "amount": f"{intended.price:.2f}",
                    "payee": intended.payee,
                    "reference": intended.reference,
                },
                ok=True,
                speech=decision.readback,
            )
        )
        return ToolResult.success(result.speech, detail=decision.readback)
    return result


def press_button(args: dict[str, str]) -> ToolResult:
    label = args["label"]
    app = args.get("app", "")
    key = label.strip().casefold()
    if key in _DESTRUCTIVE_REFUSED:
        return ToolResult.failure("I will not press that by voice. Please do it yourself.")
    if key in _SPEND_REFUSED:
        return _spend_press(label, app, args)
    return _press_and_report(app, label)


def list_buttons(args: dict[str, str]) -> ToolResult:
    app = args.get("app", "")
    if not app:
        app = ui.active_window_title()
    names = [
        item.get("name")
        for item in ui.items(app)
        # No role filter: `ui.items` already returns only pressable things, and the role of a plain button is
        # "push button" in older toolkits and "button" in current ones.
        if item.get("showing")
        and item.get("enabled")
        and item.get("name")
        and len(item["name"]) <= 40
        and item["name"].casefold() not in _FRAME_BUTTONS
    ]
    names = list(dict.fromkeys(names))
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
            parameters={
                "label": "name of the button to press",
                "app": "name of the program (optional)",
                "confirm": "the spoken word 'confirm', when the button spends money",
                "confirm_seconds": "seconds since the read-back the word was heard",
                "price": "the amount the button would spend",
                "payee": "who it is paid to",
                "address": "an account or address it is sent to",
                "source": "the shop or site it buys from",
                "reference": "an order or reference number",
            },
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
