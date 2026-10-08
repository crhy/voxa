from __future__ import annotations

from collections.abc import Callable

from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

get_echo_mode: Callable[[], str] | None = None
set_echo_mode: Callable[[str], None] | None = None


def _set_noise_cancellation_handler(args: dict[str, str]) -> ToolResult:
    state = args["state"].lower().strip()
    if state not in ("on", "off"):
        return ToolResult.failure("I did not catch whether to turn it on or off.")
    if set_echo_mode is None:
        return ToolResult.failure("I cannot change that here.")
    set_echo_mode("voxa" if state == "on" else "off")
    if state == "on":
        return ToolResult.success("Noise cancellation is on.")
    return ToolResult.success("Noise cancellation is off.")


def _get_noise_cancellation_handler(_args: dict[str, str]) -> ToolResult:
    if get_echo_mode is None:
        return ToolResult.failure("I cannot check that here.")
    mode = get_echo_mode()
    if mode == "voxa":
        return ToolResult.success("Noise cancellation is on.")
    return ToolResult.success("Noise cancellation is off.")


def anc_tools() -> list[Tool]:
    return [
        Tool(
            name="set_noise_cancellation",
            description="Turn adaptive noise cancellation on or off.",
            parameters={"state": "on or off"},
            risk=RiskLevel.REVERSIBLE,
            handler=_set_noise_cancellation_handler,
            required=("state",),
        ),
        Tool(
            name="get_noise_cancellation",
            description="Report whether adaptive noise cancellation is currently on or off.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_get_noise_cancellation_handler,
        ),
    ]
