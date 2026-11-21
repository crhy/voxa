"""Voice tools for the personal profile: your name, your home address, your phone number.

Nothing here is secret; the values go in the settings file, which is why they
may be read back out loud. Secrets go to the keyring instead.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from voxa.agent.profile import ADDRESS_FIELD, NAME_FIELD, PHONE_FIELD, field_label, spoken_profile
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

get_profile: Callable[[str], str] | None = None
save_profile: Callable[[str, str], None] | None = None

# The word the router and the model use for each field, and the field it writes.
ARGUMENTS = {NAME_FIELD: "name", ADDRESS_FIELD: "home_address", PHONE_FIELD: "phone_number"}


def _set_handler(field: str, args: dict[str, str]) -> ToolResult:
    label = field_label(field)
    value = args.get(ARGUMENTS[field], "")
    if not value:
        return ToolResult.failure(f"I did not catch that {label}.")
    if get_profile is None or save_profile is None:
        return ToolResult.failure("I cannot save that here.")
    save_profile(field, value)
    return ToolResult.success(spoken_profile(field, value))


def _get_handler(field: str, args: dict[str, str]) -> ToolResult:
    label = field_label(field)
    if get_profile is None:
        return ToolResult.failure(f"I do not have your {label}.")
    value = get_profile(field)
    if not value:
        return ToolResult.failure(f"I do not have your {label}.")
    return ToolResult.success(spoken_profile(field, value))


def _forget_handler(field: str, args: dict[str, str]) -> ToolResult:
    label = field_label(field)
    if get_profile is None or save_profile is None:
        return ToolResult.failure(f"I do not have your {label}.")
    if not get_profile(field):
        return ToolResult.failure(f"I do not have your {label}.")
    save_profile(field, "")
    return ToolResult.success(f"Forgotten your {label}.")


def _profile_tools_for(field: str) -> list[Tool]:
    arg = ARGUMENTS[field]
    label = field_label(field)
    return [
        Tool(
            name=f"set_{arg}",
            description=f"Remember your {label} from something said out loud.",
            parameters={arg: f"your {label}, said out loud"},
            risk=RiskLevel.REVERSIBLE,
            handler=partial(_set_handler, field),
            required=(arg,),
        ),
        Tool(
            name=f"get_{arg}",
            description=f"Read back your saved {label}.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=partial(_get_handler, field),
        ),
        Tool(
            name=f"forget_{arg}",
            description=f"Forget your saved {label}.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=partial(_forget_handler, field),
        ),
    ]


def profile_tools() -> list[Tool]:
    return [tool for field in (NAME_FIELD, ADDRESS_FIELD, PHONE_FIELD) for tool in _profile_tools_for(field)]
