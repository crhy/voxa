from __future__ import annotations

from collections.abc import Callable

from voxa.agent import interpreter
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.replylang import language_name

ask_model: Callable | None = None                 # set by the window: messages -> text
speak_in: Callable[[str], None] | None = None     # set by the window: speak the NEXT reply with this language's voice

# One session per run. The window shares this object, so the spoken commands and
# the microphone pipeline always see the same pair of languages.
SESSION = interpreter.Session()


def interpreter_start(args: dict[str, str]) -> ToolResult:
    if not SESSION.start(args.get("language_a", ""), args.get("language_b", "")):
        first = language_name(args.get("language_a", ""))
        second = language_name(args.get("language_b", ""))
        if first is not None and first == second:
            return ToolResult.failure(f"Interpreting needs two different languages, not {first} twice.")
        return ToolResult.failure("I can interpret between " + interpreter.spoken_languages() + ". Which two would you like?")
    sides = SESSION.sides()
    return ToolResult.success(
        f"Interpreting between {sides[0]} and {sides[1]}. Say something in either; I answer in the other.",
        detail=f"{sides[0]} <-> {sides[1]}",
    )


def interpreter_stop(args: dict[str, str]) -> ToolResult:
    if not SESSION.active:
        return ToolResult.failure("Interpreter mode is not on.")
    SESSION.stop()
    return ToolResult.success("Interpreter mode is off.")


def interpreter_status(args: dict[str, str]) -> ToolResult:
    if not SESSION.active:
        return ToolResult.failure("Interpreter mode is not on.")
    sides = SESSION.sides()
    heard = SESSION.last_side
    answer = interpreter.target_language(sides, heard)
    return ToolResult.success(
        f"Interpreting between {sides[0]} and {sides[1]}. "
        f"Now listening for {heard}, answering in {answer}.",
        detail=f"{sides[0]} <-> {sides[1]}, listening {heard}",
    )


def interpreter_switch(args: dict[str, str]) -> ToolResult:
    if not SESSION.active:
        return ToolResult.failure("Interpreter mode is not on.")
    SESSION.switch()
    sides = SESSION.sides()
    heard = SESSION.last_side
    return ToolResult.success(
        f"Listening for {heard} now, answering in {interpreter.other(sides, heard)}.",
        detail=f"{sides[0]} <-> {sides[1]}, listening {heard}",
    )


def interpreter_tools() -> list[Tool]:
    return [
        Tool(name="interpreter_start", description="Start interpreter mode: two languages, each turn answered in the other one.",
             parameters={"language_a": "the first language", "language_b": "the second language"},
             risk=RiskLevel.READ_ONLY, handler=interpreter_start, required=("language_a", "language_b")),
        Tool(name="interpreter_stop", description="Stop interpreter mode and go back to one language.",
             parameters={}, risk=RiskLevel.READ_ONLY, handler=interpreter_stop),
        Tool(name="interpreter_status", description="Which two languages are being interpreted, and which side speaks next.",
             parameters={}, risk=RiskLevel.READ_ONLY, handler=interpreter_status),
        Tool(name="interpreter_switch", description="Flip which side of interpreter mode speaks next.",
             parameters={}, risk=RiskLevel.READ_ONLY, handler=interpreter_switch),
    ]
