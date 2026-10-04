from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxa.agent.intents import route
from voxa.agent.registry import ToolError, ToolRegistry

DEFAULT_COMMAND_CASES_PATH = Path(__file__).with_name("data") / "command_cases.json"


@dataclass(frozen=True, slots=True)
class CommandCase:
    __test__ = False

    id: str
    category: str
    say: str
    tool: str | None
    args: dict[str, str] | None


@dataclass(frozen=True, slots=True)
class CommandResult:
    __test__ = False

    case: CommandCase
    got_tool: str | None
    got_args: dict[str, str]
    passed: bool
    problem: str
    micros: int


def load_command_cases(path: Path) -> tuple[list[CommandCase], list[CommandCase]]:
    """Read the bench file and return (cases, wishlist)."""
    if not path.exists():
        return [], []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Command case file must be an object with a cases array.")

    return _case_list(payload.get("cases", [])), _case_list(payload.get("wishlist", []))


def _case_list(raw: Any) -> list[CommandCase]:
    if not isinstance(raw, list):
        raise ValueError("Command cases must be a JSON array.")

    cases: list[CommandCase] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("Each command case must be an object.")
        say = str(item.get("say", "")).strip()
        if not say:
            raise ValueError(f"Command case {index} has no say text.")
        case_id = str(item.get("id", "")).strip() or f"cmd-{index + 1}"
        category = str(item.get("category", "general")).strip() or "general"
        tool = item.get("tool")
        tool = str(tool).strip() if tool is not None else None
        args = item.get("args")
        if args is not None:
            if not isinstance(args, dict):
                raise ValueError(f"Command case {case_id} has non-object args.")
            args = {str(key): str(value) for key, value in args.items()}
        cases.append(CommandCase(case_id, category, say, tool, args))
    return cases


def check(case: CommandCase, registry: ToolRegistry) -> CommandResult:
    """Route one utterance and compare it with the expected tool call."""
    started = time.perf_counter()
    call = route(case.say)
    micros = int((time.perf_counter() - started) * 1_000_000)

    got_tool = call.tool if call is not None else None
    got_args = dict(call.args) if call is not None else {}

    if case.tool is None:
        problem = f"expected no tool, got {got_tool}" if got_tool is not None else ""
    elif got_tool != case.tool:
        problem = f"expected {case.tool}, got {got_tool}"
    elif case.args is not None and got_args != case.args:
        problem = f"args {case.args} != {got_args}"
    else:
        problem = ""

    if not problem and got_tool is not None:
        try:
            registry.validate(got_tool, got_args)
        except ToolError as exc:
            problem = f"validation failed: {exc}"

    return CommandResult(case, got_tool, got_args, not problem, problem, micros)


def run_commands(cases: list[CommandCase], registry: ToolRegistry) -> list[CommandResult]:
    return [check(case, registry) for case in cases]


def summarize(results: list[CommandResult]) -> dict[str, Any]:
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        entry = by_category.setdefault(result.case.category, {"total": 0, "passed": 0})
        entry["total"] += 1
        if result.passed:
            entry["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "by_category": by_category,
        "slowest_micros": max((result.micros for result in results), default=0),
        "failures": [
            {"id": result.case.id, "say": result.case.say, "problem": result.problem}
            for result in results
            if not result.passed
        ],
    }


def format_markdown(summary: dict[str, Any], wishlist_results: list[CommandResult]) -> str:
    lines = ["# Voxa command bench", ""]
    lines.append(
        f"Total: {summary['total']} cases, {summary['passed']} passed, {summary['failed']} failed "
        f"(pass rate {summary['pass_rate']}%). Slowest route: {summary['slowest_micros']} microseconds."
    )
    lines.append("")
    lines.append("| Category | Total | Passed |")
    lines.append("| --- | ---: | ---: |")
    for category, counts in sorted(summary["by_category"].items()):
        lines.append(f"| {category} | {counts['total']} | {counts['passed']} |")

    lines.append("")
    lines.append("## Failures")
    if summary["failures"]:
        lines.append("| id | say | problem |")
        lines.append("| --- | --- | --- |")
        for failure in summary["failures"]:
            lines.append(f"| {failure['id']} | {failure['say']} | {failure['problem']} |")
    else:
        lines.append("None.")

    lines.append("")
    lines.append("## Wishlist (not handled yet)")
    if wishlist_results:
        lines.append("| id | say | router did |")
        lines.append("| --- | --- | --- |")
        for result in wishlist_results:
            if result.got_tool is None:
                action = "nothing (no rule matched)"
            else:
                action = f"{result.got_tool} {result.got_args}"
            lines.append(f"| {result.case.id} | {result.case.say} | {action} |")
    else:
        lines.append("None.")

    return "\n".join(lines) + "\n"
