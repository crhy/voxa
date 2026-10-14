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


def summarize(
    results: list[CommandResult],
    *,
    elapsed_seconds: float | None = None,
    unique_cases: int | None = None,
    repeats: int = 1,
    minimum_checks_per_hour: float = 1000.0,
) -> dict[str, Any]:
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    by_category: dict[str, dict[str, int]] = {}
    failures: dict[tuple[str, str, str], dict[str, str]] = {}
    for result in results:
        entry = by_category.setdefault(result.case.category, {"total": 0, "passed": 0})
        entry["total"] += 1
        if result.passed:
            entry["passed"] += 1
        else:
            key = (result.case.id, result.case.say, result.problem)
            failures[key] = {"id": result.case.id, "say": result.case.say, "problem": result.problem}

    summary: dict[str, Any] = {
        "total": total,
        "unique_cases": unique_cases if unique_cases is not None else total,
        "repeats": repeats,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "by_category": by_category,
        "slowest_micros": max((result.micros for result in results), default=0),
        "failures": list(failures.values()),
    }
    if elapsed_seconds is not None:
        checks_per_hour = total / elapsed_seconds * 3600 if elapsed_seconds > 0 else 0.0
        summary.update(
            {
                "elapsed_seconds": round(elapsed_seconds, 6),
                "checks_per_second": round(total / elapsed_seconds, 1) if elapsed_seconds > 0 else 0.0,
                "checks_per_hour": round(checks_per_hour),
                "minimum_checks_per_hour": minimum_checks_per_hour,
                "throughput_passed": checks_per_hour >= minimum_checks_per_hour,
            }
        )
    return summary


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_markdown(summary: dict[str, Any], wishlist_results: list[CommandResult]) -> str:
    lines = [
        "# Voxa capability audit",
        "",
        "This report measures Voxa's deterministic command router. It does not exercise the microphone, "
        "speech recognition, model backend, network, or desktop side effects. Use the spoken `voxatest run` "
        "suite for those end-to-end paths.",
        "",
    ]
    lines.append(
        f"Result: {summary['total']} checks, {summary['passed']} passed, {summary['failed']} failed "
        f"(pass rate {summary['pass_rate']}%). {summary['unique_cases']} unique cases repeated "
        f"{summary['repeats']} time(s)."
    )
    if "checks_per_hour" in summary:
        throughput = "PASS" if summary["throughput_passed"] else "FAIL"
        lines.extend(
            [
                "",
                "## Throughput",
                "",
                f"- Elapsed: {summary['elapsed_seconds']:.6f} seconds",
                f"- Rate: {summary['checks_per_second']:,.1f} checks/second "
                f"({summary['checks_per_hour']:,} checks/hour)",
                f"- Required minimum: {summary['minimum_checks_per_hour']:,.0f} checks/hour",
                f"- Throughput result: **{throughput}**",
            ]
        )

    lines.extend(
        [
            "",
            "## What works in the deterministic command router",
            "",
            "| Category | Checks | Passed | Failed | Status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for category, counts in sorted(summary["by_category"].items()):
        failed = counts["total"] - counts["passed"]
        status = "PASS" if failed == 0 else "FAIL"
        lines.append(f"| {_cell(category)} | {counts['total']} | {counts['passed']} | {failed} | {status} |")

    lines.extend(["", "## What does not work"])
    if summary["failures"]:
        lines.extend(["", "| id | say | problem |", "| --- | --- | --- |"])
        for failure in summary["failures"]:
            lines.append(
                f"| {_cell(failure['id'])} | {_cell(failure['say'])} | {_cell(failure['problem'])} |"
            )
    else:
        lines.extend(["", "No deterministic router failures were found."])

    lines.extend(["", "## Wishlist (not handled yet)"])
    if wishlist_results:
        lines.extend(["", "| id | say | router did |", "| --- | --- | --- |"])
        for result in wishlist_results:
            action = "nothing (no rule matched)" if result.got_tool is None else f"{result.got_tool} {result.got_args}"
            lines.append(f"| {_cell(result.case.id)} | {_cell(result.case.say)} | {_cell(action)} |")
    else:
        lines.extend(["", "None."])

    lines.extend(
        [
            "",
            "## Coverage limits",
            "",
            "A passing row proves that text was routed to the expected registered tool with valid arguments. "
            "It does not prove that the target application accepted the action. Hardware, audio, model, "
            "network, and desktop integration results belong in the spoken reports and `docs/CAPABILITIES.md`.",
        ]
    )
    return "\n".join(lines) + "\n"
