from __future__ import annotations

import json
from pathlib import Path

from voxa.agent.registry import ToolRegistry
from voxa.agent.tools import default_registry
from voxatest.__main__ import main
from voxatest.commands import (
    DEFAULT_COMMAND_CASES_PATH,
    CommandCase,
    format_markdown,
    load_command_cases,
    run_commands,
    summarize,
)

CATEGORIES = {"web", "youtube", "media", "keys", "typing", "apps", "windows", "dictation", "question", "heard"}


def _registry():
    return default_registry()


def test_shipped_cases_all_pass():
    cases, _ = load_command_cases(DEFAULT_COMMAND_CASES_PATH)
    summary = summarize(run_commands(cases, _registry()))
    assert summary["failed"] == 0, summary["failures"]


def test_case_file_shape():
    cases, wishlist = load_command_cases(DEFAULT_COMMAND_CASES_PATH)
    assert len(cases) >= 120
    assert len(wishlist) <= 15
    assert {case.category for case in cases} == CATEGORIES
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))
    names = set(_registry().names())
    for case in cases:
        if case.tool is not None:
            assert case.tool in names, case.id


def test_check_reports_wrong_tool():
    registry = _registry()
    case = CommandCase("x-1", "web", "open gmail", "open_app", None)
    result = run_commands([case], registry)[0]
    assert not result.passed
    assert result.problem == "expected open_app, got open_site"


def test_check_reports_unexpected_tool_for_question():
    registry = _registry()
    case = CommandCase("x-2", "question", "what is YouTube?", None, None)
    result = run_commands([case], registry)[0]
    assert result.passed and result.got_tool is None

    case = CommandCase("x-3", "question", "copy", None, None)
    result = run_commands([case], registry)[0]
    assert not result.passed
    assert result.problem == "expected no tool, got press_key"


def test_check_reports_wrong_args():
    registry = _registry()
    case = CommandCase("x-4", "web", "open gmail", "open_site", {"name": "Gmail"})
    result = run_commands([case], registry)[0]
    assert not result.passed
    assert result.problem == "args {'name': 'Gmail'} != {'name': 'gmail'}"


def test_check_reports_validation_error():
    empty = ToolRegistry()
    case = CommandCase("x-6", "keys", "press enter", "press_key", {"key": "enter"})
    result = run_commands([case], empty)[0]
    assert not result.passed
    assert result.problem == "validation failed: unknown tool: press_key"


def test_format_markdown_has_totals_and_failure_row():
    registry = _registry()
    cases = [
        CommandCase("ok-1", "web", "open gmail", "open_site", {"name": "gmail"}),
        CommandCase("bad-1", "web", "open gmail", "open_app", None),
    ]
    results = run_commands(cases, registry)
    markdown = format_markdown(summarize(results), run_commands([CommandCase("w-1", "web", "start dictating", None, None)], registry))
    assert "# Voxa command bench" in markdown
    assert "Total: 2 cases, 1 passed, 1 failed" in markdown
    assert "| bad-1 | open gmail | expected open_app, got open_site |" in markdown
    assert "## Wishlist (not handled yet)" in markdown
    assert "start dictating" in markdown


def test_commands_subcommand_writes_reports(tmp_path: Path):
    report_dir = tmp_path / "reports"
    code = main(["commands", "--report-dir", str(report_dir)])
    assert code == 0
    markdown = (report_dir / "command-bench.md").read_text(encoding="utf-8")
    assert "# Voxa command bench" in markdown
    payload = json.loads((report_dir / "command-bench.json").read_text(encoding="utf-8"))
    assert payload["summary"]["failed"] == 0
    assert len(payload["results"]) >= 120


def test_commands_subcommand_fails_on_bad_case(tmp_path: Path):
    cases_path = tmp_path / "bad.json"
    cases_path.write_text(
        json.dumps({"cases": [{"id": "bad-1", "category": "web", "say": "open gmail", "tool": "open_app"}]}),
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"
    code = main(["commands", "--cases", str(cases_path), "--report-dir", str(report_dir)])
    assert code == 1
    assert (report_dir / "command-bench.md").exists()
    assert (report_dir / "command-bench.json").exists()
