from __future__ import annotations

import pytest

from voxa.agent.policy import RiskLevel, needs_confirmation
from voxa.agent.registry import Tool, ToolError, ToolRegistry
from voxa.agent.result import ToolResult


def _tool(name: str, handler=None, required=(), parameters=None, risk=RiskLevel.READ_ONLY):
    return Tool(
        name=name,
        description=f"do {name}",
        parameters=parameters if parameters is not None else {"url": "the url"},
        risk=risk,
        handler=handler or (lambda args: ToolResult.success(f"Opening {name}.")),
        required=required,
    )


def test_register_get_names():
    registry = ToolRegistry()
    registry.register(_tool("open_url"))
    registry.register(_tool("copy_text"))
    assert registry.names() == ["copy_text", "open_url"]
    assert registry.get("open_url").name == "open_url"
    with pytest.raises(ToolError):
        registry.get("nope")


def test_duplicate_name_raises():
    registry = ToolRegistry()
    registry.register(_tool("open_url"))
    with pytest.raises(ToolError):
        registry.register(_tool("open_url"))


def test_describe():
    registry = ToolRegistry()
    registry.register(
        _tool(
            "open_url",
            parameters={"url": "the url", "tab": "which tab"},
            risk=RiskLevel.REVERSIBLE,
        )
    )
    assert registry.describe() == "open_url(url, tab): do open_url [risk 1]"


def test_validate_strips_and_converts():
    registry = ToolRegistry()
    registry.register(_tool("open_url", parameters={"url": "the url", "n": "count"}))
    assert registry.validate("open_url", {"url": " x ", "n": 3}) == {"url": "x", "n": "3"}


def test_validate_unknown_argument():
    registry = ToolRegistry()
    registry.register(_tool("open_url"))
    with pytest.raises(ToolError):
        registry.validate("open_url", {"bogus": "x"})


def test_validate_missing_and_blank_required():
    registry = ToolRegistry()
    registry.register(_tool("open_url", required=("url",)))
    with pytest.raises(ToolError):
        registry.validate("open_url", {})
    with pytest.raises(ToolError):
        registry.validate("open_url", {"url": "   "})


def test_call_returns_handler_result():
    registry = ToolRegistry()
    registry.register(_tool("open_url"))
    result = registry.call("open_url", {"url": "gmail.com"})
    assert result.ok is True
    assert result.speech == "Opening open_url."


def test_call_handler_exception():
    def boom(args):
        raise RuntimeError("kaboom")

    registry = ToolRegistry()
    registry.register(_tool("open_url", handler=boom))
    result = registry.call("open_url", {"url": "x"})
    assert result.ok is False
    assert result.speech == "That didn't work."
    assert result.detail == "kaboom"


def test_call_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolError):
        registry.call("ghost", {})


def test_needs_confirmation_truth_table():
    for level in (RiskLevel.READ_ONLY, RiskLevel.REVERSIBLE):
        assert needs_confirmation(level) is False
        assert needs_confirmation(level, granted=frozenset({"x"}), tool_name="x") is False

    assert needs_confirmation(RiskLevel.EXTERNAL_WRITE) is True
    assert needs_confirmation(RiskLevel.EXTERNAL_WRITE, granted=frozenset({"send"}), tool_name="send") is False
    assert needs_confirmation(RiskLevel.EXTERNAL_WRITE, granted=frozenset({"send"}), tool_name="other") is True

    for granted in (frozenset(), frozenset({"delete"})):
        assert needs_confirmation(RiskLevel.SENSITIVE, granted=granted, tool_name="delete") is True


def test_toolresult_success_failure():
    ok = ToolResult.success("Opening Gmail.")
    bad = ToolResult.failure("That didn't work.", detail="boom")
    assert (ok.ok, ok.speech, ok.detail) == (True, "Opening Gmail.", "")
    assert (bad.ok, bad.speech, bad.detail) == (False, "That didn't work.", "boom")
