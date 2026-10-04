"""The planner: command detection, plan parsing and plan execution."""

from __future__ import annotations

import pytest

from voxa.agent.intents import ToolCall
from voxa.agent.planner import (
    Plan,
    PlanError,
    looks_like_command,
    parse_plan,
    run_plan,
    system_prompt,
)
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool, ToolRegistry
from voxa.agent.result import ToolResult


def make_registry(calls: list[tuple[str, dict[str, str]]], fail_on: str | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    def handler(name: str):
        def run(args: dict[str, str]) -> ToolResult:
            calls.append((name, dict(args)))
            if name == fail_on:
                return ToolResult.failure("That tool failed.", detail="boom")
            return ToolResult.success("")
        return run

    registry.register(
        Tool(name="open_site", description="Open a website", parameters={"name": "site"},
             risk=RiskLevel.REVERSIBLE, handler=handler("open_site"), required=("name",))
    )
    registry.register(
        Tool(name="compose_gmail", description="Start a new Gmail message", parameters={},
             risk=RiskLevel.REVERSIBLE, handler=handler("compose_gmail"))
    )
    registry.register(
        Tool(name="press_key", description="Press a key", parameters={"key": "key"},
             risk=RiskLevel.REVERSIBLE, handler=handler("press_key"), required=("key",))
    )
    return registry


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("open gmail", True),
        ("close the browser", True),
        ("please open gmail", True),
        ("can you play jazz", True),
        ("could you mute the video", True),
        ("hey, turn it up", True),
        ("ok, open files", True),
        ("okay send it", True),
        ("search youtube for cats", True),
        ("compose a new message", True),
        ("put on some jazz", True),
        ("check the time", True),
        ("put on some jazz and then turn it down a bit", True),
        ("close brave and open libreoffice writer", True),
        ("find me a video and make it full screen", True),
        ("gmail and then a new message", False),
        ("play simon and garfunkel", True),
        ("what is the capital of France", False),
        ("how do I change the volume", False),
        ("why is my microphone not working", False),
        ("tell me a joke", False),
        ("is this working", False),
        ("playing games is fun", False),
        ("where is my resume", False),
        ("when does the store open", False),
        ("", False),
        ("Open Gmail", True),
    ],
)
def test_looks_like_command(text: str, expected: bool) -> None:
    assert looks_like_command(text) is expected


def test_system_prompt_lists_every_tool_and_mentions_json() -> None:
    registry = make_registry([])
    prompt = system_prompt(registry)
    for name in registry.names():
        assert name in prompt
    assert "JSON" in prompt
    assert "press_key keys:" in prompt
    assert "full screen" in prompt
    assert "volume up" in prompt


def test_parse_plan_clean_json() -> None:
    registry = make_registry([])
    plan = parse_plan(
        '{"steps": [{"tool": "open_site", "args": {"name": "gmail"}}, '
        '{"tool": "compose_gmail", "args": {}}], "say": "Opening Gmail."}',
        registry,
    )
    assert plan.say == "Opening Gmail."
    assert len(plan.steps) == 2
    assert plan.steps[0].tool == "open_site"
    assert plan.steps[0].args == {"name": "gmail"}
    assert plan.steps[1].tool == "compose_gmail"
    assert plan.steps[1].args == {}


def test_parse_plan_args_are_stripped_and_stringified() -> None:
    registry = make_registry([])
    plan = parse_plan('{"steps": [{"tool": "press_key", "args": {"key": "  volume up  "}}]}', registry)
    assert plan.steps[0].args == {"key": "volume up"}
    assert plan.say == ""


def test_parse_plan_inside_a_code_fence() -> None:
    registry = make_registry([])
    reply = '```json\n{"steps": [{"tool": "open_site", "args": {"name": "gmail"}}], "say": "Ok."}\n```'
    plan = parse_plan(reply, registry)
    assert plan.steps[0].tool == "open_site"
    assert plan.say == "Ok."


def test_parse_plan_with_prose_around_it() -> None:
    registry = make_registry([])
    reply = 'Sure, here is the plan: {"steps": [], "say": "I need more detail."} Hope that helps!'
    plan = parse_plan(reply, registry)
    assert plan.steps == ()
    assert plan.say == "I need more detail."


def test_parse_plan_empty_steps_is_allowed() -> None:
    registry = make_registry([])
    plan = parse_plan('{"steps": [], "say": "No tool fits."}', registry)
    assert plan.steps == ()
    assert plan.say == "No tool fits."


def test_parse_plan_rejects_unknown_tool() -> None:
    registry = make_registry([])
    with pytest.raises(PlanError, match="unknown tool"):
        parse_plan('{"steps": [{"tool": "launch_rocket", "args": {}}]}', registry)


def test_parse_plan_rejects_unknown_argument() -> None:
    registry = make_registry([])
    with pytest.raises(PlanError, match="unknown argument"):
        parse_plan('{"steps": [{"tool": "open_site", "args": {"url": "gmail"}}]}', registry)


def test_parse_plan_rejects_missing_required_argument() -> None:
    registry = make_registry([])
    with pytest.raises(PlanError, match="missing required"):
        parse_plan('{"steps": [{"tool": "open_site", "args": {}}]}', registry)


def test_parse_plan_rejects_too_many_steps() -> None:
    registry = make_registry([])
    steps = ", ".join('{"tool": "press_key", "args": {"key": "a"}}' for _ in range(7))
    with pytest.raises(PlanError, match="too many steps"):
        parse_plan(f'{{"steps": [{steps}], "say": "ok"}}', registry)


def test_parse_plan_rejects_not_json() -> None:
    registry = make_registry([])
    with pytest.raises(PlanError):
        parse_plan("I cannot do that, sorry.", registry)
    with pytest.raises(PlanError):
        parse_plan('{"steps": "open gmail"}', registry)
    with pytest.raises(PlanError):
        parse_plan('{"steps": [{"args": {}}]}', registry)


def test_run_plan_calls_steps_in_order_with_sleeps_between_only() -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    registry = make_registry(calls)
    slept: list[float] = []
    plan = Plan(
        steps=(
            ToolCall("open_site", {"name": "gmail"}),
            ToolCall("compose_gmail", {}),
        ),
        say="Done.",
    )
    sentence = run_plan(plan, registry, sleep=lambda seconds: slept.append(seconds))
    assert calls == [("open_site", {"name": "gmail"}), ("compose_gmail", {})]
    assert slept == [0.8]
    assert sentence == "Done."


def test_run_plan_stops_at_first_failure_and_returns_its_speech() -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    registry = make_registry(calls, fail_on="compose_gmail")
    from voxa.agent.intents import ToolCall

    plan = Plan(
        steps=(
            ToolCall("open_site", {"name": "gmail"}),
            ToolCall("compose_gmail", {}),
            ToolCall("press_key", {"key": "enter"}),
        ),
        say="Opening Gmail.",
    )
    sentence = run_plan(plan, registry, sleep=lambda seconds: None)
    assert calls == [("open_site", {"name": "gmail"}), ("compose_gmail", {})]
    assert sentence == "That tool failed."


def test_run_plan_say_fallbacks() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(name="open_site", description="Open a website", parameters={"name": "site"},
             risk=RiskLevel.REVERSIBLE,
             handler=lambda args: ToolResult.success("Opened the site."),
        )
    )
    from voxa.agent.intents import ToolCall

    plan_with_step_speech = Plan(steps=(ToolCall("open_site", {"name": "gmail"}),), say="")
    assert run_plan(plan_with_step_speech, registry, sleep=lambda seconds: None) == "Opened the site."

    silent = ToolRegistry()
    silent.register(
        Tool(name="press_key", description="Press a key", parameters={"key": "key"},
             risk=RiskLevel.REVERSIBLE, handler=lambda args: ToolResult.success(""))
    )
    quiet_plan = Plan(steps=(ToolCall("press_key", {"key": "a"}),), say="")
    assert run_plan(quiet_plan, silent, sleep=lambda seconds: None) == "Done."
    assert run_plan(Plan(steps=(), say=""), silent, sleep=lambda seconds: None) == "Done."


def test_run_plan_on_step_sees_every_result() -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    registry = make_registry(calls)
    from voxa.agent.intents import ToolCall

    seen: list[tuple[str, bool, int]] = []
    plan = Plan(
        steps=(ToolCall("open_site", {"name": "gmail"}), ToolCall("press_key", {"key": "volume up"})),
        say="Working.",
    )
    sentence = run_plan(
        plan,
        registry,
        on_step=lambda call, result, ms: seen.append((call.tool, result.ok, ms)),
        sleep=lambda seconds: None,
    )
    assert sentence == "Working."
    assert [tool for tool, ok, ms in seen] == ["open_site", "press_key"]
    assert all(ok for _, ok, _ in seen)
    assert all(ms >= 0 for _, _, ms in seen)
