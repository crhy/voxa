from __future__ import annotations

from voxa.agent.cdp import CdpError
from voxa.agent.intents import ToolCall
from voxa.agent.planner import Plan, plan_supported, run_browser_task
from voxa.agent.result import ToolResult
from voxa.agent.tools.browser import host as browser_host
from voxa.agent.tools.browser import open_site


def _one_step_plan(tool: str, args: dict[str, str]) -> Plan:
    return Plan((ToolCall(tool, args),), "")


def test_pause_not_turned_into_play():
    plan = _one_step_plan("play_video", {"query": "Xa, pause"})
    assert not plan_supported(plan, "Xa, pause")


def test_step_pause_not_play_jazz():
    plan = _one_step_plan("play_video", {"query": "jazz"})
    assert not plan_supported(plan, "Step, pause")


def test_legitimate_play_jazz():
    plan = _one_step_plan("play_video", {"query": "jazz"})
    assert plan_supported(plan, "play some jazz")


def test_set_timer_needs_time():
    plan = _one_step_plan("set_timer", {"duration": "5 minutes"})
    assert plan_supported(plan, "set a timer for 5 minutes")
    assert not plan_supported(plan, "set a timer")


def test_open_site_unknown_word():
    browser_host.spawn = lambda argv: None
    result = open_site({"name": "expedia"})
    assert result.ok
    assert result.detail == "https://www.expedia.com"
    assert result.speech == "Opening expedia."


def test_open_site_dotted_name():
    browser_host.spawn = lambda argv: None
    result = open_site({"name": "spacedlinux.com"})
    assert result.ok
    assert result.detail == "https://spacedlinux.com"
    assert result.speech == "Opening spacedlinux.com."


def test_open_site_known_site():
    browser_host.spawn = lambda argv: None
    result = open_site({"name": "gmail"})
    assert result.ok
    assert result.detail == "https://mail.google.com/mail/u/0/#inbox"


class FakeRegistry:
    def describe(self) -> str:
        return ""

    def names(self) -> list[str]:
        return []

    def validate(self, name: str, args: dict[str, str]) -> dict[str, str]:
        return args

    def call(self, name: str, args: dict[str, str]) -> ToolResult:
        return ToolResult.success("done")


class FakeSession:
    def __init__(self, fail_times: int) -> None:
        self.remaining = fail_times

    def ensure(self) -> None:
        pass

    def outline(self) -> list:
        if self.remaining > 0:
            self.remaining -= 1
            raise CdpError("the browser did not accept the command")
        return []

    def title(self) -> str:
        return "Page"

    def url(self) -> str:
        return "https://example.com/"


def _ask_done(messages: list[dict]) -> str:
    return '{"done": true, "say": "Found it."}'


def test_browser_recovers_once():
    session = FakeSession(fail_times=1)
    answer = run_browser_task("find it", _ask_done, session, FakeRegistry(), max_steps=3)
    assert answer == "Found it."


def test_browser_gives_up_when_always_failing():
    session = FakeSession(fail_times=99)
    answer = run_browser_task("find it", _ask_done, session, FakeRegistry(), max_steps=3)
    assert answer == "I lost contact with my browser. Say that again and I'll reopen it."
