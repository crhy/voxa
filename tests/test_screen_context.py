"""The screen context provider: a fake window, never a real desktop.

Every host read is injected, so these tests prove two things at once: the block
reaches the prompt when the request refers to the screen, and no host read is
even attempted when it does not.
"""

from __future__ import annotations

import subprocess

import pytest

from voxa.agent import screencontext
from voxa.agent.planner import system_prompt
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool, ToolRegistry
from voxa.agent.result import ToolResult

FAKE_TITLE = "Invoice — LibreOffice"
FAKE_LINES = [
    "Acme Supplies",
    "Invoice no. 2026-0417",
    "Date 5 June 2026",
    "Widget 3  14.99",
    "Gadget 1  199.00",
    "Total  213.99",
]


class FakeWindow:
    """Stands in for xdotool and the accessibility helper."""

    def __init__(self, title: str = FAKE_TITLE, lines: list[str] | None = None) -> None:
        self._title = title
        self._lines = lines if lines is not None else list(FAKE_LINES)
        self.title_calls = 0
        self.text_calls: list[str] = []

    def title(self) -> str:
        self.title_calls += 1
        return self._title

    def text(self, app: str) -> list[str]:
        self.text_calls.append(app)
        return list(self._lines)


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_window",
            description="Read the text shown in another program's window.",
            parameters={"app": "name of the program (optional)"},
            risk=RiskLevel.READ_ONLY,
            handler=lambda args: ToolResult.success(""),
        )
    )
    return registry


@pytest.mark.parametrize(
    ("said", "attached"),
    [
        ("read this to me", True),
        ("what's here", True),
        ("summarize this", True),
        ("what is on my screen", True),
        ("what window is this", True),
        ("tell me what you see", True),
        ("the window in front of me", True),
        ("what is the weather", False),
        ("search this page for prices", False),
        ("compare prices on the website", False),
        ("open gmail", False),
        ("is this working", False),
    ],
)
def test_refers_to_screen(said: str, attached: bool) -> None:
    assert screencontext.refers_to_screen(said) is attached


def test_the_screen_block_is_attached_for_a_screen_request() -> None:
    window = FakeWindow()
    block = screencontext.gather("read this to me", title=window.title, text=window.text)
    assert block == (
        f'Focused window: "{FAKE_TITLE}"\n'
        "Visible text:\n" + "\n".join(FAKE_LINES)
    )
    assert window.title_calls == 1
    # The text read is asked for by the window that is in front.
    assert window.text_calls == [FAKE_TITLE]


@pytest.mark.parametrize("said", ["what's here", "summarize this", "Voxa, read this"])
def test_the_block_reaches_the_planner_prompt(said: str) -> None:
    window = FakeWindow()
    block = screencontext.gather(said, title=window.title, text=window.text)
    prompt = system_prompt(make_registry(), block)
    assert prompt.endswith(block)
    assert "Acme Supplies" in prompt
    assert window.title_calls == 1


@pytest.mark.parametrize("said", ["what is the weather", "search this page for prices"])
def test_no_host_read_at_all_when_the_request_is_not_about_the_screen(said: str) -> None:
    window = FakeWindow()
    assert screencontext.gather(said, title=window.title, text=window.text) == ""
    assert window.title_calls == 0
    assert window.text_calls == []


def test_a_window_that_cannot_be_named_gives_nothing() -> None:
    for title in (lambda: "", lambda: "   ", boom, never_named):
        assert screencontext.gather("read this", title=title, text=FakeWindow().text) == ""


def boom() -> str:
    raise OSError("xdotool is not installed")


def never_named() -> str:
    raise subprocess.TimeoutExpired(["xdotool", "getactivewindow"], 15)


def test_a_window_whose_text_fails_gives_the_title_alone() -> None:
    def broken_text(app: str) -> list[str]:
        raise subprocess.TimeoutExpired(["python3", "-"], 15)

    block = screencontext.gather("what's here", title=lambda: FAKE_TITLE, text=broken_text)
    assert block == f'Focused window: "{FAKE_TITLE}"'


def test_an_empty_window_gives_nothing_to_say() -> None:
    assert screencontext.gather("read this", title=lambda: "", text=lambda app: []) == ""
    assert screencontext.window_text("", []) == ""
    assert screencontext.window_text("", ["", "a", "  "]) == ""


def test_the_block_is_capped_at_max_lines_and_max_chars() -> None:
    many = [f"line {index}" for index in range(200)]
    short = screencontext.window_text(FAKE_TITLE, many, max_lines=3)
    assert short.splitlines() == [
        f'Focused window: "{FAKE_TITLE}"',
        "Visible text:",
        "line 0",
        "line 1",
        "line 2",
    ]
    tight = screencontext.window_text(FAKE_TITLE, many, max_lines=200, max_chars=60)
    assert len(tight) <= 60
    assert tight.startswith(f'Focused window: "{FAKE_TITLE}"')
    capped = screencontext.gather("read this", title=lambda: FAKE_TITLE, text=lambda app: many)
    assert capped.splitlines().count("line 39") == 1
    assert "line 40" not in capped


def test_a_dict_answer_from_the_helper_is_understood() -> None:
    block = screencontext.gather(
        "read this", title=lambda: FAKE_TITLE, text=lambda app: {"ok": True, "text": FAKE_LINES}
    )
    assert "Total  213.99" in block


def test_the_prompt_without_context_is_byte_identical_to_todays_output() -> None:
    registry = make_registry()
    plain = system_prompt(registry)
    block = screencontext.window_text(FAKE_TITLE, FAKE_LINES)
    assert system_prompt(registry, "") == plain
    assert system_prompt(registry, "   ") == plain
    with_context = system_prompt(registry, block)
    assert with_context.startswith(plain)
    assert with_context == f"{plain}\n\n{block}"
    assert with_context.endswith(block)
    assert plain.count("Focused window:") == 0
