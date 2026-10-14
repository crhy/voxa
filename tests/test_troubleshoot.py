"""Troubleshooting phrases route to the new tools and the handlers answer from canned data."""

from __future__ import annotations

from voxa.agent.intents import ToolCall, route
from voxa.agent.sysinfo import find_problems, parse_ps, spoken_problems, spoken_top_cpu, spoken_top_memory
from voxa.agent.tools import default_registry
from voxa.agent.tools.system import system_problems, top_cpu, top_memory


def test_parse_ps_sums_and_skips_kernel_threads() -> None:
    text = (
        "brave  20.5  12.0\n"
        "brave  10.5  6.0\n"
        "[kthread]  99.0  99.0\n"
        "python3  12.0  8.0\n"
        "Xorg  4.0  2.0\n"
    )
    assert parse_ps(text) == [("brave", 31.0, 18.0), ("python3", 12.0, 8.0), ("Xorg", 4.0, 2.0)]
    assert parse_ps(text, top=1) == [("brave", 31.0, 18.0)]
    assert parse_ps("") == []


def test_spoken_top_cpu_variants() -> None:
    rows = [("brave", 31.0, 18.0), ("python3", 12.0, 8.0), ("Xorg", 4.0, 2.0)]
    assert spoken_top_cpu(rows) == (
        "The busiest programs are brave at 31 percent, python3 at 12 percent and Xorg at 4 percent."
    )
    assert spoken_top_cpu([("brave", 31.0, 18.0)]) == "The busiest program is brave at 31 percent."
    assert spoken_top_cpu([("sleep", 0.5, 1.0)]) == "Nothing is using much of the processor right now."
    assert spoken_top_cpu([]) == "Nothing is using much of the processor right now."


def test_spoken_top_memory_variants() -> None:
    rows = [("brave", 18.0, 12.0), ("python3", 9.0, 8.0)]
    assert spoken_top_memory(rows) == "The most memory is used by brave with 12 percent, then python3 with 8 percent."
    assert spoken_top_memory([("brave", 18.0, 12.0)]) == "The most memory is used by brave with 12 percent."
    assert spoken_top_memory([]) == "Nothing is using much memory right now."


def test_find_problems_thresholds() -> None:
    ok_mem = (24000000, 32000000)
    ok_disk = (128849018880, 536870912000)
    assert find_problems(91, 12, (46, 4, 3454, 16376), *ok_mem, *ok_disk) == [
        "the processor is hot at 91 degrees"
    ]
    assert find_problems(67, 12, (91, 4, 3454, 16376), *ok_mem, *ok_disk) == [
        "the graphics card is hot at 91 degrees"
    ]
    assert find_problems(67, 90, None, *ok_mem, *ok_disk) == ["the processor is fully busy"]
    assert find_problems(67, 12, None, 3000000, 32000000, *ok_disk) == ["memory is almost full"]
    assert find_problems(67, 12, None, *ok_mem, 21474836480, 536870912000) == [
        "the disk is almost full (4 percent free)"
    ]
    assert find_problems(67, 12, (46, 4, 3454, 16376), *ok_mem, *ok_disk) == []
    assert find_problems(None, None, None, None, None, None, None) == []


def test_spoken_problems_variants() -> None:
    assert spoken_problems([]) == "Everything looks normal."
    assert spoken_problems(["the disk is almost full (4 percent free)"]) == (
        "I see one thing: the disk is almost full (4 percent free)."
    )
    assert spoken_problems(["A", "B"]) == "I see 2 things: A, and B."


def test_troubleshoot_phrases_route() -> None:
    assert route("why is my computer slow") == ToolCall("top_cpu", {})
    assert route("why is this machine so slow") == ToolCall("top_cpu", {})
    assert route("what is slowing down my computer") == ToolCall("top_cpu", {})
    assert route("what's using all the cpu") == ToolCall("top_cpu", {})
    assert route("what is using my memory") == ToolCall("top_memory", {})
    assert route("what's eating the ram") == ToolCall("top_memory", {})
    assert route("is anything wrong with my computer") == ToolCall("system_problems", {})
    assert route("do you see anything unusual") == ToolCall("system_problems", {})
    assert route("any problems with my computer") == ToolCall("system_problems", {})
    assert route("diagnose the pc") == ToolCall("system_problems", {})


def test_troubleshoot_rules_do_not_steal_other_questions() -> None:
    assert route("what's on my to-do list") == ToolCall("list_read", {"list": "to-do"})
    assert route("what time is it") == ToolCall("tell_time", {})
    assert route("what is the capital of France") is None
    assert route("open the calculator") == ToolCall("open_app", {"name": "calculator"})
    assert route("how is my computer doing") == ToolCall("system_health", {})


def test_troubleshoot_handlers_end_to_end(monkeypatch) -> None:
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    cpu = top_cpu({})
    assert cpu.ok and cpu.speech == (
        "The busiest programs are brave at 31 percent, python3 at 12 percent and Xorg at 4 percent."
    )
    mem = top_memory({})
    assert mem.ok and mem.speech == (
        "The most memory is used by brave with 12 percent, then python3 with 8 percent."
    )
    problems = system_problems({})
    assert problems.ok and problems.speech == "Everything looks normal."


def test_registry_has_troubleshoot_tools() -> None:
    names = default_registry().names()
    for name in ("top_cpu", "top_memory", "system_problems"):
        assert name in names
