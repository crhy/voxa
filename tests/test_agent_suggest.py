"""Tests for the rule-based coaching suggestions."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

from voxa.agent.suggest import SuggestionState, suggest


def _iso(seconds: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat(timespec="seconds")


def test_shortcut_fires_for_single_step_plan(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "play jazz",
            "route": "plan",
            "tool": "play_youtube",
            "args": {"query": "jazz"},
            "ok": True,
        },
    ]
    result = suggest(records, state)
    assert result is not None
    assert result.kind == "shortcut"
    assert result.key == "shortcut:play_youtube"
    assert "play jazz" in result.text


def test_shortcut_does_not_fire_for_multi_step_plan(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "play jazz",
            "route": "plan",
            "tool": "play_youtube",
            "args": {"query": "jazz"},
            "ok": True,
        },
        {
            "time": _iso(1001),
            "heard": "play jazz",
            "route": "plan",
            "tool": "play_youtube",
            "args": {"query": "jazz"},
            "ok": True,
        },
    ]
    result = suggest(records, state)
    assert result is None


def test_shortcut_does_not_fire_for_non_routed_tool(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "do something",
            "route": "plan",
            "tool": "unknown_tool",
            "args": {},
            "ok": True,
        },
    ]
    result = suggest(records, state)
    assert result is None


def test_direct_fires_for_open_app_then_open_site(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "open brave",
            "route": "tool",
            "tool": "open_app",
            "args": {"name": "brave"},
            "ok": True,
        },
        {
            "time": _iso(1010),
            "heard": "open gmail",
            "route": "tool",
            "tool": "open_site",
            "args": {"name": "gmail"},
            "ok": True,
        },
    ]
    result = suggest(records, state)
    assert result is not None
    assert result.kind == "direct"
    assert "open gmail" in result.text


def test_direct_does_not_fire_when_gap_exceeds_60s(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "open brave",
            "route": "tool",
            "tool": "open_app",
            "args": {"name": "brave"},
            "ok": True,
        },
        {
            "time": _iso(1061),
            "heard": "open gmail",
            "route": "tool",
            "tool": "open_site",
            "args": {"name": "gmail"},
            "ok": True,
        },
    ]
    result = suggest(records, state)
    assert result is None


def test_direct_fires_for_web_search_matching_known_site(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "open brave",
            "route": "tool",
            "tool": "open_app",
            "args": {"name": "brave"},
            "ok": True,
        },
        {
            "time": _iso(1010),
            "heard": "search for gmail",
            "route": "tool",
            "tool": "web_search",
            "args": {"query": "gmail"},
            "ok": True,
        },
    ]
    result = suggest(records, state)
    assert result is not None
    assert result.kind == "direct"
    assert "open gmail" in result.text


def test_routine_fires_for_three_repeated_pairs(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = []
    for i in range(3):
        base = 1000 + i * 60
        records.append({
            "time": _iso(base),
            "heard": "close firefox",
            "route": "tool",
            "tool": "close_app",
            "args": {"name": "firefox"},
            "ok": True,
        })
        records.append({
            "time": _iso(base + 5),
            "heard": "open gmail",
            "route": "tool",
            "tool": "open_site",
            "args": {"name": "gmail"},
            "ok": True,
        })
    result = suggest(records, state)
    assert result is not None
    assert result.kind == "routine"
    assert "close Firefox" in result.text
    assert "open Gmail" in result.text


def test_routine_does_not_fire_with_only_two_occurrences(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = []
    for i in range(2):
        base = 1000 + i * 60
        records.append({
            "time": _iso(base),
            "heard": "close firefox",
            "route": "tool",
            "tool": "close_app",
            "args": {"name": "firefox"},
            "ok": True,
        })
        records.append({
            "time": _iso(base + 5),
            "heard": "open gmail",
            "route": "tool",
            "tool": "open_site",
            "args": {"name": "gmail"},
            "ok": True,
        })
    result = suggest(records, state)
    assert result is None


def test_phrasing_fires_for_repeated_unrouted_command(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000 + i * 10),
            "heard": "turn on the lights",
            "route": "model",
            "tool": "",
            "args": {},
            "ok": None,
        }
        for i in range(3)
    ]
    result = suggest(records, state)
    assert result is not None
    assert result.kind == "phrasing"
    assert "turn on the lights" in result.text


def test_phrasing_does_not_fire_for_non_command_text(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000 + i * 10),
            "heard": "what is the capital of France",
            "route": "model",
            "tool": "",
            "args": {},
            "ok": None,
        }
        for i in range(3)
    ]
    result = suggest(records, state)
    assert result is None


def test_suggest_returns_none_when_disabled(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "play jazz",
            "route": "plan",
            "tool": "play_youtube",
            "args": {"query": "jazz"},
            "ok": True,
        },
    ]
    result = suggest(records, state, enabled=False)
    assert result is None


def test_suggest_returns_none_when_newest_failed(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "play jazz",
            "route": "plan",
            "tool": "play_youtube",
            "args": {"query": "jazz"},
            "ok": False,
        },
    ]
    result = suggest(records, state)
    assert result is None


def test_state_blocks_repeated_offers(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "play jazz",
            "route": "plan",
            "tool": "play_youtube",
            "args": {"query": "jazz"},
            "ok": True,
        },
    ]
    first = suggest(records, state)
    assert first is not None
    state.offered(first.key)
    second = suggest(records, state)
    assert second is None


def test_state_blocks_after_dismiss(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "play jazz",
            "route": "plan",
            "tool": "play_youtube",
            "args": {"query": "jazz"},
            "ok": True,
        },
    ]
    first = suggest(records, state)
    assert first is not None
    state.dismiss(first.key)
    second = suggest(records, state)
    assert second is None


def test_state_blocks_after_dismiss_all(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    state.dismiss_all()
    records = [
        {
            "time": _iso(1000),
            "heard": "play jazz",
            "route": "plan",
            "tool": "play_youtube",
            "args": {"query": "jazz"},
            "ok": True,
        },
    ]
    result = suggest(records, state)
    assert result is None


def test_state_round_trips_through_json(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    state.dismiss("shortcut:play_youtube")
    state.offered("direct:open_app>open_site")
    reloaded = SuggestionState(tmp_path / "state.json")
    assert "shortcut:play_youtube" in reloaded.dismissed
    assert reloaded.counts.get("direct:open_app>open_site") == 1


def test_first_match_wins(tmp_path: Path) -> None:
    state = SuggestionState(tmp_path / "state.json")
    records = [
        {
            "time": _iso(1000),
            "heard": "open brave",
            "route": "tool",
            "tool": "open_app",
            "args": {"name": "brave"},
            "ok": True,
        },
        {
            "time": _iso(1010),
            "heard": "open gmail",
            "route": "tool",
            "tool": "open_site",
            "args": {"name": "gmail"},
            "ok": True,
        },
    ]
    result = suggest(records, state)
    assert result is not None
    assert result.kind == "direct"
