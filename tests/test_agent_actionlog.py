"""The action log: append, read, rotate and summarise."""

from __future__ import annotations

import json
from pathlib import Path

from voxa.agent.actionlog import ActionLog, ActionRecord, default_path, summarize


def _record(heard: str, route: str = "tool", **kwargs) -> ActionRecord:
    return ActionRecord(time="2026-10-03T12:00:00", heard=heard, route=route, **kwargs)


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    log = ActionLog(tmp_path / "voxa" / "actions.jsonl")
    log.append(
        _record(
            "open gmail",
            tool="open_site",
            args={"name": "gmail"},
            ok=True,
            speech="Opening gmail.",
            ms=42,
        )
    )
    log.append(_record("what is two plus two", route="model"))
    records = log.read()
    assert len(records) == 2
    first = records[0]
    assert list(first) == ["time", "heard", "route", "tool", "args", "ok", "speech", "detail", "ms"]
    assert first["heard"] == "open gmail"
    assert first["route"] == "tool"
    assert first["tool"] == "open_site"
    assert first["args"] == {"name": "gmail"}
    assert first["ok"] is True
    assert first["ms"] == 42
    assert records[1]["ok"] is None
    assert records[1]["route"] == "model"


def test_read_limit_keeps_the_newest(tmp_path: Path) -> None:
    log = ActionLog(tmp_path / "actions.jsonl")
    for index in range(5):
        log.append(_record(f"command {index}"))
    assert [r["heard"] for r in log.read()] == [f"command {i}" for i in range(5)]
    assert [r["heard"] for r in log.read(limit=2)] == ["command 3", "command 4"]
    assert len(log.read(limit=10)) == 5


def test_corrupt_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    path.write_text(
        '{"heard": "good one", "route": "tool"}\n'
        "this is not json at all\n"
        "\n"
        '{"heard": "good two", "route": "model"}\n',
        encoding="utf-8",
    )
    records = ActionLog(path).read()
    assert [r["heard"] for r in records] == ["good one", "good two"]


def test_rotation_when_the_file_grows_past_max_bytes(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    log = ActionLog(path, max_bytes=60)
    for index in range(3):
        log.append(_record(f"{'x' * 40} {index}", ok=True, ms=10))
    assert Path(str(path) + ".1").exists()
    kept = log.read()
    assert len(kept) == 1
    assert kept[0]["heard"].endswith("2")
    rotated = Path(str(path) + ".1").read_text(encoding="utf-8").splitlines()
    assert len(rotated) == 1
    assert json.loads(rotated[0])["heard"].endswith("1")


def test_default_path_honours_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXA_ACTION_LOG", str(tmp_path / "custom" / "log.jsonl"))
    assert default_path() == tmp_path / "custom" / "log.jsonl"
    monkeypatch.delenv("VOXA_ACTION_LOG")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert default_path() == tmp_path / "state" / "voxa" / "actions.jsonl"
    monkeypatch.delenv("XDG_STATE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_path() == tmp_path / ".local" / "state" / "voxa" / "actions.jsonl"


def test_unwritable_path_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should be", encoding="utf-8")
    log = ActionLog(blocker / "actions.jsonl")
    log.append(_record("open gmail"))
    assert log.read() == []


def test_summarize_counts_routes_tools_and_failures() -> None:
    records = [
        {"route": "tool", "tool": "open_site", "ok": True, "ms": 10},
        {"route": "tool", "tool": "open_site", "ok": True, "ms": 30},
        {"route": "tool", "tool": "open_site", "ok": False, "ms": 50, "heard": "open zebras"},
        {"route": "tool", "tool": "send_gmail", "ok": False, "ms": 5, "heard": "send it"},
        {"route": "dictation", "heard": "hello there", "ok": True},
        {"route": "model", "heard": "why is the sky blue"},
        {"route": "model", "heard": "why is the sky blue"},
        {"route": "model", "heard": "alpha question"},
        {"route": "legacy", "heard": "open files"},
    ]
    summary = summarize(records)
    assert summary["total"] == 9
    assert summary["by_route"] == {"tool": 4, "dictation": 1, "model": 3, "legacy": 1}
    assert summary["by_tool"]["open_site"] == {"calls": 3, "failed": 1, "avg_ms": 30}
    assert summary["by_tool"]["send_gmail"] == {"calls": 1, "failed": 1, "avg_ms": 5}
    assert [r["heard"] for r in summary["failures"]] == ["send it", "open zebras"]
    assert len(summary["failures"]) <= 20
    assert summary["unrouted"] == [
        {"heard": "why is the sky blue", "count": 2},
        {"heard": "alpha question", "count": 1},
    ]
