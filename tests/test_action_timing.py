"""Per-stage timings ride along in the action log without breaking old lines."""

from __future__ import annotations

from pathlib import Path

from voxa.agent.actionlog import ActionLog, ActionRecord


def test_round_trip_keeps_timings(tmp_path: Path) -> None:
    log = ActionLog(tmp_path / "actions.jsonl")
    log.append(
        ActionRecord(
            time="2026-10-03T12:00:00",
            heard="open gmail",
            route="tool",
            tool="open_site",
            args={"name": "gmail"},
            ok=True,
            ms=42,
            timings={"hear_ms": 120, "route_ms": 5, "act_ms": 37},
        )
    )
    records = log.read()
    assert len(records) == 1
    assert records[0]["timings"] == {"hear_ms": 120, "route_ms": 5, "act_ms": 37}


def test_record_without_timings_has_no_timings_key(tmp_path: Path) -> None:
    log = ActionLog(tmp_path / "actions.jsonl")
    log.append(ActionRecord(time="2026-10-03T12:00:00", heard="hello", route="model"))
    records = log.read()
    assert len(records) == 1
    assert "timings" not in records[0]


def test_old_line_without_timings_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    path.write_text(
        '{"time":"2026-10-01T09:00:00","heard":"old","route":"tool","tool":"x","args":{},"ok":true,"speech":"","detail":"","ms":7}\n',
        encoding="utf-8",
    )
    records = ActionLog(path).read()
    assert len(records) == 1
    assert records[0]["heard"] == "old"
    assert "timings" not in records[0]
