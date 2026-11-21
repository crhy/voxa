"""The local, opt-in turn log from crhy/voxa#116 §6.

Nothing here touches a model, the network or the audio stack: the log is a
JSONL file under the state directory that the user can view and delete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxa.agent.tools.turnlog import turn_log_clear, turn_log_show
from voxa.agent.turnlog import TurnLog, TurnRecord, default_path, summarize


def _record(**kwargs) -> TurnRecord:
    fields = {
        "time": "2026-11-02T09:15:00",
        "kind": "model",
        "brain": "little",
        "first_spoken_ms": 900,
        "escalated": None,
        "escalate_reason": "",
        "review_verdict": "",
        "review_note": "",
        "correction": "",
        "barge_in": False,
    }
    fields.update(kwargs)
    return TurnRecord(**fields)  # type: ignore[arg-type]


def test_the_log_is_off_and_writes_nothing_by_default(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    log = TurnLog(path)
    assert log.enabled is False
    log.record(_record())
    log.record(_record(kind="command"))
    assert not path.exists()
    assert log.read() == []


def test_one_record_per_turn_with_every_field_the_issue_names(tmp_path: Path) -> None:
    log = TurnLog(tmp_path / "turns.jsonl", enabled=True)
    log.record(
        _record(
            kind="model",
            brain="little",
            first_spoken_ms=820,
            escalated=True,
            escalate_reason="ask_expert",
            review_verdict="disagree",
            review_note="the date was a year off",
            correction="Correction: the date was a year off.",
            barge_in=True,
        )
    )
    log.record(_record(kind="command", brain="little", first_spoken_ms=120))
    records = log.read()
    assert len(records) == 2
    first = records[0]
    assert list(first) == [
        "time",
        "kind",
        "brain",
        "first_spoken_ms",
        "escalated",
        "escalate_reason",
        "review_verdict",
        "correction",
        "barge_in",
        "review_note",
    ]
    assert first["kind"] == "model"
    assert first["brain"] == "little"
    assert first["first_spoken_ms"] == 820
    assert first["escalated"] is True
    assert first["escalate_reason"] == "ask_expert"
    assert first["review_verdict"] == "disagree"
    assert first["correction"].startswith("Correction:")
    assert first["barge_in"] is True
    assert records[1]["escalated"] is None
    assert records[1]["review_verdict"] == ""
    assert "review_note" not in records[1]


def test_default_path_honours_the_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXA_TURN_LOG", str(tmp_path / "custom" / "log.jsonl"))
    assert default_path() == tmp_path / "custom" / "log.jsonl"
    monkeypatch.delenv("VOXA_TURN_LOG")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert default_path() == tmp_path / "state" / "voxa" / "turns.jsonl"
    monkeypatch.delenv("XDG_STATE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_path() == tmp_path / ".local" / "state" / "voxa" / "turns.jsonl"


def test_nothing_is_written_outside_the_state_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("VOXA_TURN_LOG", raising=False)
    log = TurnLog(enabled=True)
    assert log.path == tmp_path / "state" / "voxa" / "turns.jsonl"
    log.record(_record())
    written = sorted(part.relative_to(tmp_path) for part in tmp_path.rglob("*") if part.is_file())
    assert written == [Path("state") / "voxa" / "turns.jsonl"]


def test_clear_removes_the_file_and_the_rotated_copy(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    log = TurnLog(path, enabled=True, max_bytes=60)
    for index in range(3):
        log.record(_record(kind=f"kind {index}"))
    assert Path(str(path) + ".1").exists()
    assert log.clear() is True
    assert not path.exists()
    assert not Path(str(path) + ".1").exists()
    assert log.read() == []
    assert log.clear() is False


def test_an_unwritable_path_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory should be", encoding="utf-8")
    log = TurnLog(blocker / "turns.jsonl", enabled=True)
    log.record(_record())
    assert log.read() == []
    assert log.clear() is False


def test_corrupt_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    path.write_text(
        '{"kind": "model"}\nnot json at all\n\n{"kind": "command"}\n',
        encoding="utf-8",
    )
    assert [record["kind"] for record in TurnLog(path).read()] == ["model", "command"]


def test_summarize_reports_accuracy_hit_rate_and_the_median() -> None:
    records = [
        # little brain answered, reviewer agreed: no escalation needed
        {"kind": "model", "brain": "little", "escalated": False, "review_verdict": "agree", "first_spoken_ms": 400},
        {"kind": "model", "brain": "little", "escalated": False, "review_verdict": "agree", "first_spoken_ms": 600},
        # little brain answered, reviewer found a real error: it should have escalated
        {"kind": "model", "brain": "little", "escalated": False, "review_verdict": "disagree", "first_spoken_ms": 900},
        # escalated to the big brain, reviewed, and the review disagreed
        {"kind": "model", "brain": "big", "escalated": True, "review_verdict": "disagree", "first_spoken_ms": 1500},
        # escalated, never reviewed
        {"kind": "command", "brain": "big", "escalated": True, "review_verdict": "", "first_spoken_ms": 200},
        # nothing measured at all
        {"kind": "dictation", "brain": "little", "escalated": None, "review_verdict": "", "first_spoken_ms": 0},
    ]
    summary = summarize(records)
    assert summary["total"] == 6
    assert summary["by_kind"] == {"model": 4, "command": 1, "dictation": 1}
    assert summary["by_brain"] == {"little": 4, "big": 2}
    assert summary["escalation"]["decisions"] == 5
    assert summary["escalation"]["escalated"] == 2
    assert summary["escalation"]["confirmed"] == 2
    assert summary["escalation"]["missed"] == 1
    assert summary["escalation"]["accuracy"] == pytest.approx(0.667)
    assert summary["review"] == {
        "reviewed": 4,
        "disagreed": 2,
        "hit_rate": pytest.approx(0.5),
    }
    assert summary["first_spoken_ms"] == {"median": 600, "little": 600, "big": 850}
    assert summary["corrections"] == 0
    assert summary["barge_ins"] == 0


def test_summarize_of_an_empty_log_measures_nothing() -> None:
    summary = summarize([])
    assert summary["total"] == 0
    assert summary["escalation"]["accuracy"] is None
    assert summary["review"]["hit_rate"] is None
    assert summary["first_spoken_ms"] == {"median": 0, "little": 0, "big": 0}


def test_turn_log_is_a_real_setting(tmp_path: Path) -> None:
    from voxa.config import ConfigStore, Settings

    assert Settings().turn_log is False
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(turn_log=True))
    assert store.load().turn_log is True
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"wake_word": "voxa"}), encoding="utf-8")
    assert ConfigStore(path).load().turn_log is False
    path.write_text(json.dumps({"turn_log": True, "bogus": 1}), encoding="utf-8")
    assert ConfigStore(path).load(migrate=False).turn_log is True


def test_the_tools_show_and_delete_the_log(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "state" / "voxa" / "turns.jsonl"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("VOXA_TURN_LOG", raising=False)

    empty = turn_log_show({})
    assert empty.ok is True
    assert "empty" in empty.speech
    assert str(path) in empty.speech

    TurnLog(enabled=True).record(
        _record(kind="model", first_spoken_ms=500, review_verdict="agree", escalated=False)
    )
    TurnLog(enabled=True).record(
        _record(
            kind="model",
            brain="big",
            first_spoken_ms=1500,
            escalated=True,
            review_verdict="disagree",
            correction="Correction: the year was wrong.",
            barge_in=True,
        )
    )

    shown = turn_log_show({})
    assert shown.ok is True
    assert "2 turns logged" in shown.speech
    assert str(path) in shown.speech
    assert "turns.jsonl" in shown.detail

    cleared = turn_log_clear({})
    assert cleared.ok is True
    assert "deleted" in cleared.speech
    assert not path.exists()
    again = turn_log_clear({})
    assert again.ok is True
    assert "already empty" in again.speech


def test_the_tools_never_write_anything_themselves(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("VOXA_TURN_LOG", raising=False)
    turn_log_show({})
    turn_log_clear({})
    assert list(tmp_path.rglob("*")) == []
