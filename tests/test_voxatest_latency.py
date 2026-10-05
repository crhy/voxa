"""Tests for voxatest/latency.py."""

from __future__ import annotations

from voxa.agent.actionlog import ActionLog, ActionRecord
from voxatest.__main__ import main
from voxatest.latency import format_latency, latency_report

HANDMADE_RECORDS = [
    {"route": "timing", "heard": "alpha", "timings": {"total_ms": 100, "transcribe_ms": 10}},
    {"route": "timing", "heard": "beta", "timings": {"total_ms": 300, "transcribe_ms": 30, "route_ms": 5}},
    {"route": "timing", "heard": "gamma", "timings": {"total_ms": 200, "transcribe_ms": 20}},
    {"route": "tool", "heard": "delta", "timings": {"total_ms": 999}},
    {"route": "timing", "heard": "epsilon", "timings": {}},
]


def test_latency_report_basic():
    report = latency_report(HANDMADE_RECORDS)
    assert report["count"] == 3
    assert report["stages"]["transcribe_ms"] == {"median": 20, "p95": 30, "max": 30, "samples": 3}
    assert report["stages"]["route_ms"] == {"median": 5, "p95": 5, "max": 5, "samples": 1}
    assert "act_ms" not in report["stages"]
    assert report["slowest"] == [
        {"heard": "beta", "total_ms": 300},
        {"heard": "gamma", "total_ms": 200},
        {"heard": "alpha", "total_ms": 100},
    ]


def test_latency_report_ignores_non_timing():
    report = latency_report([{"route": "model", "heard": "x", "timings": {"total_ms": 5}}])
    assert report == {"count": 0, "stages": {}, "slowest": []}


def test_latency_report_caps_slowest_at_five():
    records = [
        {"route": "timing", "heard": f"cmd{i}", "timings": {"total_ms": i * 10}}
        for i in range(8)
    ]
    report = latency_report(records)
    assert report["count"] == 8
    assert len(report["slowest"]) == 5
    totals = [entry["total_ms"] for entry in report["slowest"]]
    assert totals == sorted(totals, reverse=True)


def test_format_latency_empty():
    assert format_latency({"count": 0, "stages": {}, "slowest": []}) == "No timed commands yet."


def test_format_latency_non_empty():
    text = format_latency(latency_report(HANDMADE_RECORDS))
    assert "| Stage | Median | 95th | Max | Samples |" in text
    assert "transcribe_ms" in text
    assert "## Slowest commands" in text
    assert "- beta: 300 ms" in text


def test_main_latency_with_data(tmp_path, capsys):
    path = tmp_path / "actions.jsonl"
    log = ActionLog(path)
    log.append(
        ActionRecord(
            time="2024-01-01T00:00:00Z",
            heard="open browser",
            route="timing",
            timings={"total_ms": 1500, "transcribe_ms": 200},
        )
    )
    code = main(["latency", "--action-log", str(path)])
    assert code == 0
    captured = capsys.readouterr()
    assert "No timed commands yet." not in captured.out
    assert "open browser" in captured.out


def test_main_latency_empty_log(tmp_path, capsys):
    path = tmp_path / "actions.jsonl"
    path.write_text("")
    code = main(["latency", "--action-log", str(path)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "No timed commands yet."


def test_main_latency_missing_file(tmp_path, capsys):
    path = tmp_path / "missing.jsonl"
    code = main(["latency", "--action-log", str(path)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "No timed commands yet."
