from __future__ import annotations

import subprocess

import voxa.agent.tools.filemanage as filemanage
from voxa.agent.hearing import normalize
from voxa.agent.intents import route
from voxa.agent.tools.filemanage import find_large_files


def test_list_done_past_tense() -> None:
    marked = route("marked finish the report is done")
    assert marked is not None
    assert marked.tool == "list_done"
    assert marked.args["item"] == "finish the report"
    checked = route("checked off buy stamps on my kitchen list")
    assert checked is not None
    assert checked.tool == "list_done"
    assert checked.args["item"] == "buy stamps"
    assert checked.args["list"] == "kitchen"
    crossed = route("crossed off the milk from my shopping list")
    assert crossed is not None
    assert crossed.tool == "list_done"
    assert crossed.args["item"] == "the milk"


def test_live_music_hearing() -> None:
    fixed = normalize("light music tonight in my area")
    assert "live music" in fixed
    call = route(fixed)
    assert call is not None
    assert call.tool == "local_events"
    assert call.args["what"] == "live music"


def _make_fake(records, results=None):
    def fake(command, stdin=None, check=True):
        records.append(list(command))
        key = tuple(command)
        stdout = results.get(key, "") if results else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return fake


def test_find_large_files_one(monkeypatch) -> None:
    records = []
    find_cmd = (
        "timeout", "15", "find", "/home/u", "-xdev", "(", "-name", ".*",
        "-o", "-name", "node_modules", "-o", "-name", "__pycache__", ")",
        "-prune", "-o", "-type", "f", "-size", "+2097152c", "-printf", "%s\\t%p\\n",
    )
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        find_cmd: "3221225472\t/home/u/Downloads/big.iso",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_large_files({"amount": "2", "unit": "MB"})
    assert result.speech == "I found one file bigger than 2 megabytes: big.iso."


def test_find_large_files_many(monkeypatch) -> None:
    records = []
    find_cmd = (
        "timeout", "15", "find", "/home/u", "-xdev", "(", "-name", ".*",
        "-o", "-name", "node_modules", "-o", "-name", "__pycache__", ")",
        "-prune", "-o", "-type", "f", "-size", "+2147483648c", "-printf", "%s\\t%p\\n",
    )
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        find_cmd: "3221225472\t/home/u/Documents/big.iso\n5368709120\t/home/u/Downloads/huge.iso",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_large_files({"amount": "2", "unit": "GB"})
    assert result.speech == "I found 2 files bigger than 2 gigabytes. The biggest is huge.iso."
