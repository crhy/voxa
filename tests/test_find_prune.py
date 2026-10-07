from __future__ import annotations

import subprocess

import voxa.agent.tools.filemanage as filemanage
from voxa.agent.tools.filemanage import find_file, find_large_files


def _make_fake(records, results=None):
    def fake(command, stdin=None, check=True):
        records.append(list(command))
        key = tuple(command)
        stdout = results.get(key, "") if results else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return fake


def test_find_file_prune(monkeypatch):
    records = []
    results = {("sh", "-c", "echo $HOME"): "/home/u"}
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    find_file({"name": "budget 2026"})
    argv = next(cmd for cmd in records if cmd[:3] == ["timeout", "15", "find"])
    assert argv[:5] == ["timeout", "15", "find", "/home/u", "-xdev"]
    assert "-prune" in argv
    assert "-print" in argv
    assert "-path" not in argv
    assert argv[argv.index("-name") + 1] == ".*"
    assert argv[argv.index("-iname") + 1] == "*budget*2026*"


def test_find_large_files_prune(monkeypatch):
    records = []
    results = {("sh", "-c", "echo $HOME"): "/home/u"}
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    find_large_files({"amount": "2", "unit": "gigabytes"})
    argv = next(cmd for cmd in records if cmd[:3] == ["timeout", "15", "find"])
    assert argv[:5] == ["timeout", "15", "find", "/home/u", "-xdev"]
    assert "-prune" in argv
    assert "-print" in argv
    assert "-path" not in argv
    assert argv[argv.index("-name") + 1] == ".*"
    assert argv[argv.index("-size") + 1] == "+2147483648c"
