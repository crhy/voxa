from __future__ import annotations

import subprocess

import voxa.agent.tools.filemanage as filemanage
from voxa.agent.filematch import resolve_folder
from voxa.agent.tools.filemanage import copy_file, find_file, find_large_files


def _make_fake(records, results=None):
    def fake(command, stdin=None, check=True):
        records.append(list(command))
        code, stdout, stderr = (results or {}).get(tuple(command), (0, "", ""))
        return subprocess.CompletedProcess(command, code, stdout, stderr)

    return fake


def test_resolve_folder():
    assert resolve_folder("DOCUMENTS", "/home/u", "/home/u") == "/home/u/Documents"
    assert resolve_folder("DOCUMENTS", "/home/u", "/home/u/") == "/home/u/Documents"
    assert resolve_folder("DOWNLOAD", "/home/u", "") == "/home/u/Downloads"
    assert resolve_folder("VIDEOS", "/home/u", "/data/Films") == "/data/Films"
    assert resolve_folder("", "/home/u", "anything") == "/home/u"


def test_folder_falls_back_to_standard_name(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): (0, "/home/u", ""),
        ("xdg-user-dir", "DOCUMENTS"): (0, "/home/u", ""),
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    assert filemanage._folder("documents") == "/home/u/Documents"


def test_copy_on_machine_without_userdirs(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): (0, "/home/u", ""),
        ("xdg-user-dir", "DOWNLOAD"): (0, "/home/u", ""),
        ("xdg-user-dir", "DOCUMENTS"): (0, "/home/u", ""),
        ("ls", "-1A", "/home/u/Downloads"): (0, "report.pdf", ""),
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    copy_file({"name": "report", "source": "Downloads", "destination": "Documents"})
    assert ["gio", "copy", "/home/u/Downloads/report.pdf", "/home/u/Documents/"] in records


def test_find_large_files_partial_errors(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): (0, "/home/u", ""),
        (
            "timeout",
            "15",
            "find",
            "/home/u",
            "-xdev",
            "(",
            "-name",
            ".*",
            "-o",
            "-name",
            "node_modules",
            "-o",
            "-name",
            "__pycache__",
            ")",
            "-prune",
            "-o",
            "-type",
            "f",
            "-size",
            f"+{5 * 1024 ** 3}c",
            "-print",
        ): (1, "/home/u/a.iso\n/home/u/b.iso", "find: '/home/u/partial': Permission denied"),
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_large_files({"amount": "5", "unit": "gigabytes"})
    assert result.speech == "I found 2 files bigger than 5 gigabytes: a.iso and b.iso."
    find_command = next(cmd for cmd in records if cmd[:3] == ["timeout", "15", "find"])
    assert find_command[:3] == ["timeout", "15", "find"]


def test_find_file_out_of_time(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): (0, "/home/u", ""),
        (
            "timeout",
            "15",
            "find",
            "/home/u",
            "-xdev",
            "(",
            "-name",
            ".*",
            "-o",
            "-name",
            "node_modules",
            "-o",
            "-name",
            "__pycache__",
            ")",
            "-prune",
            "-o",
            "-iname",
            "*zebra*",
            "-print",
        ): (124, "", ""),
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_file({"name": "zebra"})
    assert result.speech == "I ran out of time looking. Try naming a folder."
