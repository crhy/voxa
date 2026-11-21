"""Find files by the words inside them (crhy/voxa#64's first step).

Everything runs through filemanage._run, so the search is checked as the exact argv it
would hand the host: no filesystem, no desktop, nothing moved.
"""

from __future__ import annotations

import subprocess

import voxa.agent.tools.filemanage as filemanage
from voxa.agent.intents import route
from voxa.agent.policy import RiskLevel
from voxa.agent.tools.filemanage import search_file_contents


def _grep_command(words: str, path: str) -> tuple[str, ...]:
    """The one bounded, read-only grep the tool is allowed to issue."""
    return (
        "timeout",
        "15",
        "grep",
        "-rIn",
        "-i",
        "--exclude-dir=.*",
        "--exclude-dir=node_modules",
        "--exclude-dir=__pycache__",
        "-m",
        "3",
        "--",
        words,
        path,
    )


def _make_fake(records, results, grep_returncode=0):
    def fake(command, stdin=None, check=True):
        records.append(list(command))
        key = tuple(command)
        stdout = results.get(key, "") if results else ""
        code = grep_returncode if key[0] == "timeout" else 0
        return subprocess.CompletedProcess(command, code, stdout, "")

    return fake


def test_test_document_is_found_by_content(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        _grep_command("lease", "/home/u/Documents"): "/home/u/Documents/lease.txt:4:Monthly lease 2026",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = search_file_contents({"text": "lease"})
    assert result.ok
    assert result.speech == "One file in Documents says lease: lease.txt: “Monthly lease 2026”."
    assert result.detail == "/home/u/Documents/lease.txt:4"


def test_no_match_is_honest(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = search_file_contents({"text": "zebra", "folder": "documents"})
    assert not result.ok
    assert result.speech == "No file in Documents says zebra."


def test_running_out_of_time_names_a_folder(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results, grep_returncode=124))
    result = search_file_contents({"text": "lease"})
    assert result.speech == "I ran out of time looking. Try naming a folder."


def test_search_uses_the_resolved_standard_folder(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOWNLOAD"): "/home/u/Downloads",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = search_file_contents({"text": "invoice", "folder": "my downloads folder"})
    assert records[-1] == list(_grep_command("invoice", "/home/u/Downloads"))
    assert result.speech == "No file in Downloads says invoice."


def test_unknown_folder_falls_back_to_documents(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = search_file_contents({"text": "lease", "folder": "attic"})
    assert records[-1] == list(_grep_command("lease", "/home/u/Documents"))
    assert result.speech == "No file in Documents says lease."


def test_standard_folder_hits_outrank_the_rest(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        _grep_command("lease", "/home/u"): (
            "/home/u/scratch/notes.txt:1:notes about the lease\n/home/u/Documents/lease.txt:2:The lease ends in June"
        ),
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = search_file_contents({"text": "lease", "folder": "home"})
    assert result.speech == (
        "I found 2 files in your home folder that say lease: "
        "lease.txt: “The lease ends in June” and notes.txt: “notes about the lease”."
    )


def test_more_than_three_hits_are_counted_not_recited(monkeypatch):
    records = []
    hits = "\n".join(f"/home/u/Documents/part{i}.txt:1:lease line {i}" for i in range(6))
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        _grep_command("lease", "/home/u/Documents"): hits,
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = search_file_contents({"text": "lease"})
    assert result.speech == (
        "I found 6 files in Documents that say lease: "
        "part0.txt: “lease line 0”, part1.txt: “lease line 1” and part2.txt: “lease line 2” and 3 more."
    )


def test_the_search_is_read_only(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        _grep_command("lease", "/home/u/Documents"): "/home/u/Documents/lease.txt:1:the lease",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    search_file_contents({"text": "lease"})
    assert records == [
        ["sh", "-c", "echo $HOME"],
        ["xdg-user-dir", "DOCUMENTS"],
        list(_grep_command("lease", "/home/u/Documents")),
    ]
    for command in records:
        assert not any(word in {"mv", "rm", "sed", "cp", "gio", "mkdir", "truncate"} for word in command)


def test_the_search_skips_binaries_and_stops_early(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    search_file_contents({"text": "-rf", "folder": "documents"})
    argv = records[-1]
    assert any(token.startswith("-") and "I" in token for token in argv)
    assert argv[argv.index("-m") + 1] == "3"
    assert argv.index("--") < argv.index("-rf")


def test_a_long_line_is_cut_not_recited(monkeypatch):
    records = []
    long_line = "lease " + "word " * 60
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        _grep_command("lease", "/home/u/Documents"): f"/home/u/Documents/notes.txt:1:{long_line}",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = search_file_contents({"text": "lease"})
    assert "…" in result.speech
    assert long_line not in result.speech


def test_the_tool_is_registered_read_only():
    tools = {tool.name: tool for tool in filemanage.file_manage_tools()}
    assert tools["search_file_contents"].risk == RiskLevel.READ_ONLY
    assert tools["search_file_contents"].required == ("text",)


def test_router_content_search_forms():
    for spoken in (
        "search my documents for lease",
        "find files containing lease",
        "which files mention lease",
    ):
        call = route(spoken)
        assert call is not None
        assert call.tool == "search_file_contents"
        assert call.args["text"]
    assert route("find the file called lease").tool == "find_file"
    assert route("find files bigger than 2 gigabytes").tool == "find_large_files"
