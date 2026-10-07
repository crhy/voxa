from __future__ import annotations

import subprocess

import voxa.agent.tools.filemanage as filemanage
from voxa.agent.intents import ROUTED_TOOLS, ToolCall, route
from voxa.agent.policy import RiskLevel
from voxa.agent.tools.filemanage import file_manage_tools, open_file


def _make_fake(records, results=None, raises=None):
    def fake(command, stdin=None, check=True):
        records.append(list(command))
        key = tuple(command)
        if raises is not None and key == raises[0]:
            raise raises[1]
        stdout = results.get(key, "") if results else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return fake


def test_open_file_registered():
    tools = {tool.name: tool for tool in file_manage_tools()}
    names = [tool.name for tool in file_manage_tools()]
    assert "open_file" in names
    assert names.index("open_file") < names.index("open_folder")
    assert tools["open_file"].risk == RiskLevel.REVERSIBLE
    assert tools["open_file"].handler is open_file
    assert tools["open_file"].required == ("name", "folder")
    assert tools["open_file"].parameters == {"name": "the file to open", "folder": "the folder it is in"}


def test_open_file_in_documents(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        ("ls", "-1A", "/home/u/Documents"): "Budget 2026.ods\nnotes.odt",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = open_file({"name": "budget", "folder": "Documents"})
    assert ["gio", "open", "/home/u/Documents/Budget 2026.ods"] in records
    assert result.speech == "Opening Budget 2026.ods."


def test_open_file_no_match_in_folder(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        ("ls", "-1A", "/home/u/Documents"): "budget.pdf",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = open_file({"name": "zebra", "folder": "Documents"})
    assert result.speech == "I could not find zebra in Documents."


def test_open_file_unknown_folder(monkeypatch):
    records = []
    monkeypatch.setattr(filemanage, "_run", _make_fake(records))
    result = open_file({"name": "budget", "folder": "attic"})
    assert records == []
    assert result.speech == "I do not know the folder attic."


def test_open_file_downloads_only(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DESKTOP"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u",
        ("xdg-user-dir", "DOWNLOAD"): "/home/u/Downloads",
        ("ls", "-1A", "/home/u/Desktop"): "notes.txt",
        ("ls", "-1A", "/home/u/Documents"): "budget.pdf",
        ("ls", "-1A", "/home/u/Downloads"): "invoice.pdf",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = open_file({"name": "invoice", "folder": ""})
    assert ["gio", "open", "/home/u/Downloads/invoice.pdf"] in records
    assert result.speech == "Opening invoice.pdf."


def test_open_file_nowhere_found(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DESKTOP"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u",
        ("xdg-user-dir", "DOWNLOAD"): "/home/u/Downloads",
        ("ls", "-1A", "/home/u/Desktop"): "notes.txt",
        ("ls", "-1A", "/home/u/Documents"): "budget.pdf",
        ("ls", "-1A", "/home/u/Downloads"): "invoice.pdf",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = open_file({"name": "zebra", "folder": ""})
    assert not any(command[:2] == ["gio", "open"] for command in records)
    assert result.speech == "I could not find a file called zebra on the Desktop, in Documents or in Downloads."


def test_open_file_gio_failure(monkeypatch):
    records = []
    error = subprocess.CalledProcessError(1, ["gio", "open", "/home/u/Documents/budget.pdf"], stderr=b"boom")
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        ("ls", "-1A", "/home/u/Documents"): "budget.pdf",
    }
    monkeypatch.setattr(
        filemanage,
        "_run",
        _make_fake(
            records,
            results,
            raises=(("gio", "open", "/home/u/Documents/budget.pdf"), error),
        ),
    )
    result = open_file({"name": "budget", "folder": "Documents"})
    assert ["gio", "open", "/home/u/Documents/budget.pdf"] in records
    assert result.speech == "That did not work: boom"


def test_open_file_routing():
    assert route("open the file budget in Documents") == ToolCall("open_file", {"name": "budget", "folder": "Documents"})
    assert route("open the file budget") == ToolCall("open_file", {"name": "budget", "folder": ""})
    assert "open_file" in ROUTED_TOOLS


def test_open_file_preserves_existing_phrases():
    assert route("open file") == ToolCall("file_dialog", {"action": "load"})
    assert route("open the file manager") == ToolCall("open_app", {"name": "file manager"})
    assert route("open my file manager") == ToolCall("open_app", {"name": "my file manager"})
    assert route("open LibreOffice") == ToolCall("open_app", {"name": "LibreOffice"})
    assert route("open my email") == ToolCall("open_site", {"name": "gmail"})
    assert route("open the downloads folder") == ToolCall("open_folder", {"folder": "downloads"})
    assert route("open my home directory") == ToolCall("open_folder", {"folder": "home"})
