from __future__ import annotations

import subprocess

from voxa.agent.intents import route
from voxa.agent.tools import filemanage

NEW_TOOLS = {"open_folder", "list_folder", "rename_file", "make_folder"}


def install_run(monkeypatch, listing):
    calls = []

    def run(command, stdin=None, check=True):
        calls.append(list(command))
        if command[:2] == ["sh", "-c"]:
            return subprocess.CompletedProcess(command, 0, "/home/u", "")
        if command[0] == "xdg-user-dir":
            return subprocess.CompletedProcess(command, 0, "/home/u", "")
        if command[0] == "ls":
            return subprocess.CompletedProcess(command, 0, "\n".join(listing), "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(filemanage, "_run", run)
    return calls


def install_failing_run(monkeypatch, command_to_fail, stderr):
    def run(command, stdin=None, check=True):
        if command[:2] == ["sh", "-c"]:
            return subprocess.CompletedProcess(command, 0, "/home/u", "")
        if command[0] == "xdg-user-dir":
            return subprocess.CompletedProcess(command, 0, "/home/u", "")
        if command[0] == "ls":
            return subprocess.CompletedProcess(command, 0, "", "")
        if check and command == command_to_fail:
            raise subprocess.CalledProcessError(1, command, stderr=stderr)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(filemanage, "_run", run)


def test_open_folder_argv_and_sentence(monkeypatch):
    calls = install_run(monkeypatch, [])
    result = filemanage.open_folder({"folder": "my downloads"})
    assert result.ok
    assert result.speech == "Opening my downloads."
    assert calls[-1] == ["gio", "open", "/home/u/Downloads"]


def test_list_folder_names_and_sentence(monkeypatch):
    calls = install_run(monkeypatch, ["alpha.pdf", "beta.txt", "gamma.doc"])
    result = filemanage.list_folder({"folder": "my documents"})
    assert result.ok
    assert result.speech == (
        "3 items in my documents: alpha.pdf, beta.txt and gamma.doc."
    )
    assert calls[-1] == ["ls", "-1A", "/home/u/Documents"]


def test_list_folder_empty(monkeypatch):
    calls = install_run(monkeypatch, [])
    result = filemanage.list_folder({"folder": "my documents"})
    assert result.ok
    assert result.speech == "Documents is empty."
    assert calls[-1] == ["ls", "-1A", "/home/u/Documents"]


def test_rename_file_keeps_extension(monkeypatch):
    calls = install_run(monkeypatch, ["Report.pdf"])
    result = filemanage.rename_file(
        {"name": "report", "new_name": "final report", "folder": "documents"}
    )
    assert result.ok
    assert result.speech == "Renamed Report.pdf to final report.pdf."
    assert calls[-1] == ["gio", "rename", "/home/u/Documents/Report.pdf", "final report.pdf"]


def test_make_folder_argv_and_sentence(monkeypatch):
    calls = install_run(monkeypatch, [])
    result = filemanage.make_folder({"name": "taxes", "folder": "documents"})
    assert result.ok
    assert result.speech == "Created the folder Taxes in documents."
    assert calls[-1] == ["mkdir", "--", "/home/u/Documents/Taxes"]


def test_make_folder_failure(monkeypatch):
    command = ["mkdir", "--", "/home/u/Documents/Taxes"]
    install_failing_run(monkeypatch, command, "mkdir: cannot create directory: File exists")
    result = filemanage.make_folder({"name": "taxes", "folder": "documents"})
    assert not result.ok
    assert "File exists" in result.speech


def test_router_open_folder():
    call = route("open my documents folder")
    assert call is not None
    assert call.tool == "open_folder"
    assert call.args == {"folder": "documents"}


def test_router_list_folder():
    call = route("what's in my documents")
    assert call is not None
    assert call.tool == "list_folder"
    assert call.args == {"folder": "documents"}


def test_router_rename_file():
    call = route("rename report in documents to final report")
    assert call is not None
    assert call.tool == "rename_file"
    assert call.args == {"name": "report", "folder": "documents", "new_name": "final report"}


def test_router_make_folder():
    call = route("create a folder called taxes in documents")
    assert call is not None
    assert call.tool == "make_folder"
    assert call.args == {"name": "taxes", "folder": "documents"}


def test_router_keeps_other_phrases():
    for phrase in ("open my email", "open downloads", "what's in the news"):
        call = route(phrase)
        assert call is None or call.tool not in NEW_TOOLS
