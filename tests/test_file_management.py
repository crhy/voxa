from __future__ import annotations

import subprocess

import voxa.agent.tools.filemanage as filemanage
from voxa.agent.filematch import best_match, folder_key, normalise, size_bytes, spoken_list
from voxa.agent.intents import route
from voxa.agent.tools.filemanage import copy_file, empty_trash, find_file, find_large_files, move_file, trash_file


def _make_fake(records, results=None, raises=None):
    def fake(command, stdin=None, check=True):
        records.append(list(command))
        key = tuple(command)
        if raises is not None and key == raises[0]:
            raise raises[1]
        stdout = results.get(key, "") if results else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return fake


def test_folder_key():
    assert folder_key("home") == ""
    assert folder_key("home directory") == ""
    assert folder_key("my home folder") == ""
    assert folder_key("Documents") == "DOCUMENTS"
    assert folder_key("the documents folder") == "DOCUMENTS"
    assert folder_key("my downloads") == "DOWNLOAD"
    assert folder_key("download") == "DOWNLOAD"
    assert folder_key("movies") == "VIDEOS"
    assert folder_key("photos") == "PICTURES"
    assert folder_key("attic") is None
    assert folder_key("left") is None


def test_normalise():
    assert normalise("Report Final.pdf") == "report final"
    assert normalise("holiday-video.MP4") == "holiday video"
    assert normalise("budget 2026!!") == "budget 2026"
    assert normalise("notes") == "notes"


def test_best_match():
    names = ["Report Final.pdf", "notes.odt", "holiday video.mp4"]
    assert best_match("report final", names) == "Report Final.pdf"
    assert best_match("holiday", names) == "holiday video.mp4"
    assert best_match("holiday video dot mp4", names) == "holiday video.mp4"
    assert best_match("report dot pdf", ["report.pdf", "report.odt"]) == "report.pdf"
    assert best_match("zebra", names) is None
    assert best_match("", names) is None
    assert best_match("report", []) is None


def test_size_bytes():
    assert size_bytes("2", "gigabytes") == 2 * 1024**3
    assert size_bytes("1", "gigabyte") == 1024**3
    assert size_bytes("3", "mb") == 3 * 1024**2
    assert size_bytes("5", "megabytes") == 5 * 1024**2
    assert size_bytes("10", "megabyte") == 10 * 1024**2


def test_spoken_list():
    assert spoken_list([]) == ""
    assert spoken_list(["a"]) == "a"
    assert spoken_list(["a", "b"]) == "a and b"
    assert spoken_list(["a", "b", "c"]) == "a, b and c"
    assert spoken_list(["a", "b", "c", "d"]) == "a, b and c and 1 more"
    assert spoken_list(["a", "b", "c", "d", "e", "f", "g"]) == "a, b and c and 4 more"


def test_copy_file(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOWNLOAD"): "/home/u/Downloads",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        ("ls", "-1A", "/home/u/Downloads"): "Report Final.pdf\nnotes.odt",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = copy_file({"name": "report final", "source": "Downloads", "destination": "Documents"})
    assert records == [
        ["sh", "-c", "echo $HOME"],
        ["xdg-user-dir", "DOWNLOAD"],
        ["sh", "-c", "echo $HOME"],
        ["xdg-user-dir", "DOCUMENTS"],
        ["ls", "-1A", "/home/u/Downloads"],
        ["gio", "copy", "/home/u/Downloads/Report Final.pdf", "/home/u/Documents/"],
    ]
    assert result.speech == "Copied Report Final.pdf to Documents."


def test_move_file(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOWNLOAD"): "/home/u/Downloads",
        ("xdg-user-dir", "VIDEOS"): "/home/u/Videos",
        ("ls", "-1A", "/home/u/Downloads"): "holiday video.mp4",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = move_file({"name": "holiday video", "source": "Downloads", "destination": "Videos"})
    assert ["gio", "move", "/home/u/Downloads/holiday video.mp4", "/home/u/Videos/"] in records
    assert result.speech == "Moved holiday video.mp4 to Videos."


def test_trash_file(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        ("ls", "-1A", "/home/u/Documents"): "old notes.txt",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = trash_file({"name": "old notes", "folder": "Documents"})
    assert ["gio", "trash", "/home/u/Documents/old notes.txt"] in records
    assert result.speech == "Moved old notes.txt to the trash."


def test_empty_trash(monkeypatch):
    records = []
    monkeypatch.setattr(filemanage, "_run", _make_fake(records))
    result = empty_trash({})
    assert ["gio", "trash", "--empty"] in records
    assert result.speech == "The trash is empty."


def test_unknown_folder(monkeypatch):
    records = []
    monkeypatch.setattr(filemanage, "_run", _make_fake(records))
    result = copy_file({"name": "report", "source": "attic", "destination": "Documents"})
    assert result.speech == "I do not know the folder attic."


def test_no_match(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        ("ls", "-1A", "/home/u/Documents"): "budget.pdf",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = trash_file({"name": "zebra", "folder": "Documents"})
    assert result.speech == "I could not find zebra in Documents."


def test_failing_gio(monkeypatch):
    records = []
    error = subprocess.CalledProcessError(1, ["gio"], stderr="gio: Error: File exists\nmore")
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOWNLOAD"): "/home/u/Downloads",
        ("xdg-user-dir", "DOCUMENTS"): "/home/u/Documents",
        ("ls", "-1A", "/home/u/Downloads"): "report.pdf",
    }
    monkeypatch.setattr(
        filemanage,
        "_run",
        _make_fake(
            records,
            results,
            raises=(("gio", "copy", "/home/u/Downloads/report.pdf", "/home/u/Documents/"), error),
        ),
    )
    result = copy_file({"name": "report", "source": "Downloads", "destination": "Documents"})
    assert result.speech == "That did not work: gio: Error: File exists"


def test_find_file_none(monkeypatch):
    records = []
    results = {("sh", "-c", "echo $HOME"): "/home/u"}
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_file({"name": "zebra"})
    assert result.speech == "I could not find a file called zebra."


def test_find_file_one(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
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
            "*budget*",
            "-print",
        ): "/home/u/Documents/budget.pdf",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_file({"name": "budget"})
    assert ["gio", "open", "/home/u/Documents"] in records
    assert result.speech == "I found 1: budget.pdf in Documents."


def test_find_file_three(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
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
            "*report*",
            "-print",
        ): "/home/u/Documents/report.pdf\n/home/u/Downloads/report.odt\n/home/u/Desktop/report.txt",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_file({"name": "report"})
    assert result.speech == (
        "I found 3: report.pdf in Documents, report.odt in Downloads and report.txt in Desktop."
    )


def test_find_large_files(monkeypatch):
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
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
            f"+{2 * 1024 ** 3}c",
            "-print",
        ): "/home/u/Documents/big.iso\n/home/u/Downloads/huge.iso",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_large_files({"amount": "2", "unit": "gigabytes"})
    assert result.speech == "I found 2 files bigger than 2 gigabytes: big.iso and huge.iso."


def test_find_large_files_none(monkeypatch):
    records = []
    results = {("sh", "-c", "echo $HOME"): "/home/u"}
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = find_large_files({"amount": "2", "unit": "gigabytes"})
    assert result.speech == "No files are bigger than that."


def test_router_copy():
    call = route("Copy report from Downloads to Documents")
    assert call.tool == "copy_file"
    assert call.args == {"name": "report", "source": "Downloads", "destination": "Documents"}


def test_router_move():
    call = route("Move holiday video from Downloads to Videos")
    assert call.tool == "move_file"
    assert call.args == {"name": "holiday video", "source": "Downloads", "destination": "Videos"}


def test_router_trash():
    call = route("Delete old notes from Documents")
    assert call.tool == "trash_file"
    assert call.args == {"name": "old notes", "folder": "Documents"}


def test_router_empty_trash():
    call = route("Empty trash")
    assert call.tool == "empty_trash"
    assert call.args == {}


def test_router_find():
    call = route("Find the file called budget")
    assert call.tool == "find_file"
    assert call.args == {"name": "budget"}


def test_router_large():
    call = route("Find files bigger than 2 gigabytes")
    assert call.tool == "find_large_files"
    assert call.args == {"amount": "2", "unit": "gigabytes"}


def test_router_not_move_window():
    call = route("move the window from left to right")
    assert call is None or call.tool != "move_file"
