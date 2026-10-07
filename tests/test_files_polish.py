from __future__ import annotations

import subprocess

import voxa.agent.tools.filemanage as filemanage
from voxa.agent.filematch import count_noun, speakable
from voxa.agent.intents import route


def _make_fake(records, results=None):
    def fake(command, stdin=None, check=True):
        records.append(list(command))
        key = tuple(command)
        stdout = results.get(key, "") if results else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    return fake


def test_count_noun():
    assert count_noun(1, "file") == "1 file"
    assert count_noun(3, "file") == "3 files"
    assert count_noun(1, "item") == "1 item"
    assert count_noun(0, "item") == "0 items"


def test_speakable():
    names = [
        "1953_buick_skylark.glb",
        "8321d560a31f9d507bdd632eeaf2fae0aee39a0e-1.jpeg",
        "IMG_0001.jpg",
        "Tax Return 2025.pdf",
        "a.txt",
    ]
    assert speakable(names) == ["1953 buick skylark", "Tax Return 2025"]


def test_list_folder_sentence(monkeypatch):
    names = [
        "1953_buick_skylark.glb",
        "8321d560a31f9d507bdd632eeaf2fae0aee39a0e-1.jpeg",
        "IMG_0001.jpg",
        "Tax Return 2025.pdf",
        "a.txt",
    ]
    records = []
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        ("xdg-user-dir", "DOWNLOAD"): "",
        ("ls", "-1A", "/home/u/Downloads"): "\n".join(names),
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = filemanage.list_folder({"folder": "downloads"})
    assert result.speech.startswith("5 items in downloads, including")


def test_find_large_files_one(monkeypatch):
    records = []
    find_cmd = (
        "timeout", "15", "find", "/home/u", "-xdev", "(", "-name", ".*",
        "-o", "-name", "node_modules", "-o", "-name", "__pycache__", ")",
        "-prune", "-o", "-type", "f", "-size", "+5368709120c", "-print",
    )
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        find_cmd: "/home/u/Downloads/big.iso",
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = filemanage.find_large_files({"amount": "5", "unit": "gigabytes"})
    assert "1 file bigger" in result.speech


def test_find_file_five_depths(monkeypatch):
    records = []
    paths = [
        "/home/u/x/y/z/deep.txt",
        "/home/u/x/y/mid.txt",
        "/home/u/Downloads/c.txt",
        "/home/u/Documents/b.txt",
        "/home/u/a.txt",
    ]
    find_cmd = (
        "timeout", "15", "find", "/home/u", "-xdev", "(", "-name", ".*",
        "-o", "-name", "node_modules", "-o", "-name", "__pycache__", ")",
        "-prune", "-o", "-iname", "*t*", "-print",
    )
    results = {
        ("sh", "-c", "echo $HOME"): "/home/u",
        find_cmd: "\n".join(paths),
    }
    monkeypatch.setattr(filemanage, "_run", _make_fake(records, results))
    result = filemanage.find_file({"name": "t"})
    assert result.speech.startswith("I found 5. The closest are")
    assert "a.txt in u" in result.speech
    assert "b.txt in Documents" in result.speech
    assert "c.txt in Downloads" in result.speech


def test_router_where_is_the_file():
    call = route("where is the file holiday plans")
    assert call is not None
    assert call.tool == "find_file"
    assert call.args["name"] == "holiday plans"


def test_router_gas_station_unchanged():
    call = route("where is the nearest gas station")
    assert call is not None
    assert call.tool != "find_file"
