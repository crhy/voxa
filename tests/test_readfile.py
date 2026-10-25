from __future__ import annotations

import pytest

from voxa.agent.tools import readfile


@pytest.fixture
def host(monkeypatch):
    files: dict[str, bytes] = {}
    monkeypatch.setattr(readfile, "host_read", lambda path: files.get(path, None) if path in files else None)
    monkeypatch.setattr(
        readfile,
        "host_list",
        lambda folder: sorted(p.rsplit("/", 1)[1] for p in files if p.rsplit("/", 1)[0] == folder),
    )
    monkeypatch.setattr(readfile.os.path, "expanduser", lambda p: p.replace("~", "/home/u", 1))
    return files


def test_line_breaks_become_pauses():
    poem = b"Once upon a midnight dreary\nWhile I pondered, weak and weary,\n\nOver many a quaint and curious volume"
    assert readfile.plain_text(poem) == (
        "Once upon a midnight dreary. While I pondered, weak and weary, Over many a quaint and curious volume."
    )


def test_reads_a_file_found_by_spoken_name_in_documents(host):
    host["/home/u/Documents/Meeting Notes.txt"] = b"First line.\nSecond line."
    result = readfile.read_text_file({"name": "meeting notes"})
    assert result.ok and result.speech == "First line. Second line."
    assert result.detail == "/home/u/Documents/Meeting Notes.txt"


def test_reads_an_absolute_path_and_strips_markdown(host):
    host["/srv/intro.md"] = b"# Welcome\n\n- **Spaced** Linux\n> quoted `code`"
    assert readfile.read_text_file({"name": "/srv/intro.md"}).speech == "Welcome. Spaced Linux. quoted code."


def test_reads_the_file_selected_in_caja(host, monkeypatch):
    host["/home/u/Desktop/a b.txt"] = b"Hello."
    monkeypatch.setattr(readfile, "selected_text", lambda: "file:///home/u/Desktop/a%20b.txt\n")
    assert readfile.read_text_file({}).speech == "Hello."


def test_selection_that_is_not_a_file_asks_for_a_file(host, monkeypatch):
    monkeypatch.setattr(readfile, "selected_text", lambda: "just some words")
    result = readfile.read_text_file({})
    assert not result.ok and "Select a text file" in result.speech


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"\x00\x01binary", "does not look like a text file"), (b"\xff\xfe\xfa", "does not look like a text file"),
     (b"   \n", "is empty"), (b"x" * (readfile.MAX_BYTES + 1), "too long")],
)
def test_refuses_binary_empty_and_huge_files(host, content, message):
    host["/home/u/f.txt"] = content
    result = readfile.read_text_file({"name": "/home/u/f.txt"})
    assert not result.ok and message in result.speech


def test_missing_file(host):
    result = readfile.read_text_file({"name": "nothing like it"})
    assert not result.ok and "could not find" in result.speech


def test_only_text_files_are_candidates(host):
    host["/home/u/report.pdf"] = b"%PDF"
    assert readfile.find_file("report", home="/home/u") is None


def test_host_read_caps_the_bytes_it_asks_for(monkeypatch):
    seen = []

    class Done:
        returncode = 0
        stdout = b"ok"

    monkeypatch.setattr(readfile.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or Done())
    assert readfile.host_read("/x") == b"ok"
    command = seen[0]
    assert command[command.index("head"):] == ["head", "-c", str(readfile.MAX_BYTES + 1), "--", "/x"]
