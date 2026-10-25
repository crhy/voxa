"""Capability 83 (manage files and folders): the find_file 'name' paraphrase, the
large-file and folder cases that need a file present on a fresh machine."""

import json
import subprocess
from pathlib import Path

from voxa.agent.intents import route
from voxatest.setup import run_setup


def test_heard_find_file_name_paraphrase_routes_cleanly():
    call = route("can you find the file name notes")
    assert call is not None
    assert call.tool == "find_file"
    assert call.args["name"] == "notes"


def test_heard_find_file_called_variants_keep_the_name():
    for heard, name in (
        ("find the file called notes", "notes"),
        ("find the file named notes", "notes"),
        ("find the file called notes please", "notes"),
    ):
        call = route(heard)
        assert call is not None
        assert call.tool == "find_file"
        assert call.args["name"] == name


def test_heard_large_file_phrasings_route_to_find_large_files():
    for heard in (
        "find files bigger than 10 megabytes",
        "find all files bigger than 10 megabytes",
        "locate files larger than 10 megabytes",
    ):
        call = route(heard)
        assert call is not None
        assert call.tool == "find_large_files"
        assert call.args["amount"] == "10"


def test_run_setup_makefile_records_truncate(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def host(cmd: list[str], *, stdin: bytes | None = None) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    error = run_setup(
        [{"makefile": "~/Documents/big.bin", "size": 11534336}],
        data_dir=tmp_path,
        host=host,
    )
    assert error == ""
    assert len(calls) == 1
    assert calls[0][0] == "sh"
    assert "truncate -s 11534336" in calls[0][-1]
    assert "$HOME/Documents/big.bin" in calls[0][-1]


def test_run_setup_makefile_failure_returns_error() -> None:
    error = run_setup(
        [{"makefile": "~/Documents/big.bin", "size": 10}],
        data_dir=Path("."),
        host=lambda cmd, *, stdin=None: subprocess.CompletedProcess(cmd, 1, b"", b"no space"),
    )
    assert "Create" in error


def test_shipped_find_file_cases_seed_a_notes_file() -> None:
    data = json.loads(Path("voxatest/data/test_cases_cap.json").read_text())
    cases = {c["id"]: c for c in data["cases"] if c["id"].startswith("cap83-e1")}
    assert set(cases) == {"cap83-e1", "cap83-e1-p1", "cap83-e1-p2"}
    for case in cases.values():
        assert case["expected_tool"] == "find_file"
        assert {"copy": "samples/notes-sample.txt", "to": "~/Documents/notes.txt"} in case["setup"]
        assert case["vm_only"] is True


def test_shipped_large_file_cases_seed_a_big_file() -> None:
    data = json.loads(Path("voxatest/data/test_cases_cap.json").read_text())
    cases = {c["id"]: c for c in data["cases"] if c["id"].startswith("cap83-e2")}
    assert set(cases) == {"cap83-e2", "cap83-e2-p1", "cap83-e2-p2"}
    for case in cases.values():
        assert case["expected_tool"] == "find_large_files"
        assert {"makefile": "~/Documents/big.bin", "size": 11534336} in case["setup"]
        assert case["vm_only"] is True


def test_shipped_folder_paraphrase_clears_the_existing_folder() -> None:
    data = json.loads(Path("voxatest/data/test_cases_cap.json").read_text())
    case = next(c for c in data["cases"] if c["id"] == "cap83-e3-p2")
    assert case["expected_tool"] == "make_folder"
    assert {"remove": "~/Documents/Test Folder"} in case["setup"]
    assert case["vm_only"] is True
