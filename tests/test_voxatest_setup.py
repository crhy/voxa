import json
import subprocess
from pathlib import Path

from voxatest.cases import TestCase, load_cases, save_cases
from voxatest.setup import run_setup


class FakeHost:
    def __init__(self):
        self.calls: list[tuple[list[str], bytes | None]] = []

    def __call__(self, cmd: list[str], *, stdin: bytes | None = None) -> subprocess.CompletedProcess:
        self.calls.append((cmd, stdin))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")


def _case_with_setup(tmp_path: Path) -> TestCase:
    payload = {
        "cases": [
            {
                "id": "edit-selection",
                "category": "editing",
                "prompt": "make the selection bold",
                "setup": [
                    {"copy": "samples/x.txt", "to": "~/Documents/x.txt"},
                    {"open": "~/Documents/x.txt", "with": "pluma"},
                    {"wait": 3},
                    {"keys": "ctrl+a"},
                ],
                "vm_only": True,
            }
        ]
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cases = load_cases(path)
    assert len(cases) == 1
    return cases[0]


def test_load_case_with_setup_and_vm_only(tmp_path: Path) -> None:
    case = _case_with_setup(tmp_path)
    assert case.id == "edit-selection"
    assert case.vm_only is True
    assert len(case.setup) == 4
    assert case.setup[0] == {"copy": "samples/x.txt", "to": "~/Documents/x.txt"}
    assert case.setup[1] == {"open": "~/Documents/x.txt", "with": "pluma"}
    assert case.setup[2] == {"wait": 3}
    assert case.setup[3] == {"keys": "ctrl+a"}


def test_setup_round_trips_through_to_dict(tmp_path: Path) -> None:
    case = _case_with_setup(tmp_path)
    raw = case.to_dict()
    assert raw["vm_only"] is True
    assert raw["setup"] == case.setup
    path = tmp_path / "out.json"
    save_cases(path, [case])
    reloaded = load_cases(path)
    assert reloaded[0].setup == case.setup
    assert reloaded[0].vm_only is True


def test_run_setup_copy_records_command(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "samples").mkdir(parents=True)
    (data_dir / "samples" / "x.txt").write_text("hello", encoding="utf-8")
    host = FakeHost()
    error = run_setup(
        [{"copy": "samples/x.txt", "to": "~/Documents/x.txt"}],
        data_dir=data_dir,
        host=host,
    )
    assert error == ""
    assert len(host.calls) == 1
    cmd, stdin = host.calls[0]
    assert cmd[0] == "sh"
    assert "$HOME/Documents/x.txt" in cmd[-1]
    assert stdin == b"hello"


def test_run_setup_open_records_detached_command(tmp_path: Path) -> None:
    host = FakeHost()
    error = run_setup(
        [{"open": "~/Documents/x.txt", "with": "pluma"}],
        data_dir=tmp_path,
        host=host,
    )
    assert error == ""
    assert len(host.calls) == 1
    cmd, _stdin = host.calls[0]
    assert "setsid -f pluma" in cmd[-1]
    assert "$HOME/Documents/x.txt" in cmd[-1]


def test_run_setup_keys_records_xdotool(tmp_path: Path) -> None:
    host = FakeHost()
    error = run_setup(
        [{"keys": "ctrl+a"}],
        data_dir=tmp_path,
        host=host,
    )
    assert error == ""
    assert len(host.calls) == 1
    cmd, _stdin = host.calls[0]
    assert cmd == ["xdotool", "key", "--clearmodifiers", "ctrl+a"]


def test_run_setup_remove_records_rm(tmp_path: Path) -> None:
    host = FakeHost()
    error = run_setup(
        [{"remove": "~/Documents/x.txt"}],
        data_dir=tmp_path,
        host=host,
    )
    assert error == ""
    assert len(host.calls) == 1
    cmd, _stdin = host.calls[0]
    assert cmd[0] == "sh"
    assert "rm -f" in cmd[-1]
    assert "$HOME/Documents/x.txt" in cmd[-1]


def test_run_setup_wait_calls_sleep(tmp_path: Path) -> None:
    slept: list[float] = []
    host = FakeHost()
    error = run_setup(
        [{"wait": 2.5}],
        data_dir=tmp_path,
        host=host,
        sleep=lambda s: slept.append(s),
    )
    assert error == ""
    assert slept == [2.5]
    assert host.calls == []


def test_run_setup_unknown_step_returns_error(tmp_path: Path) -> None:
    host = FakeHost()
    error = run_setup(
        [{"frobnicate": True}],
        data_dir=tmp_path,
        host=host,
    )
    assert error != ""
    assert "Unknown" in error
    assert host.calls == []


def test_run_setup_copy_missing_sample_returns_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    host = FakeHost()
    error = run_setup(
        [{"copy": "samples/missing.txt", "to": "~/Documents/missing.txt"}],
        data_dir=data_dir,
        host=host,
    )
    assert error != ""
    assert "Could not read" in error
    assert host.calls == []


def test_run_setup_copy_failure_returns_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "samples").mkdir(parents=True)
    (data_dir / "samples" / "x.txt").write_text("hello", encoding="utf-8")

    class FailingHost:
        def __call__(self, cmd, *, stdin=None):
            return subprocess.CompletedProcess(cmd, 1, b"", b"permission denied")

    error = run_setup(
        [{"copy": "samples/x.txt", "to": "~/Documents/x.txt"}],
        data_dir=data_dir,
        host=FailingHost(),
    )
    assert error != ""
    assert "failed" in error
