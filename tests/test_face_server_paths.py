import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FACE = ROOT / "face-server"
SCRIPTS = [
    "run-face-server.sh",
    "run-sandboxed-py.sh",
    "run-sandboxed.sh",
    "stop-face-server.sh",
    "install.sh",
]


def test_no_developer_home_path():
    for name in SCRIPTS:
        assert "/home/rhy" not in (FACE / name).read_text(), name
    assert "/home/rhy" not in (FACE / "face_server.py").read_text()


def test_shell_scripts_parse():
    for name in SCRIPTS:
        subprocess.run(["bash", "-n", str(FACE / name)], check=True)


def test_face_server_command_never_voxa_neural(monkeypatch):
    pytest.importorskip("gi")
    import voxa.window as window

    monkeypatch.delenv("VOXA_FACE_SERVER", raising=False)
    monkeypatch.setattr(window.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(list(a), 0))
    command = window.face_server_command()
    assert command is not None
    assert "voxa-neural" not in command[0]
