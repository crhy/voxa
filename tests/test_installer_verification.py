"""The gate that decides whether downloaded bytes may reach a root shell (#132).

Nothing here touches the network or starts a privileged process: fetching and
``subprocess.Popen`` stay patched, and verification is pure ``hashlib``.
"""

from __future__ import annotations

import io
import re
import threading
from unittest.mock import patch

import pytest

from voxa import installer

SCRIPT = "#!/bin/sh\necho installing ollama\n"
DIGEST = installer.script_digest(SCRIPT)


class RecordingStdin:
    """Keeps what the app wrote, and stays readable after _feed_stdin closes it."""

    def __init__(self) -> None:
        self.written = ""
        self.closed = False

    def write(self, data: str) -> None:
        self.written += data

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    """Stand-in for the privileged process, which is never started in tests."""

    def __init__(self, output: str = "", returncode: int = 0) -> None:
        self.stdin = RecordingStdin()
        self.stdout = io.StringIO(output)
        self._returncode = returncode
        self.terminated = False

    def wait(self) -> int:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True


def test_script_digest_is_stable_sha256_hex() -> None:
    digest = installer.script_digest(SCRIPT)
    assert digest == installer.script_digest(SCRIPT)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest != installer.script_digest(SCRIPT + "\n")


def test_pinned_digests_are_all_full_sha256_hex() -> None:
    """A truncated or mistyped key would silently disable the gate for everyone."""
    for digest, note in installer.PINNED_SCRIPT_DIGESTS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), digest
        assert note.strip(), digest


def test_verify_install_script_returns_the_digest_of_pinned_bytes() -> None:
    assert installer.verify_install_script(SCRIPT, pinned={DIGEST: "fixture"}) == DIGEST


def test_verify_install_script_names_the_digest_when_checking_fails() -> None:
    with pytest.raises(installer.InstallerError) as excinfo:
        installer.verify_install_script(SCRIPT, pinned={})
    message = str(excinfo.value)
    assert DIGEST in message
    assert "root" in message
    assert "voxa/installer.py" in message


def test_install_ollama_refuses_before_starting_any_process_when_digest_is_unknown() -> None:
    with (
        patch.object(installer, "_fetch_install_script", return_value=SCRIPT),
        patch.object(installer, "PINNED_SCRIPT_DIGESTS", {}),
        patch("subprocess.Popen") as popen,
        pytest.raises(installer.InstallerError),
    ):
        installer.install_ollama(on_output=lambda _line: None, cancel_event=threading.Event())

    popen.assert_not_called()


def test_install_ollama_feeds_the_exact_verified_bytes_and_keeps_the_flatpak_command() -> None:
    fake_process = FakeProcess()
    fed = threading.Event()
    real_feed = installer._feed_stdin

    def feed(process, script: str) -> None:
        real_feed(process, script)
        fed.set()

    with (
        patch.object(installer, "_fetch_install_script", return_value=SCRIPT),
        patch.object(installer, "PINNED_SCRIPT_DIGESTS", {DIGEST: "fixture"}),
        patch.object(installer, "is_flatpak", return_value=True),
        patch.object(installer, "_feed_stdin", feed),
        patch("subprocess.Popen", return_value=fake_process) as popen,
    ):
        exit_code = installer.install_ollama(on_output=lambda _line: None, cancel_event=threading.Event())

    assert exit_code == 0
    assert fed.wait(timeout=5.0) is True
    assert fake_process.stdin.written == SCRIPT
    assert fake_process.stdin.closed is True
    assert popen.call_args.args[0] == [
        "flatpak-spawn",
        "--host",
        "--directory=/",
        "pkexec",
        "sh",
        "-s",
    ]


def test_install_ollama_still_terminates_the_process_when_cancelled() -> None:
    fake_process = FakeProcess("line one\nline two\n")
    cancel_event = threading.Event()
    lines: list[str] = []

    def capture(line: str) -> None:
        lines.append(line)
        cancel_event.set()

    with (
        patch.object(installer, "_fetch_install_script", return_value=SCRIPT),
        patch.object(installer, "PINNED_SCRIPT_DIGESTS", {DIGEST: "fixture"}),
        patch("subprocess.Popen", return_value=fake_process),
    ):
        installer.install_ollama(on_output=capture, cancel_event=cancel_event)

    assert lines == ["line one"]
    assert fake_process.terminated is True
