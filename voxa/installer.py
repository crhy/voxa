from __future__ import annotations

import contextlib
import hashlib
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .agent.host import host_command  # noqa: E402

INSTALL_SCRIPT_URL = "https://ollama.com/install.sh"

# Pinned integrity pin for the official install script. This is a PLACEHOLDER
# digest committed by the project; a maintainer replaces it with the real
# sha256 of the current https://ollama.com/install.sh before each release, so
# any tampered or silently-changed remote script is rejected before it ever
# reaches the privileged shell.
EXPECTED_INSTALL_SHA256 = "56a79f3b115448072387c2480044bfa2cf8f90e4f5fddd8c943b4e051b81f80b"


class InstallerError(RuntimeError):
    pass


def is_flatpak() -> bool:
    return Path("/.flatpak-info").exists()


def ollama_installed(runner=subprocess.run) -> bool:
    """True when the `ollama` program exists on the host."""
    try:
        result = runner(
            host_command(["sh", "-c", "command -v ollama"]),
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _fetch_install_script(timeout: float = 15.0) -> str:
    request = urllib.request.Request(INSTALL_SCRIPT_URL, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise InstallerError(f"Could not download the Ollama install script: {exc}") from exc


def verify_install_script(script: str, expected_sha256: str) -> None:
    """Verify the fetched install script against a pinned sha256 digest.

    Raises ``InstallerError`` naming the mismatch (never the script bytes)
    when the bytes we fetched are not exactly the script the pin describes,
    so a network tamper or upstream change is caught before the bytes are
    handed to the privileged shell.
    """
    actual = hashlib.sha256(script.encode("utf-8")).hexdigest()
    if actual != expected_sha256:
        raise InstallerError(
            "The downloaded Ollama install script does not match its expected "
            f"sha256 digest (expected {expected_sha256}, got {actual}). "
            "Refusing to run an unverified script."
        )


def _privileged_shell_command() -> list[str]:
    # "-s" reads the script from stdin, so the elevated process only ever
    # sees bytes we already fetched ourselves (see install_ollama).
    base = ["pkexec", "sh", "-s"]
    if is_flatpak():
        # flatpak-spawn --host hands the command to the *host* pkexec, since
        # a sandboxed process cannot show a PolicyKit prompt or install onto
        # the host itself. Requires --talk-name=org.freedesktop.Flatpak.
        return ["flatpak-spawn", "--host", "--directory=/", *base]
    return base


def _feed_stdin(process: subprocess.Popen, script: str) -> None:
    assert process.stdin is not None
    try:
        process.stdin.write(script)
    except (BrokenPipeError, OSError):
        pass
    finally:
        with contextlib.suppress(OSError):
            process.stdin.close()


def install_ollama(
    *,
    on_output: Callable[[str], None],
    cancel_event: threading.Event,
) -> int:
    """Fetch the official Ollama install script and run it as root via pkexec.

    Fetching happens from inside this (network-enabled) app process; only the
    already-downloaded script text is handed to the privileged process, so
    the elevated shell never reaches the network itself. The fetched bytes
    are verified against ``EXPECTED_INSTALL_SHA256`` before anything runs with
    elevated privileges, so unverified bytes never reach root. The user sees a
    native PolicyKit password prompt from ``pkexec`` rather than a terminal.

    Requires ``pkexec`` on the host, and under Flatpak also requires
    ``--talk-name=org.freedesktop.Flatpak`` so ``flatpak-spawn --host`` can
    reach the host's ``pkexec``.
    """
    script = _fetch_install_script()
    verify_install_script(script, EXPECTED_INSTALL_SHA256)
    command = _privileged_shell_command()
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise InstallerError(
            "Could not find pkexec. Install Ollama manually from a terminal: "
            "download the Ollama install script (https://ollama.com/install.sh), "
            "inspect it, then run it as root yourself."
        ) from exc

    assert process.stdout is not None
    threading.Thread(target=_feed_stdin, args=(process, script), daemon=True).start()

    for line in process.stdout:
        if cancel_event.is_set():
            process.terminate()
            break
        stripped = line.rstrip("\n")
        if stripped:
            on_output(stripped)

    return process.wait()
