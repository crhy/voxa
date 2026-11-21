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

# Scripts Voxa is willing to hand to a root shell, keyed by SHA-256 of their exact
# bytes. Left empty on purpose: a digest may only be added after a human has read
# the script it pins. Run ``python tools/pin_ollama_installer.py`` to fetch the
# current script and print the line to paste here (see docs/notes/voxa-verify-before-root.md).
PINNED_SCRIPT_DIGESTS: dict[str, str] = {}


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


def script_digest(script: str) -> str:
    """SHA-256 of the install script's UTF-8 bytes, as 64 lowercase hex characters."""
    return hashlib.sha256(script.encode("utf-8")).hexdigest()


def verify_install_script(script: str, pinned: dict[str, str] | None = None) -> str:
    """Return the script's digest, or raise when that digest is not pinned.

    Hashing is pure ``hashlib`` over bytes already in memory: no network, no subprocess.
    Callers must not start a privileged process unless this function returns.
    """
    known = pinned if pinned is not None else PINNED_SCRIPT_DIGESTS
    digest = script_digest(script)
    if digest in known:
        return digest
    raise InstallerError(
        f"The Ollama installer downloaded from ollama.com has SHA-256 {digest}, which "
        "Voxa has not pinned, so Voxa will not run unverified content as root. Two "
        "honest routes: review that script and pin this digest in voxa/installer.py "
        "(tools/pin_ollama_installer.py prints the line), or install Ollama yourself "
        "from a terminal: curl -fsSL https://ollama.com/install.sh | sh"
    )


def _privileged_shell_command() -> list[str]:
    # "-s" reads the script from stdin, so the elevated process only ever
    # sees bytes we already fetched ourselves and verified (see install_ollama).
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

    The downloaded bytes are hashed and matched against ``PINNED_SCRIPT_DIGESTS``
    (``verify_install_script``) before any privileged process is started; only bytes
    whose digest someone has pinned are handed over, so the elevated shell never
    reaches the network itself and never runs mutable web content. The user sees a
    native PolicyKit password prompt from ``pkexec`` rather than a terminal.

    Requires ``pkexec`` on the host, and under Flatpak also requires
    ``--talk-name=org.freedesktop.Flatpak`` so ``flatpak-spawn --host`` can
    reach the host's ``pkexec``.
    """
    script = _fetch_install_script()
    verify_install_script(script)
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
            "curl -fsSL https://ollama.com/install.sh | sh"
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
