"""Secrets live in the system keyring, never in a file Voxa writes.

Pure Python: no GTK, no model, no settings file. The keyring is reached through a
backend object, so tests use a fake one and the real app picks its helper at
runtime: libsecret's ``secret-tool`` first, then the ``keyring`` command line.
When a machine has neither, the store refuses and nothing is kept anywhere;
there is no fallback that writes a secret to disk.

The value never appears in a spoken reply or in the action log: ``store``,
``describe`` and ``forget`` talk about the secret by name, and only ``lookup``
returns the value itself.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import Protocol

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)

__all__ = [
    "NAMESPACE",
    "LABEL",
    "SecretBackend",
    "SecretStore",
    "SecretToolBackend",
    "KeyringCliBackend",
    "available_backend",
    "default_store",
    "secret_key",
    "qualified_key",
]

NAMESPACE = "voxa"
LABEL = "Voxa"
KEY_MAX = 64
VALUE_MAX = 4096

NO_KEYRING = "I have no keyring on this machine, so I did not keep that."
NO_VALUE = "I did not catch that."
TOO_LONG = "That is too long to keep."


class SecretBackend(Protocol):
    """What a keyring helper has to provide: three operations and a name."""

    label: str

    def get(self, key: str) -> str: ...
    def set(self, key: str, value: str) -> bool: ...
    def delete(self, key: str) -> bool: ...


def secret_key(spoken: str) -> str:
    """A secret's name as spoken, made into a keyring key: "My WiFi Password" -> "wifi_password"."""
    key = (spoken or "").strip().lower()
    if key.startswith("my "):
        key = key[len("my "):]
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key[:KEY_MAX]


def qualified_key(key: str) -> str:
    """The namespaced key the helper sees, so Voxa's entries never collide with another program's."""
    return f"{NAMESPACE}/{key}"


class SecretStore:
    """Add, read back and forget a secret through an injected backend."""

    def __init__(self, backend: SecretBackend | None = None) -> None:
        self.backend = backend

    def available(self) -> bool:
        """True when a keyring helper is in reach; nothing is stored when it is False."""
        return self.backend is not None

    def key(self, name: str) -> str:
        key = secret_key(name)
        return qualified_key(key) if key else ""

    def store(self, name: str, value: str) -> ToolResult:
        key = secret_key(name)
        value = (value or "").strip()
        if not key or not value:
            return ToolResult.failure(NO_VALUE)
        if len(value) > VALUE_MAX:
            return ToolResult.failure(TOO_LONG)
        if self.backend is None:
            return ToolResult.failure(NO_KEYRING)
        if not self.backend.set(qualified_key(key), value):
            return ToolResult.failure(f"I could not keep {key} in your keyring.", detail="the keyring refused")
        return ToolResult.success(f"Saved {key} in your keyring.")

    def lookup(self, name: str) -> str:
        """The value itself, or "" when it is not there. The only path that returns it."""
        key = self.key(name)
        if not key or self.backend is None:
            return ""
        return self.backend.get(key)

    def describe(self, name: str) -> ToolResult:
        """Read back that a secret is kept, by name, without repeating it."""
        key = secret_key(name)
        if not key:
            return ToolResult.failure(NO_VALUE)
        if self.backend is None:
            return ToolResult.failure(NO_KEYRING)
        if not self.lookup(name):
            return ToolResult.failure(f"I do not have {key} saved.")
        return ToolResult.success(f"You have {key} saved in your keyring.")

    def forget(self, name: str) -> ToolResult:
        key = secret_key(name)
        if not key:
            return ToolResult.failure(NO_VALUE)
        if self.backend is None:
            return ToolResult.failure(NO_KEYRING)
        if not self.lookup(name):
            return ToolResult.failure(f"I do not have {key} saved.")
        if not self.backend.delete(qualified_key(key)):
            return ToolResult.failure(f"I could not forget {key}.", detail="the keyring refused")
        return ToolResult.success(f"Forgotten {key}.")


class SecretToolBackend:
    """libsecret's helper: ``secret-tool store|lookup|delete --label "Voxa <key>" <key> <value>``."""

    label = "secret-tool"

    def __init__(self, binary: str = "secret-tool") -> None:
        self.binary = binary

    def _run(self, args: list[str], secret: str = "") -> subprocess.CompletedProcess[str] | None:
        command = [self.binary, *args]
        if simulation.actions_simulated():
            logged = [part if part != secret else "(value)" for part in command] if secret else command
            log.info("simulated action: would run %s", logged)
            return subprocess.CompletedProcess(command, 0, "", "")
        try:
            return subprocess.run(
                host_command(command),
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            # A timeout's text repeats the command line, which carries the value here.
            log.warning("keyring helper failed: %s", exc.__class__.__name__ if secret else exc)
            return None

    def set(self, key: str, value: str) -> bool:
        done = self._run(["store", "--label", f"{LABEL} {key}", key, value], secret=value)
        return done is not None and done.returncode == 0

    def get(self, key: str) -> str:
        done = self._run(["lookup", key])
        if done is None or done.returncode != 0:
            return ""
        return done.stdout.strip()

    def delete(self, key: str) -> bool:
        done = self._run(["delete", key])
        return done is not None and done.returncode == 0


class KeyringCliBackend:
    """The Python keyring command line: ``keyring set|get|delete <service> <account>``.

    The value arrives on standard input, never on the command line, so it is not
    visible to anything watching running processes.
    """

    label = "keyring"
    SERVICE = NAMESPACE

    def __init__(self, binary: str = "keyring") -> None:
        self.binary = binary

    def _run(self, args: list[str], stdin: str = "") -> subprocess.CompletedProcess[str] | None:
        command = [self.binary, *args]
        if simulation.actions_simulated():
            log.info("simulated action: would run %s", command)
            return subprocess.CompletedProcess(command, 0, "", "")
        try:
            return subprocess.run(
                host_command(command),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("keyring helper failed: %s", exc)
            return None

    def set(self, key: str, value: str) -> bool:
        done = self._run(["set", self.SERVICE, key], stdin=value)
        return done is not None and done.returncode == 0

    def get(self, key: str) -> str:
        done = self._run(["get", self.SERVICE, key])
        if done is None or done.returncode != 0:
            return ""
        return done.stdout.strip()

    def delete(self, key: str) -> bool:
        done = self._run(["delete", self.SERVICE, key])
        return done is not None and done.returncode == 0


def available_backend() -> SecretBackend | None:
    """The keyring helper this machine has, or None when it has none."""
    if shutil.which("secret-tool"):
        return SecretToolBackend()
    if shutil.which("keyring"):
        return KeyringCliBackend()
    return None


def default_store() -> SecretStore:
    """The store the app uses: the helper found at runtime, or none at all."""
    return SecretStore(available_backend())
