"""The keyring wrapper (#41): secrets go to the system keyring and nowhere else.

A fake backend stands in for the keyring; the real helpers are only probed for,
never run. What must hold: keys are namespaced, the value never reaches the
settings file or the action log, and a machine with no keyring helper refuses
instead of keeping a copy of the secret somewhere.
"""

from __future__ import annotations

import subprocess

import pytest

from voxa.agent import secrets
from voxa.agent.actionlog import ActionLog, ActionRecord
from voxa.config import ConfigStore, Settings

SECRET = "hunter2keepme"


class FakeBackend:
    """A keyring in memory: records the keys it was given."""

    label = "fake"

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str) -> str:
        return self.data.get(key, "")

    def set(self, key: str, value: str) -> bool:
        self.data[key] = value
        return True

    def delete(self, key: str) -> bool:
        return self.data.pop(key, None) is not None


class RefusingBackend(FakeBackend):
    def set(self, key: str, value: str) -> bool:
        return False


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


def test_spoken_key_becomes_a_keyring_key(backend):
    store = secrets.SecretStore(backend)
    assert store.key("My WiFi Password") == "voxa/wifi_password"
    assert secrets.secret_key("the door code!") == "the_door_code"


def test_store_read_forget_round_trip(backend):
    store = secrets.SecretStore(backend)
    assert store.store("My WiFi Password", SECRET).ok
    assert backend.data == {"voxa/wifi_password": SECRET}
    assert store.lookup("wifi password") == SECRET
    assert store.forget("wifi password").ok
    assert backend.data == {}
    assert store.lookup("wifi password") == ""


def test_read_back_names_the_secret_not_its_value(backend):
    store = secrets.SecretStore(backend)
    store.store("My WiFi Password", SECRET)
    result = store.describe("wifi password")
    assert result.ok
    assert result.speech == "You have wifi_password saved in your keyring."
    assert SECRET not in result.speech + result.detail


def test_store_never_reveals_the_value(backend):
    store = secrets.SecretStore(backend)
    result = store.store("My WiFi Password", SECRET)
    assert SECRET not in result.speech + result.detail


def test_forget_without_a_secret(backend):
    store = secrets.SecretStore(backend)
    result = store.forget("wifi password")
    assert not result.ok
    assert result.speech == "I do not have wifi_password saved."


def test_empty_value_is_refused(backend):
    store = secrets.SecretStore(backend)
    assert not store.store("wifi password", "   ").ok
    assert backend.data == {}


def test_an_absurd_value_is_refused_not_truncated(backend):
    store = secrets.SecretStore(backend)
    assert not store.store("wifi password", "x" * (secrets.VALUE_MAX + 1)).ok
    assert backend.data == {}


def test_keyring_refusing_is_reported(backend):
    store = secrets.SecretStore(RefusingBackend())
    result = store.store("wifi password", SECRET)
    assert not result.ok
    assert result.speech == "I could not keep wifi_password in your keyring."
    assert result.detail == "the keyring refused"


def test_no_keyring_helper_means_no_storage(monkeypatch, tmp_path):
    """This machine has neither secret-tool nor the keyring CLI: nothing is kept anywhere."""
    monkeypatch.setattr(secrets.shutil, "which", lambda name: None)
    for env in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "HOME"):
        monkeypatch.setenv(env, str(tmp_path))
    assert secrets.available_backend() is None
    store = secrets.default_store()
    assert not store.available()
    result = store.store("wifi password", SECRET)
    assert not result.ok
    assert result.speech == secrets.NO_KEYRING
    assert store.lookup("wifi password") == ""
    # No plain file anywhere under the machine's config, data or state directories.
    assert list(tmp_path.rglob("*")) == []


def test_secret_tool_command_shape(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(secrets.subprocess, "run", fake_run)
    backend = secrets.SecretToolBackend()
    assert backend.label == "secret-tool"
    assert backend.set("voxa/wifi_password", SECRET)
    assert calls[0] == ["secret-tool", "store", "--label", "Voxa voxa/wifi_password", "voxa/wifi_password", SECRET]
    assert backend.get("voxa/wifi_password") == ""
    assert calls[1] == ["secret-tool", "lookup", "voxa/wifi_password"]
    assert backend.delete("voxa/wifi_password")
    assert calls[2] == ["secret-tool", "delete", "voxa/wifi_password"]


def test_keyring_cli_takes_the_value_on_stdin(monkeypatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(command, *, input=None, **kwargs):
        calls.append((list(command), input or ""))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(secrets.subprocess, "run", fake_run)
    backend = secrets.KeyringCliBackend()
    assert backend.label == "keyring"
    assert backend.set("voxa/wifi_password", SECRET)
    command, stdin = calls[0]
    assert command == ["keyring", "set", "voxa", "voxa/wifi_password"]
    assert SECRET not in command
    assert stdin == SECRET


def test_secret_never_reaches_the_settings_file(tmp_path, backend):
    store = secrets.SecretStore(backend)
    assert store.store("wifi password", SECRET).ok
    config = ConfigStore(tmp_path / "config.json")
    settings = Settings()
    settings.name = "Rhy"
    settings.home_address = "12 oak street"
    settings.phone_number = "555 1212"
    config.save(settings)
    saved = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert SECRET not in saved
    assert "12 oak street" in saved


def test_secret_never_reaches_the_action_log(tmp_path, backend):
    store = secrets.SecretStore(backend)
    stored = store.store("wifi password", SECRET)
    read_back = store.describe("wifi password")
    forgotten = store.forget("wifi password")
    log = ActionLog(tmp_path / "actions.jsonl")
    for result in (stored, read_back, forgotten):
        log.append(
            ActionRecord(
                time="2026-11-06T09:00:00",
                heard="save my wifi password",
                route="tool",
                tool="set_secret",
                args={"name": "wifi password"},
                ok=result.ok,
                speech=result.speech,
                detail=result.detail,
            )
        )
    written = (tmp_path / "actions.jsonl").read_text(encoding="utf-8")
    assert SECRET not in written
    assert len(log.read()) == 3
