"""The listening state file scripted live runs read (issue #130)."""

from __future__ import annotations

import json

from voxa.listening import default_path, read_state, write_state


def test_default_path_follows_the_xdg_state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("VOXA_LISTENING_STATE", raising=False)
    assert default_path() == tmp_path / "state" / "voxa" / "listening.json"


def test_default_path_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXA_LISTENING_STATE", str(tmp_path / "listen.json"))
    assert default_path() == tmp_path / "listen.json"


def test_write_then_read_records_the_truth(tmp_path):
    path = tmp_path / "voxa" / "listening.json"
    write_state(path, False, "Whisper is still loading.")
    state = read_state(path)
    assert state is not None
    assert state["active"] is False
    assert state["reason"] == "Whisper is still loading."
    assert isinstance(state["since"], int) and state["since"] > 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"active", "since", "reason"}


def test_write_creates_the_folder_and_active_carries_no_reason(tmp_path):
    path = tmp_path / "deep" / "nested" / "listening.json"
    write_state(path, True)
    state = read_state(path)
    assert state is not None and state["active"] is True and state["reason"] == ""


def test_read_state_is_honest_about_garbage(tmp_path):
    path = tmp_path / "listening.json"
    assert read_state(path) is None
    path.write_text("{not json", encoding="utf-8")
    assert read_state(path) is None
    path.write_text('{"active": "yes"}', encoding="utf-8")
    assert read_state(path) is None
