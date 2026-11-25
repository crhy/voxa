"""The VoxaTest live-run listening preflight (issue #130).

Offline throughout: a fake command runner plays the role of busctl/flatpak/gapplication,
and a fake listening.json plays Voxa. The point of every test: a live run either starts
listening before the first case, or refuses before playing anything.
"""

from __future__ import annotations

import json
import os
import sys
import types

import voxatest.__main__ as voxatest_main
import voxatest.listening as listening_module
from voxatest.config import HarnessSettings
from voxatest.doctor import FAIL, PASS, SKIP, WARN, check_listening
from voxatest.grader import GradeResult
from voxatest.runner import RunResult
from voxatest.listening import (
    ListeningPreflight,
    action_command,
    choose_state_path,
    read_state,
    state_path_candidates,
)

APP_ID = "io.github.crhy.voxa"


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRunner:
    """busctl / flatpak list / gapplication, with the answers a real machine would give."""

    def __init__(self, *, bus=False, flatpak=False, refuse_action=False, on_action=None):
        self.commands: list[list[str]] = []
        self.bus = bus
        self.flatpak = flatpak
        self.refuse_action = refuse_action
        self.on_action = on_action

    def __call__(self, command):
        self.commands.append(list(command))
        if command[0] == "busctl":
            return Completed(0, stdout=f"{APP_ID}\n" if self.bus else "")
        if command[:2] == ["flatpak", "list"]:
            return Completed(0, stdout=f"{APP_ID}\n" if self.flatpak else "")
        if self.refuse_action:
            return Completed(1, stderr="error: The application is not running")
        if self.on_action is not None:
            self.on_action(command)
        return Completed(0)


def write_state(path, active, reason=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"active": active, "since": 1, "reason": reason}) + "\n", encoding="utf-8")


def _preflight(tmp_path, runner, monkeypatch, *, timeout=0.2):
    state = tmp_path / "listening.json"
    monkeypatch.setenv("VOXATEST_LISTENING_STATE", str(state))
    return ListeningPreflight(timeout=timeout, runner=runner), state


# ------------------------------------------------------------------ command choice


def test_a_running_voxa_is_reached_by_plain_gapplication(monkeypatch):
    monkeypatch.setattr("voxatest.listening.shutil.which", lambda name: f"/usr/bin/{name}")
    command = action_command("start-listening", runner=FakeRunner(bus=True))
    assert command == ["gapplication", "action", APP_ID, "start-listening"]


def test_a_flatpak_voxa_is_started_through_the_flatpak(monkeypatch):
    monkeypatch.setattr("voxatest.listening.shutil.which", lambda name: f"/usr/bin/{name}")
    command = action_command("start-listening", runner=FakeRunner(bus=False, flatpak=True))
    assert command == [
        "flatpak", "run", "--command=gapplication", APP_ID, "action", APP_ID, "start-listening",
    ]


def test_nothing_can_reach_voxa_without_gapplication(monkeypatch):
    monkeypatch.setattr("voxatest.listening.shutil.which", lambda name: None)
    assert action_command("start-listening", runner=FakeRunner()) is None


# ------------------------------------------------------------------ state file


def test_candidates_cover_the_host_and_the_flatpak_homes(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("VOXATEST_LISTENING_STATE", raising=False)
    # Keep the home-relative candidates under tmp_path so a real ~/.var/.../listening.json
    # from a prior live run never outranks the most-specific candidate.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    candidates = state_path_candidates()
    assert tmp_path / "state" / "voxa" / "listening.json" in candidates
    assert any(".var" in str(candidate) and APP_ID in str(candidate) for candidate in candidates)
    assert choose_state_path(candidates) == candidates[0]


def test_the_newest_listening_json_wins(tmp_path):
    old = tmp_path / "old" / "voxa" / "listening.json"
    fresh = tmp_path / "fresh" / "voxa" / "listening.json"
    write_state(old, False)
    write_state(fresh, True)
    os.utime(old, (0, 0))
    assert choose_state_path([old, fresh]) == fresh


# ------------------------------------------------------------------ preflight itself


def test_preflight_accepts_a_voxa_that_is_already_listening(tmp_path, monkeypatch):
    runner = FakeRunner(bus=True)
    preflight, state = _preflight(tmp_path, runner, monkeypatch)
    write_state(state, True)
    assert preflight.ensure_ready() is True
    assert preflight.started_by_us is False
    assert runner.commands == []
    preflight.finish()  # not ours to stop
    assert runner.commands == []


def test_preflight_starts_a_silent_voxa_and_owns_the_stop(tmp_path, monkeypatch):
    state = tmp_path / "listening.json"
    monkeypatch.setenv("VOXATEST_LISTENING_STATE", str(state))
    write_state(state, False, "as shipped: never listening on launch")
    runner = FakeRunner(bus=True, on_action=lambda _c: write_state(state, True))
    preflight = ListeningPreflight(timeout=0.2, runner=runner)
    assert preflight.ensure_ready() is True
    assert preflight.started_by_us is True
    assert runner.commands[-1][-1] == "start-listening"
    assert read_state(state) is not None and read_state(state)["active"] is True
    preflight.finish()
    assert runner.commands[-1][-1] == "stop-listening"
    preflight.finish()  # only once
    assert [c for c in runner.commands if c[-1] == "stop-listening"] == [["gapplication", "action", APP_ID, "stop-listening"]]


def test_preflight_refuses_when_the_action_is_refused(tmp_path, monkeypatch):
    runner = FakeRunner(bus=True, refuse_action=True)
    preflight, _state = _preflight(tmp_path, runner, monkeypatch)
    assert preflight.ensure_ready() is False
    assert "not running" in preflight.reason
    assert preflight.started_by_us is False


def test_preflight_refuses_when_the_state_stays_inactive_and_says_why(tmp_path, monkeypatch):
    runner = FakeRunner(bus=True)
    preflight, state = _preflight(tmp_path, runner, monkeypatch)
    write_state(state, False, "Whisper is still loading.")
    assert preflight.ensure_ready() is False
    assert "Whisper is still loading." in preflight.reason
    # if she woke up late, the run that started her must still take her back down:
    write_state(state, True)
    preflight.finish()
    assert runner.commands[-1][-1] == "stop-listening"


# ------------------------------------------------------------------ the run itself


class FakeMainLoop:
    def __init__(self):
        self.quit_called = False

    def run(self):
        pass

    def quit(self):
        self.quit_called = True


class FakeCapture:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def list_devices(self):
        return []

    def start(self, device_id, on_audio, on_level, on_error):
        self.started += 1

    def stop(self):
        self.stopped += 1


class FakeSpeech:
    def __init__(self):
        self.played = 0

    def speak(self, *args, **kwargs):
        self.played += 1

    def stop(self):
        pass


class FakeGradingClient:
    def list_models(self):
        return ["qwen2.5:0.5b"]


class FakeWhisper:
    def transcribe(self, pcm, language):
        return "Hello there."


def _stub_harness(monkeypatch, capture, speech):
    # Warm the report's lazy imports before the stub lands: after it, the real
    # voxa.speech is unreachable and voxa.bench.render could not be imported any more.
    import voxa.bench.render  # noqa: F401
    import voxatest.hearing  # noqa: F401

    audio_module = types.ModuleType("voxa.audio")
    audio_module.AudioCapture = lambda: capture
    speech_module = types.ModuleType("voxa.speech")
    speech_module.SpeechService = lambda **_kwargs: speech
    monkeypatch.setitem(sys.modules, "voxa.audio", audio_module)
    monkeypatch.setitem(sys.modules, "voxa.speech", speech_module)

    loop = FakeMainLoop()
    repository = types.ModuleType("gi.repository")
    repository.GLib = types.SimpleNamespace(MainLoop=lambda: loop)
    gi_module = types.ModuleType("gi")
    gi_module.repository = repository
    monkeypatch.setitem(sys.modules, "gi", gi_module)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)

    monkeypatch.setattr(voxatest_main, "_load_whisper", lambda _settings, _log: (FakeWhisper(), ""))
    monkeypatch.setattr(HarnessSettings, "make_client", lambda _self: FakeGradingClient())


def test_run_refuses_a_deaf_voxa_before_playing_anything(tmp_path, monkeypatch, capsys):
    capture, speech = FakeCapture(), FakeSpeech()
    _stub_harness(monkeypatch, capture, speech)
    monkeypatch.setattr(voxatest_main, "run_case", lambda *a, **k: (_ for _ in ()).throw(AssertionError("played anyway")))

    runner = FakeRunner(bus=True)
    preflight, state = _preflight(tmp_path, runner, monkeypatch)
    write_state(state, False, "Whisper is still loading.")

    settings = HarnessSettings(voxa_echo_mode="off", settle_seconds=0.0)
    cases = [types.SimpleNamespace(id="one", category="general", prompt="Say hello")]
    code = voxatest_main._run(
        settings, cases, None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"),
        listening=preflight,
    )
    preflight.finish()
    err = capsys.readouterr().err
    assert code == 2
    assert "Voxa is not listening and could not be made to" in err
    assert "Whisper is still loading." in err
    assert speech.played == 0 and capture.started == 0
    assert not list((tmp_path / "reports").glob("run_*.json"))


def test_run_stops_mid_run_when_voxa_goes_deaf_and_the_re_send_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(listening_module, "RECHECK_TIMEOUT_SECONDS", 0.0)
    capture, speech = FakeCapture(), FakeSpeech()
    _stub_harness(monkeypatch, capture, speech)

    seen: list[str] = []

    def fake_run_case(case, *_args, **_kwargs):
        seen.append(case.id)
        if case.id == "two":
            write_state(state, False, "she went OFFLINE")
        return RunResult(case.id, case.category, case.prompt, "Hello there.",
                         GradeResult(90, True, "ok", "model"), duration_seconds=1.0)

    monkeypatch.setattr(voxatest_main, "run_case", fake_run_case)

    runner = FakeRunner(bus=True)
    preflight, state = _preflight(tmp_path, runner, monkeypatch)
    write_state(state, True)

    settings = HarnessSettings(voxa_echo_mode="off", settle_seconds=0.0, retries=0)
    cases = [
        types.SimpleNamespace(id=case_id, category="general", prompt="Say hello")
        for case_id in ("one", "two", "three")
    ]
    code = voxatest_main._run(
        settings, cases, None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"),
        listening=preflight,
    )
    preflight.finish()
    out = capsys.readouterr().out
    assert code == 3
    assert seen == ["one", "two"]  # case two started while she could still hear; case three never started
    assert "she went OFFLINE" in out
    sends = [c for c in runner.commands if c[-1] == "start-listening"]
    assert len(sends) == 1  # re-sent exactly once
    assert not [c for c in runner.commands if c[-1] == "stop-listening"]  # she is deaf already: nothing to stop


def test_run_continues_when_the_re_send_brings_her_back(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(listening_module, "RECHECK_TIMEOUT_SECONDS", 0.0)
    capture, speech = FakeCapture(), FakeSpeech()
    _stub_harness(monkeypatch, capture, speech)

    state = tmp_path / "listening.json"
    monkeypatch.setenv("VOXATEST_LISTENING_STATE", str(state))
    write_state(state, True)
    runner = FakeRunner(
        bus=True,
        on_action=lambda command: command[-1] == "start-listening" and write_state(state, True),
    )
    preflight = ListeningPreflight(timeout=0.2, runner=runner)

    def fake_run_case(case, *_args, **_kwargs):
        if case.id == "two":
            write_state(state, False, "brief dropout")
        return RunResult(case.id, case.category, case.prompt, "Hello there.",
                         GradeResult(90, True, "ok", "model"), duration_seconds=1.0)

    monkeypatch.setattr(voxatest_main, "run_case", fake_run_case)

    settings = HarnessSettings(voxa_echo_mode="off", settle_seconds=0.0, retries=0)
    cases = [
        types.SimpleNamespace(id=case_id, category="general", prompt="Say hello")
        for case_id in ("one", "two", "three")
    ]
    code = voxatest_main._run(
        settings, cases, None, None, str(tmp_path / "reports"), str(tmp_path / "failures.md"),
        listening=preflight,
    )
    assert code == 0
    assert len([c for c in runner.commands if c[-1] == "start-listening"]) == 1  # once, recovered, carried on
    assert len([c for c in runner.commands if c[-1] == "stop-listening"]) == 0  # she was listening before us


# ------------------------------------------------------------------ doctor


def test_the_doctor_does_not_probe_listening_in_a_dry_run():
    assert check_listening(True).status == SKIP


def test_the_doctor_passes_a_reachable_voxa_and_warns_without_her(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXATEST_LISTENING_STATE", str(tmp_path / "listening.json"))
    monkeypatch.setattr("voxatest.listening.shutil.which", lambda name: f"/usr/bin/{name}")
    write_state(tmp_path / "listening.json", True)
    assert check_listening(False, runner=FakeRunner(bus=True)).status == PASS
    write_state(tmp_path / "listening.json", False, "as shipped: never listening on launch")
    assert check_listening(False, runner=FakeRunner(bus=True)).status == PASS
    (tmp_path / "listening.json").unlink()
    assert check_listening(False, runner=FakeRunner(bus=True)).status == WARN

    monkeypatch.setattr("voxatest.listening.shutil.which", lambda name: None)
    assert check_listening(False, runner=FakeRunner()).status == FAIL
