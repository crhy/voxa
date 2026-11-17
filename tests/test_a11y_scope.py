"""The a11y bus is on only during an app-control action, and always back as it was.

Everything runs against a fake bus: no test here reads or writes the real desktop's accessibility setting.
"""

from __future__ import annotations

import subprocess

import pytest

from voxa.agent import ui
from voxa.agent.a11y_scope import TAKING_CONTROL, AccessibilityScope, GdbusBus


class FakeBus:
    """Records every call, and can fail the way a missing or refusing bus does."""

    def __init__(self, enabled=False, read_fails=False, set_fails=False):
        self.enabled = enabled
        self.read_fails = read_fails
        self.set_fails = set_fails
        self.calls: list[tuple[str, bool]] = []

    def is_enabled(self):
        self.calls.append(("get", False))
        if self.read_fails:
            raise RuntimeError("no a11y bus on this computer")
        return self.enabled

    def set_enabled(self, value):
        self.calls.append(("set", value))
        if self.set_fails:
            raise RuntimeError("the bus refused")
        self.enabled = value
        return True


def _boom():
    raise RuntimeError("the program vanished mid-action")


def test_scope_turns_accessibility_on_only_inside_the_block():
    bus = FakeBus(enabled=False)
    with AccessibilityScope(bus) as scope:
        inside = bus.enabled
        assert scope.took_control is True
    assert inside is True
    assert bus.enabled is False


def test_previous_state_restored_on_success():
    bus = FakeBus(enabled=False)
    with AccessibilityScope(bus):
        pass
    assert bus.calls == [("get", False), ("set", True), ("set", False)]
    assert bus.enabled is False


def test_previous_state_restored_when_the_action_raises():
    bus = FakeBus(enabled=False)
    with pytest.raises(RuntimeError):
        with AccessibilityScope(bus):
            _boom()
    assert bus.enabled is False


def test_never_left_on_whatever_the_action_does():
    for action in (lambda: None, _boom):
        for start in (False, True):
            bus = FakeBus(enabled=start)
            try:
                with AccessibilityScope(bus):
                    action()
            except RuntimeError:
                pass
            assert bus.enabled is start


def test_accessibility_the_user_left_on_is_left_alone():
    bus = FakeBus(enabled=True)
    with AccessibilityScope(bus):
        assert bus.enabled is True
    assert bus.calls == [("get", False)]
    assert bus.enabled is True


def test_without_consent_nothing_is_touched():
    bus = FakeBus(enabled=False)
    with AccessibilityScope(bus, consent=False):
        assert bus.enabled is False
    assert bus.calls == []


def test_notice_is_spoken_when_voxa_takes_control(caplog):
    spoken = []
    bus = FakeBus(enabled=False)
    with caplog.at_level("INFO", logger="voxa.agent.a11y_scope"):
        with AccessibilityScope(bus, notice=spoken.append, subject="Spaced Update"):
            pass
    assert spoken == [f"{TAKING_CONTROL} Spaced Update"]
    assert TAKING_CONTROL in caplog.text


def test_no_notice_when_the_user_already_left_it_on():
    spoken = []
    with AccessibilityScope(FakeBus(enabled=True), notice=spoken.append):
        pass
    assert spoken == []


def test_stop_restores_it_straight_away():
    bus = FakeBus(enabled=False)
    scope = AccessibilityScope(bus)
    with scope:
        assert bus.enabled is True
        scope.stop()
        assert bus.enabled is False
    assert bus.calls == [("get", False), ("set", True), ("set", False)]


def test_a_missing_bus_leaves_the_action_running_and_nothing_changed():
    bus = FakeBus(read_fails=True)
    with AccessibilityScope(bus) as scope:
        assert scope.took_control is False
        assert bus.enabled is False
    assert bus.enabled is False


def test_a_refused_bus_is_not_reported_as_taken_control():
    bus = FakeBus(enabled=False, set_fails=True)
    spoken = []
    with AccessibilityScope(bus, notice=spoken.append) as scope:
        assert scope.took_control is False
    assert spoken == []
    assert bus.enabled is False


class FakeSubprocess:
    """Stands in for the subprocess module inside a11y_scope, so no gdbus is ever run."""

    SubprocessError = subprocess.SubprocessError

    def __init__(self, output="(<false>,)\n"):
        self.output = output
        self.argv: list[list[str]] = []

    def run(self, argv, **kw):
        self.argv.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, self.output, "")


def test_gdbus_bus_reads_the_value_gdbus_prints(monkeypatch):
    fake = FakeSubprocess("(<true>,)\n")
    monkeypatch.setattr("voxa.agent.a11y_scope.subprocess", fake)
    bus = GdbusBus()
    assert bus.is_enabled() is True
    assert fake.argv[0][:2] == ["gdbus", "call"]
    assert "org.a11y.Status" in fake.argv[0] and "IsEnabled" in fake.argv[0]

    fake.output = "(<false>,)\n"
    assert bus.is_enabled() is False

    assert bus.set_enabled(True) is True
    assert fake.argv[-1][-3:] == ["org.a11y.Status", "IsEnabled", "<true>"]


def test_gdbus_bus_says_no_when_the_bus_refuses(monkeypatch):
    fake = FakeSubprocess("Error executing rpc: object org.a11y.Bus not found\n")
    monkeypatch.setattr("voxa.agent.a11y_scope.subprocess", fake)
    bus = GdbusBus()
    assert bus.is_enabled() is False
    assert bus.set_enabled(True) is False


def test_gdbus_bus_runs_nothing_while_actions_are_simulated(monkeypatch):
    fake = FakeSubprocess("(<false>,)\n")
    monkeypatch.setattr("voxa.agent.a11y_scope.subprocess", fake)
    monkeypatch.setattr("voxa.simulation.actions_simulated", lambda: True)
    bus = GdbusBus()
    assert bus.is_enabled() is False
    assert bus.set_enabled(True) is True
    assert fake.argv == []


def test_the_helper_runs_with_accessibility_on_and_is_restored_after(monkeypatch):
    bus = FakeBus(enabled=False)
    monkeypatch.setattr(ui, "a11y_bus", bus)
    states: list[bool] = []

    def runner(command, **kw):
        states.append(bus.enabled)
        return subprocess.CompletedProcess(command, 0, '{"ok": true, "pressed": {"name": "Save"}}', "")

    result = ui.press("GIMP", "Save", runner=runner)
    assert result["ok"]
    assert states == [True]
    assert bus.enabled is False
    assert bus.calls == [("get", False), ("set", True), ("set", False)]


def test_a_failed_action_still_leaves_accessibility_off(monkeypatch):
    bus = FakeBus(enabled=False)
    monkeypatch.setattr(ui, "a11y_bus", bus)
    monkeypatch.setattr(ui, "call", lambda command, *args, **kw: {"ok": False, "error": "program not found"})
    result = ui.press("Spaced Update", "Update System")
    assert not result["ok"]
    assert bus.enabled is False
    assert bus.calls == [("get", False), ("set", True), ("set", False)]
