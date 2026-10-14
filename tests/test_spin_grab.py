import pytest

from voxa.agent.tools import compiz


def test_spin_dx():
    assert compiz.spin_dx(7680) == 15
    assert compiz.spin_dx(1920) == 4
    assert compiz.spin_dx(7680, turn_seconds=10, step_ms=40) == 31
    assert compiz.spin_dx(0) == 1


def test_desktop_size():
    assert compiz.desktop_size("0  * DG: 7680x1080  VP: 0,0  WA: 0,25 1920x1055  Workspace 1") == (7680, 1920, 1055)
    assert compiz.desktop_size("") == (0, 0, 0)


def test_mouse_position_output():
    assert compiz.mouse_position_from_output("X=1551\nY=785\nSCREEN=0\nWINDOW=48234500\n") == (1551, 785)
    assert compiz.mouse_position_from_output("x:1551 y:785 screen:0 window:48234500") == (1551, 785)
    assert compiz.mouse_position_from_output("xdotool failed") is None


def test_grab_balanced(monkeypatch):
    sent = []
    monkeypatch.setattr(compiz, "_run", lambda command: sent.append(command) or True)
    monkeypatch.setattr(compiz, "SPIN_MAX_SECONDS", 0.12)
    compiz._spin_stop.clear()
    compiz._spin_grab((960, 540), 7680, 1)
    downs = [c[1] for c in sent if c[1] in ("keydown", "mousedown")]
    ups = [c[1] for c in sent if c[1] in ("keyup", "mouseup")]
    assert downs == ["keydown", "mousedown"] and ups == ["mouseup", "keyup"]
    assert sent[:2] == [["xdotool", "keydown", "ctrl+alt"], ["xdotool", "mousedown", "1"]]
    assert sent[-3:] == [
        ["xdotool", "mouseup", "1"],
        ["xdotool", "keyup", "ctrl+alt"],
        ["xdotool", "mousemove", "960", "540"],
    ]


def test_grab_balanced_when_a_step_raises(monkeypatch):
    sent = []

    def flaky(command):
        sent.append(command)
        if len(sent) == 4:
            raise RuntimeError("step failed")
        return True

    monkeypatch.setattr(compiz, "_run", flaky)
    monkeypatch.setattr(compiz, "SPIN_MAX_SECONDS", 0.12)
    compiz._spin_stop.clear()
    with pytest.raises(RuntimeError):
        compiz._spin_grab((960, 540), 7680, 1)
    downs = [c[1] for c in sent if c[1] in ("keydown", "mousedown")]
    ups = [c[1] for c in sent if c[1] in ("keyup", "mouseup")]
    assert downs == ["keydown", "mousedown"] and ups == ["mouseup", "keyup"]
    assert sent[-3:] == [
        ["xdotool", "mouseup", "1"],
        ["xdotool", "keyup", "ctrl+alt"],
        ["xdotool", "mousemove", "960", "540"],
    ]


def test_grab_failure_leaves_nothing_pressed(monkeypatch):
    sent = []

    def no_button(command):
        sent.append(command)
        return command[1] != "mousedown"

    monkeypatch.setattr(compiz, "_run", no_button)
    compiz._spin_stop.clear()
    compiz._spin_grab((10, 10), 7680, 1)
    assert [c[1] for c in sent] == ["keydown", "mousedown", "keyup", "mousemove"]


def test_grab_stays_held_near_edge(monkeypatch):
    sent = []
    monkeypatch.setattr(compiz, "_run", lambda command: sent.append(command) or True)
    monkeypatch.setattr(compiz, "SPIN_MAX_SECONDS", 0.04)
    compiz._spin_stop.clear()
    compiz._spin_grab((10, 540), 7680, -1)
    assert [c[1] for c in sent] == [
        "keydown",
        "mousedown",
        "mousemove_relative",
        "mouseup",
        "keyup",
        "mousemove",
    ]
    assert sent[-1] == ["xdotool", "mousemove", "10", "540"]


def test_spin_defaults_to_a_left_drag(monkeypatch):
    started = []
    thread = type("Thread", (), {"is_alive": lambda self: False})()
    monkeypatch.setattr(compiz, "_spin_thread", thread)
    monkeypatch.setattr(compiz, "bindings", lambda: {})
    monkeypatch.setattr(compiz, "desktop_size", lambda listing: (7680, 1920, 1080))
    monkeypatch.setattr(compiz, "_mouse_position", lambda: (960, 540))
    monkeypatch.setattr(compiz.subprocess, "run", lambda *args, **kwargs: type("Done", (), {"stdout": ""})())

    class FakeThread:
        def __init__(self, *, target, args, name, daemon):
            started.append((target, args, name, daemon))

        def start(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(compiz.threading, "Thread", FakeThread)
    assert compiz.start_spin().ok
    assert started[0][1][-1] == -1
