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


def test_grab_balanced(monkeypatch):
    sent = []
    monkeypatch.setattr(compiz, "_run", lambda command: sent.append(command) or True)
    monkeypatch.setattr(compiz, "SPIN_MAX_SECONDS", 0.12)
    compiz._spin_stop.clear()
    compiz._spin_grab((960, 540), 7680, 1920, 1080, 1)
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
        compiz._spin_grab((960, 540), 7680, 1920, 1080, 1)
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
    compiz._spin_grab((10, 10), 7680, 1920, 1080, 1)
    assert [c[1] for c in sent] == ["keydown", "mousedown", "keyup", "mousemove"]


def test_grab_regrabs_near_edge(monkeypatch):
    sent = []
    monkeypatch.setattr(compiz, "_run", lambda command: sent.append(command) or True)
    monkeypatch.setattr(compiz, "SPIN_MAX_SECONDS", 0.04)
    compiz._spin_stop.clear()
    compiz._spin_grab((100, 540), 7680, 1920, 1080, -1)
    assert [c[1] for c in sent] == [
        "keydown",
        "mousedown",
        "mouseup",
        "keyup",
        "mousemove",
        "keydown",
        "mousedown",
        "mousemove_relative",
        "mouseup",
        "keyup",
        "mousemove",
    ]
    assert sent[4] == ["xdotool", "mousemove", "960", "540"]
    assert sent[-1] == ["xdotool", "mousemove", "100", "540"]
