from voxa.agent import intents
from voxa.agent.tools import compiz


def test_faces_and_way_back():
    assert compiz.faces_from_wmctrl("0  * DG: 7680x1080  VP: 0,0  WA: 0,25 1920x1055  Workspace 1") == 4
    assert compiz.faces_from_wmctrl("0 * DG: 1920x1080 VP: N/A WA: 0,0 1920x1080 a\n1 - DG: 1920x1080 VP: N/A WA: 0,0 1920x1080 b\n2 - x") == 3
    assert compiz.faces_from_wmctrl("") == 4
    assert compiz.way_back(0, 4) == ("cube_left", 0)
    assert compiz.way_back(1, 4) == ("cube_left", 1)
    assert compiz.way_back(2, 4) == ("cube_left", 2)
    assert compiz.way_back(3, 4) == ("cube_right", 1)
    assert compiz.way_back(9, 4) == ("cube_left", 1)
    assert compiz.way_back(8, 4)[1] == 0


def test_spin_turns_then_comes_home(monkeypatch):
    pressed = []
    monkeypatch.setattr(compiz, "_press", lambda combo, times=1: pressed.append((combo, times)) or True)
    monkeypatch.setattr(compiz, "SPIN_SECONDS", 0.01)
    monkeypatch.setattr(compiz, "SPIN_MAX_FACES", 3)
    compiz._spin_stop.clear()
    compiz._spin({"cube_right": "R", "cube_left": "L"}, 4)
    assert pressed == [("R", 1), ("R", 1), ("R", 1), ("R", 1)]  # three turns, then one more right is the short way home
    assert compiz.stop_spin() is False


def test_router():
    assert intents.route("spin cube").args == {"action": "spin"}
    assert intents.route("Spin the cube.").args == {"action": "spin"}
    assert intents.route("spin cube right").args == {"action": "cube_right"}
    assert intents.route("spin cube left").args == {"action": "cube_left"}
    assert intents.route("rotate cube right").args == {"action": "cube_right"}
    assert intents.route("stop spinning").args == {"action": "spin_stop"}
