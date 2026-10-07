import subprocess

from voxa.agent.intents import route
from voxa.agent.tools import windows as windows_mod


def fake_run(cmd, check=True):
    if "wmctrl" in cmd:
        stdout = (
            "0x01 0 pluma.Pluma u1 notes.txt - Pluma\n"
            "0x02 0 brave.Browser u1 Amazon.com: tents - Brave\n"
            "0x03 0 caja.Caja u1 Caja\n"
            "0x00 0 mate-panel.Desktop_window u1 mate-panel\n"
            "0x04 0 foo.Desktop u1 x\n"
            "0x05 0 bar.Panel u1 y\n"
            "0x06 0 voxa.Voxa u1 Voxa\n"
        )
    elif "xdotool" in cmd:
        stdout = "notes.txt - Pluma"
    else:
        stdout = ""
    return subprocess.CompletedProcess(cmd, 0, stdout, "")


def test_window_label_table():
    assert windows_mod.window_label("notes.txt - Pluma", "pluma.Pluma") == "Pluma"
    assert windows_mod.window_label("Amazon.com: tents - Brave", "brave.Browser") == "Brave"
    assert windows_mod.window_label("doc \u2014 Pluma", "pluma.Pluma") == "Pluma"
    assert windows_mod.window_label("Caja", "caja.Caja") == "Caja"
    assert windows_mod.window_label("", "gnome-terminal.Gnome-terminal") == "Gnome-terminal"
    assert windows_mod.window_label("", "firefox.Firefox Web Browser") == "Firefox"
    assert windows_mod.window_label("", "thunar.Thunar File Manager") == "Thunar"
    assert windows_mod.window_label("", "x." + "a" * 40) == "A" + "a" * 29


def test_describe_windows():
    assert windows_mod.describe_windows([]) == "Nothing is open."
    assert windows_mod.describe_windows(["Brave"]) == "You have one window open: Brave."
    assert windows_mod.describe_windows(["Brave", "Brave"]) == "You have 2 windows open: two Brave windows."
    assert (
        windows_mod.describe_windows(["Brave", "Pluma", "Pluma", "Caja"])
        == "You have 4 windows open: Brave, two Pluma windows and Caja."
    )
    nine = [chr(ord("A") + i) for i in range(9)]
    assert windows_mod.describe_windows(nine) == "You have 9 windows open: A, B, C, D, E, F and 3 more."


def test_list_open_windows_skips():
    windows_mod._run = fake_run
    result = windows_mod.list_open_windows({})
    assert result.speech == "You have 3 windows open: Pluma, Brave and Caja."


def test_active_window_name_title():
    windows_mod._run = fake_run
    result = windows_mod.active_window_name({})
    assert result.speech == "This is notes.txt - Pluma."


def test_active_window_name_long_title():
    def long_run(cmd, check=True):
        return subprocess.CompletedProcess(cmd, 0, "x" * 200, "")

    windows_mod._run = long_run
    result = windows_mod.active_window_name({})
    assert result.speech == "This is " + "x" * 80 + "."


def test_active_window_name_empty():
    def empty_run(cmd, check=True):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    windows_mod._run = empty_run
    result = windows_mod.active_window_name({})
    assert result.speech == "I cannot tell which window is in front."


def test_router_phrases():
    assert route("what's open").tool == "list_open_windows"
    assert route("which programs are open").tool == "list_open_windows"
    assert route("list the open windows").tool == "list_open_windows"
    assert route("what window is this").tool == "active_window_name"
    assert route("which window is in front").tool == "active_window_name"
    assert route("what am i looking at").tool == "active_window_name"


def test_what_open_source_not_routed():
    call = route("what's open source")
    assert call is None or call.tool != "list_open_windows"
