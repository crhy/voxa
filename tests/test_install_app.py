"""Tests for the Flathub/Spaced Bazaar install and uninstall flow."""
from __future__ import annotations

import json

from voxa.agent.appstore import choose, clean_query, parse_hits, spoken_choice, valid_app_id
from voxa.agent.intents import route


def test_clean_query():
    assert clean_query("the most popular snes emulator") == ("snes emulator", True)
    assert clean_query("vlc from flathub") == ("vlc", False)
    assert clean_query("gimp") == ("gimp", False)
    assert clean_query("the best app") == ("", True)
    assert clean_query("top snes emulator please") == ("snes emulator", True)


def test_parse_hits():
    text = json.dumps(
        {
            "hits": [
                {"app_id": "a", "name": "A", "type": "desktop-application"},
                {"app_id": "b", "name": "B", "type": "addon"},
                {"app_id": "c", "name": "C"},
            ]
        }
    )
    assert parse_hits(text) == [{"app_id": "a", "name": "A", "type": "desktop-application"}, {"app_id": "c", "name": "C"}]
    assert parse_hits("not json") == []


def test_choose():
    hits = [
        {"app_id": "com.snes9x.Snes9x", "name": "Snes9x", "installs_last_month": 5758},
        {"app_id": "io.Bsnes.Bsnes", "name": "bsnes", "installs_last_month": 1628},
        {"app_id": "org.zsnes.Zsnes", "name": "ZSNES", "installs_last_month": 2432},
        {"app_id": "ca.punes.Punes", "name": "puNES", "installs_last_month": 619},
        {"app_id": "org.m64py.M64py", "name": "M64Py", "installs_last_month": 936},
    ]
    assert choose("snes emulator", hits, True)["app_id"] == "com.snes9x.Snes9x"
    popular = [
        {"app_id": "a", "name": "first", "installs_last_month": 100},
        {"app_id": "b", "name": "second", "installs_last_month": 9000},
    ]
    assert choose("anything", popular, False)["app_id"] == "b"
    exact = [
        {"app_id": "org.gimp.GIMP", "name": "GNU Image Manipulation Program", "installs_last_month": 1},
        {"app_id": "com.other.App", "name": "gimp", "installs_last_month": 9999},
    ]
    assert choose("gimp", exact, False)["app_id"] == "org.gimp.GIMP"
    assert choose("nothing", [], False) is None


def test_valid_app_id():
    assert valid_app_id("com.snes9x.Snes9x")
    assert not valid_app_id("bad id; rm -rf")


def test_spoken_choice():
    assert spoken_choice({"name": "Snes9x", "summary": "A Super Nintendo emulator"}) == "Snes9x, A Super Nintendo emulator"
    assert spoken_choice({"name": "Snes9x"}) == "Snes9x"


class FakeUI:
    def __init__(self):
        self.press_log = []
        self.wait_ok = True
        self.items_script = []

    def press(self, app, label, role=""):
        self.press_log.append((app, label, role) if role else (app, label))
        return {"ok": self.press_ok}

    def wait(self, app, label, seconds):
        return {"ok": self.wait_ok}

    def items(self, app, *roles):
        return self.items_script

    def announce_password_once(self, state):
        return False

    press_ok = True


def fake_run(info_zero_from):
    calls = {"n": 0}

    def run(command, timeout, check=False):
        import subprocess

        if command[:2] == ["flatpak", "info"]:
            calls["n"] += 1
            code = 0 if calls["n"] >= info_zero_from else 1
            return subprocess.CompletedProcess(command, code, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1000.0
        return self.t


def test_install_happy_path():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    spawn_log = []
    hits = json.dumps({"hits": [{"app_id": "com.snes9x.Snes9x", "name": "Snes9x", "summary": "A Super Nintendo emulator", "installs_last_month": 5758, "type": "desktop-application"}]})
    appstore.ui = ui
    appstore._run = fake_run(3)
    appstore.spawn = lambda command: spawn_log.append(command)
    appstore.search = lambda query: hits
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()
    ui.items_script = [{"role": "button", "name": "Install", "enabled": True, "showing": True}]
    result = appstore.install_app({"name": "the most popular snes emulator"})
    assert any(c == ["flatpak", "run", "io.github.crhy.SpacedBazaar", "appstream://com.snes9x.Snes9x"] for c in spawn_log)
    assert ("bazaar", "Install Snes9x") in ui.press_log
    assert ("bazaar", "Install") in ui.press_log
    assert ui.press_log.count(("bazaar", "Install")) == 1
    assert result.ok
    assert result.speech.startswith("Installed Snes9x")


def test_install_dialog_pressed_once():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    appstore.ui = ui
    appstore._run = fake_run(3)
    appstore.spawn = lambda command: None
    appstore.search = lambda query: json.dumps(
        {"hits": [{"app_id": "com.snes9x.Snes9x", "name": "Snes9x", "type": "desktop-application"}]}
    )
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()
    ui.items_script = [{"role": "button", "name": "Install", "enabled": True, "showing": True}]
    appstore.install_app({"name": "snes9x"})
    assert ui.press_log.count(("bazaar", "Install")) == 1


def test_install_never_installed():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    appstore.ui = ui
    appstore._run = fake_run(999)
    appstore.spawn = lambda command: None
    appstore.search = lambda query: json.dumps(
        {"hits": [{"app_id": "com.snes9x.Snes9x", "name": "Snes9x", "type": "desktop-application"}]}
    )
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()
    ui.items_script = []
    result = appstore.install_app({"name": "snes9x"})
    assert not result.ok
    assert "not installed yet" in result.speech


def test_install_press_failing():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    ui.press_ok = False
    appstore.ui = ui
    appstore._run = fake_run(999)
    appstore.spawn = lambda command: None
    appstore.search = lambda query: json.dumps(
        {"hits": [{"app_id": "com.snes9x.Snes9x", "name": "Snes9x", "type": "desktop-application"}]}
    )
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()
    result = appstore.install_app({"name": "snes9x"})
    assert not result.ok
    assert "could not press Install" in result.speech


def test_install_page_never_showing():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    ui.wait_ok = False
    appstore.ui = ui
    appstore._run = fake_run(999)
    appstore.spawn = lambda command: None
    appstore.search = lambda query: json.dumps(
        {"hits": [{"app_id": "com.snes9x.Snes9x", "name": "Snes9x", "type": "desktop-application"}]}
    )
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()
    result = appstore.install_app({"name": "snes9x"})
    assert not result.ok
    assert "did not show" in result.speech


def test_install_already_installed():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    appstore.ui = ui
    appstore._run = fake_run(1)
    appstore.spawn = lambda command: None
    appstore.search = lambda query: json.dumps(
        {"hits": [{"app_id": "com.snes9x.Snes9x", "name": "Snes9x", "type": "desktop-application"}]}
    )
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()
    result = appstore.install_app({"name": "snes9x"})
    assert result.ok
    assert "already installed" in result.speech
    assert ui.press_log == []


def test_install_no_hits_search():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    spawn_log = []
    appstore.ui = ui
    appstore._run = fake_run(999)
    appstore.spawn = lambda command: spawn_log.append(command)
    appstore.search = lambda query: json.dumps({"hits": []})
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()
    result = appstore.install_app({"name": "nonexistent app"})
    assert not result.ok
    assert any(c[-2:] == ["--search-for", "nonexistent"] for c in spawn_log)
    assert "open on the search" in result.speech


def test_uninstall_happy_path():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    appstore.ui = ui
    appstore.spawn = lambda command: None
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()

    def run(command, timeout, check=False):
        import subprocess

        if command[:3] == ["flatpak", "list", "--app"]:
            return subprocess.CompletedProcess(command, 0, "com.snes9x.Snes9x\tSnes9x\n", "")
        if command[:2] == ["flatpak", "info"]:
            return subprocess.CompletedProcess(command, 1, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    appstore._run = run
    result = appstore.uninstall_app({"name": "snes9x"})
    assert result.ok
    assert "Removed Snes9x" in result.speech
    assert ("bazaar", "Uninstall Application") in ui.press_log
    assert ("bazaar", "Remove", "button") in ui.press_log
    assert all("Delete All Data" not in entry for entry in ui.press_log)


def test_uninstall_never_gone():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    appstore.ui = ui
    appstore.spawn = lambda command: None
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()

    def run(command, timeout, check=False):
        import subprocess

        if command[:3] == ["flatpak", "list", "--app"]:
            return subprocess.CompletedProcess(command, 0, "com.snes9x.Snes9x\tSnes9x\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    appstore._run = run
    result = appstore.uninstall_app({"name": "snes9x"})
    assert not result.ok
    assert "could not remove" in result.speech


def test_uninstall_not_installed():
    import voxa.agent.tools.appstore as appstore

    ui = FakeUI()
    appstore.ui = ui
    appstore.spawn = lambda command: None
    appstore._sleep = lambda seconds: None
    appstore._clock = Clock()
    appstore._run = lambda command, timeout, check=False: __import__("subprocess").CompletedProcess(command, 0, "", "")
    result = appstore.uninstall_app({"name": "ghost app"})
    assert not result.ok
    assert "could not find an installed app" in result.speech


def test_router():
    assert route("install the most popular snes emulator").tool == "install_app"
    assert route("install gimp").tool == "install_app"
    assert route("open spaced bazaar and install the most popular snes emulator").tool == "install_app"
    assert route("install vlc from flathub").args["name"] == "vlc"
    assert route("install ollama") is None or route("install ollama").tool != "install_app"
    assert route("install all the latest updates").tool != "install_app"
    assert route("install it") is None or route("install it").tool != "install_app"
    assert route("uninstall snes9x").tool == "uninstall_app"
