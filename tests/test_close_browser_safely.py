"""“Close Brave” must never close the user's own browser windows or kill a program with unsaved work."""

import types

from voxa.agent.tools import windows as w


def _setup(monkeypatch, matching, own_browser_open=False):
    ran = []
    monkeypatch.setattr(w, "_run", lambda cmd, check=True: ran.append(cmd) or types.SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr(w, "list_windows", lambda: matching)
    monkeypatch.setattr(w, "match_windows", lambda name, wins: wins)
    monkeypatch.setattr(w.apps, "list_apps", lambda: [])
    monkeypatch.setattr(w.apps, "match_app", lambda name, items: types.SimpleNamespace(path="x.desktop", name="App"))
    monkeypatch.setattr(w, "flatpak_app_id", lambda path: "org.example.App")
    closed = []
    session = types.SimpleNamespace(_port=1, close_browser=lambda: closed.append(True))
    import voxa.agent.browser_session as bs
    import voxa.agent.tools.web as web

    monkeypatch.setattr(bs, "browser_socket", lambda port: object() if own_browser_open else None)
    monkeypatch.setattr(web, "get_session", lambda: session)
    return ran, closed


def test_close_brave_closes_only_her_own_browser(monkeypatch):
    ran, closed = _setup(monkeypatch, [types.SimpleNamespace(id="0x1"), types.SimpleNamespace(id="0x2")], own_browser_open=True)
    result = w.close_app({"name": "Brave"})
    assert closed == [True] and ran == []
    assert "untouched" in result.speech


def test_close_brave_leaves_the_users_windows_alone(monkeypatch):
    ran, closed = _setup(monkeypatch, [types.SimpleNamespace(id="0x1"), types.SimpleNamespace(id="0x2")])
    result = w.close_app({"name": "brave"})
    assert ran == [] and closed == []
    assert "2 windows of yours" in result.speech


def test_other_programs_are_asked_to_close_and_never_killed(monkeypatch):
    ran, _closed = _setup(monkeypatch, [types.SimpleNamespace(id="0x9")])
    w.close_app({"name": "LibreOffice"})
    assert ran == [["wmctrl", "-i", "-c", "0x9"]]


def test_a_program_with_no_window_is_still_stopped(monkeypatch):
    ran, _closed = _setup(monkeypatch, [])
    w.close_app({"name": "Some Daemon"})
    assert ran == [["flatpak", "kill", "org.example.App"]]
