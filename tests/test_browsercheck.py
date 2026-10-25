from __future__ import annotations

import pytest

from voxa.agent import browsercheck
from voxa.agent.tools import browser, default_registry


def make_run(mime=None, settings=None, categories=""):
    """Fake run(cmd) -> (returncode, stdout) for the browser-detection commands."""
    calls: list[list[str]] = []

    def run(command):
        calls.append(list(command))
        if command[:2] == ["xdg-mime", "query"]:
            if mime is None:
                return (1, "")
            return (0, mime + "\n")
        if command[:2] == ["xdg-settings", "get"]:
            if settings is None:
                return (1, "")
            return (0, settings + "\n")
        if command[:1] == ["grep"]:
            return (0, categories + "\n")
        raise AssertionError(f"unexpected command {command}")

    run.calls = calls
    return run


def test_browser_present_via_xdg_mime():
    browsercheck.clear_cache()
    run = make_run(mime="firefox.desktop")
    assert browsercheck.has_web_browser(run) is True


def test_browser_present_via_xdg_settings():
    browsercheck.clear_cache()
    run = make_run(settings="org.mozilla.firefox.desktop")
    assert browsercheck.has_web_browser(run) is True


def test_no_browser_when_no_handler():
    browsercheck.clear_cache()
    run = make_run()
    assert browsercheck.has_web_browser(run) is False


def test_text_editor_handler_is_not_a_browser():
    browsercheck.clear_cache()
    run = make_run(
        mime="org.gnome.Pluma.desktop",
        categories="Categories=GNOME;GTK;TextEditor;Application;",
    )
    assert browsercheck.has_web_browser(run) is False


def test_browser_detected_from_categories_even_if_name_is_plain():
    browsercheck.clear_cache()
    run = make_run(
        mime="mystuff.desktop",
        categories="Categories=Application;WebBrowser;",
    )
    assert browsercheck.has_web_browser(run) is True


def test_answer_is_cached_for_sixty_seconds():
    browsercheck.clear_cache()
    run = make_run(mime="firefox.desktop")
    assert browsercheck.has_web_browser(run) is True
    first = len(run.calls)
    assert browsercheck.has_web_browser(run) is True
    assert len(run.calls) == first  # second call served from cache


def test_clear_cache_forces_requery():
    browsercheck.clear_cache()
    run = make_run(mime="firefox.desktop")
    assert browsercheck.has_web_browser(run) is True
    browsercheck.clear_cache()
    run2 = make_run()
    assert browsercheck.has_web_browser(run2) is False


def test_open_url_reports_no_browser(monkeypatch):
    monkeypatch.setattr(browser, "_host_run", make_run(mime="org.gnome.Pluma.desktop"))
    browsercheck.clear_cache()
    spawned: list[list[str]] = []
    monkeypatch.setattr(browser.host, "spawn", lambda command: spawned.append(command) or True)
    result = default_registry().call("open_url", {"url": "example.com"})
    assert not result.ok
    assert result.speech == browser.NO_BROWSER_MESSAGE
    assert spawned == []


def test_open_url_opens_when_browser_present(monkeypatch):
    monkeypatch.setattr(browser, "_host_run", make_run(mime="firefox.desktop"))
    browsercheck.clear_cache()
    spawned: list[list[str]] = []
    monkeypatch.setattr(browser.host, "spawn", lambda command: spawned.append(command) or True)
    result = default_registry().call("open_url", {"url": "example.com"})
    assert result.ok
    assert spawned == [["xdg-open", "https://example.com"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
