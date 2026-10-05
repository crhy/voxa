from __future__ import annotations

from voxa.apps import DesktopApp
from voxa.vocabulary import build_hint, refresh_app_names


def test_build_hint_lists_wake_word_verbs_and_names() -> None:
    hint = build_hint(["GIMP", "Firefox"], "voxa")
    assert hint.startswith(
        "Voxa. Commands: open, close, play, pause, type, search, switch to, start dictating, stop dictating."
    )
    assert hint.endswith("Apps: GIMP, Firefox.")


def test_build_hint_deduplicates_case_insensitively() -> None:
    hint = build_hint(["Firefox", "firefox", "FireFOx"], "voxa")
    assert hint.count("Firefox") == 1


def test_build_hint_orders_names_shortest_first() -> None:
    hint = build_hint(["LibreOffice", "GIMP", "Atom"], "voxa")
    assert "Apps: Atom, GIMP, LibreOffice." in hint


def test_build_hint_skips_overlong_names() -> None:
    long_name = "a" * 41
    hint = build_hint([long_name, "GIMP"], "voxa")
    assert long_name not in hint
    assert "GIMP" in hint


def test_build_hint_respects_max_chars() -> None:
    hint = build_hint(["Firefox", "GIMP", "Blender"], "voxa", max_chars=115)
    assert len(hint) <= 115
    assert "GIMP" in hint
    assert "Blender" not in hint
    assert "Firefox" not in hint


def test_build_hint_includes_extra_names() -> None:
    hint = build_hint(["GIMP"], "voxa", extra=["VLC"])
    assert "VLC" in hint


def test_refresh_app_names_collects_names_and_wordish_aliases(monkeypatch) -> None:
    apps = [
        DesktopApp(name="GIMP", path="/x", aliases=("GNU Image Manipulation Program", "gimp", "TheGIMP")),
        DesktopApp(name="Boxed App", path="/y", aliases=("not a word!",)),
    ]
    monkeypatch.setattr("voxa.vocabulary.list_apps", lambda: apps)
    monkeypatch.setattr("voxa.vocabulary._cache", None)
    assert refresh_app_names() == ["GIMP", "TheGIMP", "Boxed App"]


def test_refresh_app_names_caches_for_ten_minutes(monkeypatch) -> None:
    calls = {"n": 0}

    def fake() -> list[DesktopApp]:
        calls["n"] += 1
        return [DesktopApp(name="GIMP", path="/x")]

    monkeypatch.setattr("voxa.vocabulary.list_apps", fake)
    monkeypatch.setattr("voxa.vocabulary._cache", None)
    first = refresh_app_names()
    second = refresh_app_names()
    assert calls["n"] == 1
    assert first == second


def test_refresh_app_names_cache_expires_after_ten_minutes(monkeypatch) -> None:
    calls = {"n": 0}

    def fake() -> list[DesktopApp]:
        calls["n"] += 1
        return []

    class FakeTime:
        readings = iter((0.0, 601.0))

        def monotonic(self) -> float:
            return next(self.readings)

    monkeypatch.setattr("voxa.vocabulary.list_apps", fake)
    monkeypatch.setattr("voxa.vocabulary._cache", None)
    monkeypatch.setattr("voxa.vocabulary.time", FakeTime())
    refresh_app_names()
    refresh_app_names()
    assert calls["n"] == 2


def test_refresh_app_names_returns_empty_on_error(monkeypatch) -> None:
    def boom() -> list[DesktopApp]:
        raise RuntimeError("no apps here")

    monkeypatch.setattr("voxa.vocabulary.list_apps", boom)
    monkeypatch.setattr("voxa.vocabulary._cache", None)
    assert refresh_app_names() == []
