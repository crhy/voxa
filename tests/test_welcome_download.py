from __future__ import annotations

import inspect
import types

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

Adw.init()

from voxa.config import Settings  # noqa: E402
from voxa.welcome import SMALLEST_MODEL, download_status  # noqa: E402
from voxa.window import MainWindow  # noqa: E402


def test_download_status_table() -> None:
    assert download_status("", 0, 0) == f"Downloading {SMALLEST_MODEL}…"
    assert download_status("pulling manifest", 0, 0) == f"Downloading {SMALLEST_MODEL}… pulling manifest"
    assert download_status("downloading", 50, 200) == f"Downloading {SMALLEST_MODEL}… 25%"
    assert download_status("downloading", 300, 200) == f"Downloading {SMALLEST_MODEL}… 100%"


def test_on_welcome_download_done() -> None:
    calls: list[str] = []
    ns = types.SimpleNamespace(
        settings=Settings(),
        _set_status=lambda text, busy=False: calls.append("status"),
        _toast=lambda message: calls.append("toast"),
        _refresh_ollama_models=lambda: calls.append("refresh"),
        _show_model_manager=lambda: calls.append("manager"),
    )
    assert MainWindow._on_welcome_download_done(ns, True, "x") is False
    assert "refresh" in calls
    assert "manager" not in calls

    calls.clear()
    assert MainWindow._on_welcome_download_done(ns, False, "boom") is False
    assert "manager" in calls
    assert "refresh" not in calls


def test_welcome_primary_uses_download() -> None:
    source = inspect.getsource(MainWindow._on_welcome_primary)
    assert "_download_welcome_model" in source
    assert "_show_model_manager" not in source


def test_on_welcome_primary_model_downloads_directly() -> None:
    """The "model" step routes to the direct download, never to the model-selection screen."""
    calls: list[str] = []
    ns = types.SimpleNamespace(
        speech=types.SimpleNamespace(stop=lambda: calls.append("stop")),
        _start_ollama_install=lambda: calls.append("install"),
        _download_welcome_model=lambda: calls.append("download"),
        _download_small_brain=lambda: calls.append("brain"),
        _show_model_manager=lambda: calls.append("manager"),
    )
    MainWindow._on_welcome_primary(ns, "model")
    assert calls == ["stop", "download"]
    assert "manager" not in calls


def test_download_welcome_model_pulls_smallest_without_network(monkeypatch) -> None:
    """The "model" step pulls SMALLEST_MODEL straight through the Ollama download call:
    no model screen opens, and no real network is touched (the client is faked)."""
    import time

    import voxa.window as w

    pulled: dict[str, str] = {}

    class FakeClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def list_models(self) -> list[str]:
            return []

        def pull_model(self, name, *, cancel_event, on_progress) -> None:
            pulled["name"] = name
            on_progress("success", 100, 100)

    monkeypatch.setattr(w, "OllamaClient", FakeClient)
    monkeypatch.setattr(w, "idle", lambda fn, *args, **kw: fn(*args, **kw))

    calls: list[tuple[str, str]] = []
    ns = types.SimpleNamespace(
        settings=Settings(),
        _welcome_downloading=False,
        _welcome_cancel=None,
        _set_status=lambda text, busy=False: calls.append(("status", text)),
        _toast=lambda message: calls.append(("toast", message)),
        _refresh_ollama_models=lambda: calls.append(("refresh", "")),
        _show_model_manager=lambda: calls.append(("manager", "")),
    )

    MainWindow._download_welcome_model(ns)
    for _ in range(100):
        if not ns._welcome_downloading:
            break
        time.sleep(0.02)

    assert pulled["name"] == SMALLEST_MODEL
    assert not any(kind == "manager" for kind, _ in calls)
    assert any(kind == "refresh" for kind, _ in calls)
    assert any(kind == "status" and SMALLEST_MODEL in text for kind, text in calls)
