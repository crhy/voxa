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
