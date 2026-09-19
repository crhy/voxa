from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.styles import VOXA_CSS, install_styles  # noqa: E402


def test_install_styles_is_idempotent() -> None:
    first = install_styles()
    second = install_styles()
    assert first is second
    assert isinstance(first, Gtk.CssProvider)


def test_install_styles_accepts_explicit_display() -> None:
    display = Gdk.Display.get_default()
    assert install_styles(display) is install_styles(display)


def test_css_contains_expected_classes() -> None:
    for cls in (
        ".voxa-active",
        ".voxa-offline",
        ".voxa-task-panel",
        ".voxa-task-title",
        ".voxa-task-detail",
        ".voxa-section",
        ".voxa-choice",
        ".voxa-state",
        ".selected",
        ".dimmed",
    ):
        assert cls in VOXA_CSS
