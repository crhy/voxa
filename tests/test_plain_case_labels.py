from __future__ import annotations

from pathlib import Path

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.assistant_view import STATE_HINTS  # noqa: E402
from voxa.ui.state import AssistantState  # noqa: E402
from voxa.ui.status_controls import StatusControls  # noqa: E402


def test_button_labels_are_plain_case() -> None:
    controls = StatusControls()
    assert controls.active_button.get_label() == "Active"
    assert controls.pause_button.get_label() == "Pause"
    assert controls.offline_button.get_label() == "Offline"


def test_offline_hint_uses_plain_case() -> None:
    assert STATE_HINTS[AssistantState.OFFLINE] == "Press Active to start listening"


def test_face_quality_has_no_hi_res_text() -> None:
    source = Path("voxa/ui/face_quality.py").read_text(encoding="utf-8")
    assert "Hi-Res" not in source
