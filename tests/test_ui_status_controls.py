from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.state import AssistantState  # noqa: E402
from voxa.ui.status_controls import StatusControls  # noqa: E402


def test_offline_selects_offline_button() -> None:
    controls = StatusControls()
    controls.set_state(AssistantState.OFFLINE)
    assert controls.offline_button.has_css_class("selected")
    assert controls.active_button.has_css_class("dimmed")
    assert not controls.active_button.has_css_class("selected")


def test_any_active_state_selects_active_button() -> None:
    controls = StatusControls()
    for state in (
        AssistantState.READY,
        AssistantState.LISTENING,
        AssistantState.THINKING,
        AssistantState.SPEAKING,
        AssistantState.WORKING,
        AssistantState.WAITING,
        AssistantState.ERROR,
    ):
        controls.set_state(state)
        assert controls.active_button.has_css_class("selected")
        assert controls.offline_button.has_css_class("dimmed")
        assert not controls.offline_button.has_css_class("selected")
    controls.set_state(AssistantState.OFFLINE)
    assert controls.active_button.has_css_class("dimmed")
    assert not controls.active_button.has_css_class("selected")


def test_callbacks_fire() -> None:
    events: list[str] = []
    controls = StatusControls(on_active=lambda: events.append("active"), on_offline=lambda: events.append("offline"))
    controls.active_button.emit("clicked")
    controls.offline_button.emit("clicked")
    assert events == ["active", "offline"]
