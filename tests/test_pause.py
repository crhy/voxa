"""X6a: the PAUSED state and the PAUSE button."""

import pytest

from voxa.controller import AssistantController
from voxa.ui.state import AssistantModel, AssistantState, can_transition


class FakePorts:
    """Minimal stand-in for the side-effect ports the controller drives."""

    def __init__(self) -> None:
        self.stop_speech_calls = 0

    def start_listening(self) -> None:
        pass

    def stop_listening(self) -> None:
        pass

    def stop_speech(self) -> None:
        self.stop_speech_calls += 1

    def cancel_inference(self) -> None:
        pass

    def microphone_active(self) -> bool:
        return False


def test_pause_resume_transitions():
    model = AssistantModel()
    assert model.set_state(AssistantState.READY)
    ports = FakePorts()
    controller = AssistantController(model, ports)

    assert controller.pause() is True
    assert model.state is AssistantState.PAUSED
    assert ports.stop_speech_calls == 1

    assert controller.resume() is True
    assert model.state is AssistantState.READY


def test_paused_to_offline():
    model = AssistantModel()
    assert model.set_state(AssistantState.READY)
    assert model.set_state(AssistantState.PAUSED)
    assert model.set_state(AssistantState.OFFLINE) is True
    assert model.state is AssistantState.OFFLINE


def test_offline_to_paused_refused():
    assert can_transition(AssistantState.OFFLINE, AssistantState.PAUSED) is False
    model = AssistantModel()
    assert model.set_state(AssistantState.PAUSED) is False
    assert model.state is AssistantState.OFFLINE


def test_pause_button_fires_on_pause_once():
    gi = pytest.importorskip("gi")
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk  # noqa: E402

    from voxa.ui.status_controls import StatusControls  # noqa: E402

    fired = []
    controls = StatusControls(on_pause=lambda: fired.append(1))
    assert isinstance(controls, Gtk.Box)
    controls.pause_button.emit("clicked")
    assert fired == [1]


def test_set_state_paused_selects_pause_button():
    gi = pytest.importorskip("gi")
    gi.require_version("Gtk", "4.0")

    from voxa.ui.status_controls import StatusControls  # noqa: E402

    controls = StatusControls()
    controls.set_state(AssistantState.PAUSED)

    assert controls.pause_button.has_css_class("selected")
    assert not controls.pause_button.has_css_class("dimmed")
    for other in (controls.active_button, controls.offline_button):
        assert other.has_css_class("dimmed")
        assert not other.has_css_class("selected")
