from __future__ import annotations

from voxa.controller import AssistantController, ControllerPorts
from voxa.ui.state import AssistantModel, AssistantState


def _controller() -> AssistantController:
    ports = ControllerPorts(
        start_listening=lambda: None,
        stop_listening=lambda: None,
        stop_speech=lambda: None,
        cancel_inference=lambda: None,
        microphone_active=lambda: False,
    )
    return AssistantController(AssistantModel(), ports)


def test_speaking_again_during_an_error_is_not_swallowed() -> None:
    controller = _controller()
    controller.activate()
    token = controller.token()
    controller.wake(token)
    controller.prompt_accepted(token)
    assert controller.failed(token, "The AI request failed")
    assert controller.model.state is AssistantState.ERROR

    # The user speaks again before the automatic recovery: the next request proceeds.
    assert controller.wake(token)
    assert controller.model.state is AssistantState.LISTENING
    assert controller.prompt_accepted(token)
    assert controller.reply_started(token)
    assert controller.model.state is AssistantState.SPEAKING


def test_a_late_recovery_does_not_disturb_a_request_that_started_meanwhile() -> None:
    controller = _controller()
    controller.activate()
    token = controller.token()
    controller.failed(token, "oops")
    controller.prompt_accepted(token)
    assert controller.model.state is AssistantState.THINKING
    assert controller.recover(token) is False  # the timer fires, but we are no longer in ERROR
    assert controller.model.state is AssistantState.THINKING


def test_errors_never_revive_an_offline_assistant() -> None:
    controller = _controller()
    controller.activate()
    token = controller.token()
    controller.failed(token, "oops")
    controller.go_offline()
    assert controller.wake(token) is False and controller.prompt_accepted(token) is False
    assert controller.model.state is AssistantState.OFFLINE
