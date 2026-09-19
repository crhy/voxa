"""Reply identity: a stale 'speech finished' must never disturb a newer request (issue #7 sections 13 and 26)."""

from __future__ import annotations

from voxa.controller import AssistantController, ControllerPorts
from voxa.ui.state import AssistantModel, AssistantState


def _controller() -> tuple[AssistantController, int]:
    ports = ControllerPorts(
        start_listening=lambda: None,
        stop_listening=lambda: None,
        stop_speech=lambda: None,
        cancel_inference=lambda: None,
        microphone_active=lambda: False,
    )
    controller = AssistantController(AssistantModel(), ports)
    controller.activate()
    return controller, controller.token()


def _speaking(controller: AssistantController, token: int) -> int:
    controller.wake(token)
    controller.prompt_accepted(token)
    assert controller.reply_started(token)
    return controller.current_reply()


def test_finishing_the_current_reply_returns_to_ready() -> None:
    controller, token = _controller()
    reply = _speaking(controller, token)
    assert controller.reply_finished(token, reply_id=reply)
    assert controller.model.state is AssistantState.READY


def test_waiting_for_a_prompt_returns_to_listening() -> None:
    controller, token = _controller()
    reply = _speaking(controller, token)
    assert controller.reply_finished(token, waiting_for_prompt=True, reply_id=reply)
    assert controller.model.state is AssistantState.LISTENING


def test_completion_of_an_interrupted_reply_cannot_cut_short_the_next_request() -> None:
    controller, token = _controller()
    old_reply = _speaking(controller, token)
    assert controller.barge_in(token)

    controller.prompt_accepted(token)  # the user's interruption becomes a new request
    assert controller.model.state is AssistantState.THINKING
    assert controller.reply_finished(token, reply_id=old_reply) is False  # stale completion of the old speech
    assert controller.model.state is AssistantState.THINKING


def test_a_reply_id_from_an_earlier_reply_is_stale() -> None:
    controller, token = _controller()
    first = _speaking(controller, token)
    assert controller.reply_finished(token, reply_id=first)
    controller.prompt_accepted(token)
    second = _reply(controller, token)
    assert second != first
    assert controller.reply_finished(token, reply_id=first) is False
    assert controller.model.state is AssistantState.SPEAKING
    assert controller.reply_finished(token, reply_id=second)


def _reply(controller: AssistantController, token: int) -> int:
    assert controller.reply_started(token)
    return controller.current_reply()


def test_nothing_finishes_when_no_reply_is_in_progress() -> None:
    controller, token = _controller()
    assert controller.reply_finished(token) is False  # READY
    controller.wake(token)
    assert controller.reply_finished(token) is False  # LISTENING: the user is speaking
    assert controller.model.state is AssistantState.LISTENING


def test_a_request_that_produces_no_reply_can_finish_from_thinking() -> None:
    controller, token = _controller()
    controller.wake(token)
    controller.prompt_accepted(token)
    assert controller.reply_finished(token)
    assert controller.model.state is AssistantState.READY


def test_abandon_returns_every_working_state_to_ready_and_invalidates_the_reply() -> None:
    for setup in ("listening", "thinking", "speaking"):
        controller, token = _controller()
        controller.wake(token)
        reply = controller.current_reply()
        if setup in ("thinking", "speaking"):
            controller.prompt_accepted(token)
        if setup == "speaking":
            controller.reply_started(token)
            reply = controller.current_reply()
        assert controller.abandon(token), setup
        assert controller.model.state is AssistantState.READY, setup
        assert controller.reply_finished(token, reply_id=reply) is False, setup


def test_abandon_does_nothing_when_offline_or_stale() -> None:
    controller, token = _controller()
    controller.go_offline()
    assert controller.abandon(token) is False
    assert controller.model.state is AssistantState.OFFLINE
