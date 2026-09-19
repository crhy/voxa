"""OFFLINE as a real kill switch: the controller's guarantees (roadmap #3, #26).

Every test wires ``AssistantController`` to fake ports that record each call in
order, so the assertions are about what the assistant actually *did*, not just
what state it ended in. No GTK, no audio, no model: nothing starts at import.
"""

from __future__ import annotations

from voxa.controller import (
    OFFLINE_CONFIRMED,
    OFFLINE_UNCONFIRMED,
    AssistantController,
    ControllerPorts,
)
from voxa.ui.state import AssistantModel, AssistantState, TaskState


class FakePorts:
    """Records every port call in order; any port can be made to raise."""

    def __init__(
        self,
        *,
        start_fails: str = "",
        stop_listening_fails: str = "",
        stop_speech_fails: str = "",
        cancel_inference_fails: str = "",
        microphone_probe_fails: str = "",
        microphone_stays_on: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self._microphone_on = False
        self.start_fails = start_fails
        self.stop_listening_fails = stop_listening_fails
        self.stop_speech_fails = stop_speech_fails
        self.cancel_inference_fails = cancel_inference_fails
        self.microphone_probe_fails = microphone_probe_fails
        self.microphone_stays_on = microphone_stays_on

    def start_listening(self) -> None:
        self.calls.append("start_listening")
        if self.start_fails:
            raise RuntimeError(self.start_fails)
        self._microphone_on = True

    def stop_listening(self) -> None:
        self.calls.append("stop_listening")
        if self.stop_listening_fails:
            raise RuntimeError(self.stop_listening_fails)
        self._microphone_on = False

    def stop_speech(self) -> None:
        self.calls.append("stop_speech")
        if self.stop_speech_fails:
            raise RuntimeError(self.stop_speech_fails)

    def cancel_inference(self) -> None:
        self.calls.append("cancel_inference")
        if self.cancel_inference_fails:
            raise RuntimeError(self.cancel_inference_fails)

    def microphone_active(self) -> bool:
        self.calls.append("microphone_active")
        if self.microphone_probe_fails:
            raise RuntimeError(self.microphone_probe_fails)
        return self._microphone_on or self.microphone_stays_on

    def calls_without_probe(self) -> list[str]:
        return [name for name in self.calls if name != "microphone_active"]


def _controller(**port_kwargs) -> tuple[AssistantController, FakePorts]:
    ports = FakePorts(**port_kwargs)
    model_ports = ControllerPorts(
        start_listening=ports.start_listening,
        stop_listening=ports.stop_listening,
        stop_speech=ports.stop_speech,
        cancel_inference=ports.cancel_inference,
        microphone_active=ports.microphone_active,
    )
    return AssistantController(AssistantModel(), model_ports), ports


def _state_events(model: AssistantModel) -> list[tuple[AssistantState, str]]:
    events: list[tuple[AssistantState, str]] = []
    model.on_state_changed = lambda state, detail: events.append((state, detail))
    return events


def _active(controller: AssistantController) -> int:
    """Activate and return the session token the window would capture."""
    assert controller.activate()
    return controller.token()


# --------------------------------------------------------------------- ACTIVE


def test_activation_success_starts_listening_and_ends_ready() -> None:
    controller, ports = _controller()
    assert controller.activate() is True
    assert ports.calls_without_probe() == ["start_listening"]
    assert controller.model.state is AssistantState.READY
    assert controller.is_active
    assert controller.model.generation == 1


def test_activation_failure_shows_error_first_and_ends_offline(caplog) -> None:
    controller, ports = _controller(start_fails="no microphone device")
    events = _state_events(controller.model)

    assert controller.activate() is False
    assert ports.calls_without_probe() == ["start_listening", "stop_listening"]
    assert events == [
        (AssistantState.ERROR, "no microphone device"),
        (AssistantState.OFFLINE, "Offline"),
    ]
    assert controller.model.state is AssistantState.OFFLINE
    assert controller.is_active is False


def test_activating_twice_does_not_restart_the_microphone() -> None:
    controller, ports = _controller()
    controller.activate()
    assert controller.activate() is True
    assert ports.calls_without_probe() == ["start_listening"]


# --------------------------------------------------- OFFLINE from each pipeline


def test_offline_while_listening_stops_the_microphone_and_blocks_wake() -> None:
    controller, ports = _controller()
    token = _active(controller)
    assert controller.wake(token)
    assert controller.model.state is AssistantState.LISTENING

    assert controller.go_offline() is True
    assert ports.calls_without_probe() == ["start_listening", "stop_listening", "stop_speech", "cancel_inference"]
    assert controller.model.state is AssistantState.OFFLINE
    assert controller.model.detail == OFFLINE_CONFIRMED
    assert controller.wake(token) is False


def test_offline_while_transcribing_ignores_the_late_prompt_and_calls_nothing() -> None:
    controller, ports = _controller()
    token = _active(controller)
    controller.wake(token)
    controller.prompt_accepted(token)  # would move to THINKING if it were current

    assert controller.model.state is AssistantState.THINKING
    assert controller.go_offline() is True
    settled = list(ports.calls)

    assert controller.prompt_accepted(token, "half a request") is False
    assert ports.calls == settled
    assert controller.model.state is AssistantState.OFFLINE
    assert controller.model.detail == OFFLINE_CONFIRMED


def test_offline_while_thinking_cancels_inference_and_ignores_the_late_reply() -> None:
    controller, ports = _controller()
    token = _active(controller)
    controller.wake(token)
    assert controller.prompt_accepted(token)
    assert controller.model.state is AssistantState.THINKING

    assert controller.go_offline() is True
    assert "cancel_inference" in ports.calls

    assert controller.reply_started(token) is False
    assert controller.reply_finished(token) is False
    assert controller.model.state is AssistantState.OFFLINE


def test_offline_while_speaking_stops_speech_and_ignores_the_late_finish() -> None:
    controller, ports = _controller()
    token = _active(controller)
    controller.wake(token)
    controller.prompt_accepted(token)
    assert controller.reply_started(token)
    assert controller.model.state is AssistantState.SPEAKING

    assert controller.go_offline() is True
    assert "stop_speech" in ports.calls

    assert controller.reply_finished(token) is False
    assert controller.reply_finished(token, waiting_for_prompt=True) is False
    assert controller.model.state is AssistantState.OFFLINE


def test_offline_while_a_task_is_running_cancels_it_and_refuses_new_work() -> None:
    controller, ports = _controller()
    _active(controller)
    task = controller.begin_task("Rename the screenshots folder")
    assert task is not None
    assert task.state is TaskState.RUNNING

    assert controller.go_offline() is True
    assert controller.model.tasks[task.id].state is TaskState.CANCELLED
    assert controller.begin_task("Anything else") is None
    assert controller.model.active_tasks() == []


def test_offline_while_a_task_waits_for_a_choice_cancels_it_and_clears_the_choice() -> None:
    controller, ports = _controller()
    _active(controller)
    task = controller.begin_task("Install the TTS voices")
    assert task is not None
    controller.model.request_choice(task.id, ["yes", "no"])
    assert controller.model.tasks[task.id].requires_user_input

    assert controller.go_offline() is True
    cancelled = controller.model.tasks[task.id]
    assert cancelled.state is TaskState.CANCELLED
    assert cancelled.requires_user_input is False
    assert controller.model.active_tasks() == []
    assert ports.calls_without_probe() == [
        "start_listening",
        "stop_listening",
        "stop_speech",
        "cancel_inference",
    ]


# ------------------------------------------------------- OFFLINE robustness


def test_a_failing_stop_listening_still_stops_speech_and_inference(caplog) -> None:
    controller, ports = _controller(stop_listening_fails="audio device busy")
    _active(controller)

    assert controller.go_offline() is False
    assert ports.calls_without_probe() == [
        "start_listening",
        "stop_listening",
        "stop_speech",
        "cancel_inference",
        "stop_listening",
    ]
    assert controller.model.state is AssistantState.OFFLINE
    assert "audio device busy" in caplog.text


def test_a_microphone_that_stays_on_is_reported_as_unconfirmed(caplog) -> None:
    controller, ports = _controller(microphone_stays_on=True)
    _active(controller)

    assert controller.go_offline() is False
    assert controller.model.state is AssistantState.OFFLINE
    assert controller.model.detail == OFFLINE_UNCONFIRMED
    assert ports.calls_without_probe().count("stop_listening") == 2
    assert "microphone still reports as active" in caplog.text


def test_a_broken_microphone_probe_is_not_read_as_confirmed_offline(caplog) -> None:
    controller, ports = _controller(microphone_probe_fails="device vanished")
    _active(controller)

    assert controller.go_offline() is False
    assert controller.model.state is AssistantState.OFFLINE
    assert controller.model.detail == OFFLINE_UNCONFIRMED
    assert "microphone_active probe failed" in caplog.text


def test_go_offline_twice_is_harmless() -> None:
    controller, ports = _controller()
    _active(controller)
    assert controller.go_offline() is True
    first_generation = controller.model.generation

    assert controller.go_offline() is True
    assert controller.model.generation == first_generation + 1
    assert controller.model.state is AssistantState.OFFLINE
    assert controller.is_active is False


def test_reactivating_works_and_every_older_token_is_still_rejected() -> None:
    controller, ports = _controller()
    old_token = _active(controller)
    controller.wake(old_token)
    controller.prompt_accepted(old_token)
    assert controller.go_offline() is True

    assert controller.activate() is True
    assert controller.model.state is AssistantState.READY
    assert controller.is_active
    assert ports.calls_without_probe().count("start_listening") == 2

    assert controller.accepts(old_token) is False
    assert controller.wake(old_token) is False
    assert controller.prompt_accepted(old_token) is False
    assert controller.reply_started(old_token) is False
    assert controller.reply_finished(old_token) is False
    assert controller.begin_task("A leftover task") is not None

    new_token = controller.token()
    assert controller.accepts(new_token) is True
    assert controller.wake(new_token) is True


def test_rapid_barge_in_then_a_stale_reply_finished_is_rejected() -> None:
    controller, ports = _controller()
    token = _active(controller)
    controller.wake(token)
    controller.prompt_accepted(token)
    assert controller.reply_started(token)
    assert controller.model.state is AssistantState.SPEAKING

    assert controller.barge_in(token) is True
    assert controller.model.state is AssistantState.LISTENING

    assert controller.reply_finished(token) is False
    assert controller.reply_finished(token, waiting_for_prompt=True) is False
    assert controller.model.state is AssistantState.LISTENING

    assert controller.prompt_accepted(token) is True
    assert controller.model.state is AssistantState.THINKING
