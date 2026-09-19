"""The assistant controller: ACTIVE and OFFLINE as a real, testable kill switch.

This module is GTK-free and starts nothing at import or construction. The window
wires it to the real services through ``ControllerPorts``; tests wire it to fakes.

Two ideas make OFFLINE trustworthy:

* **Generations.** Every activation and every ``go_offline`` bumps
  ``AssistantModel.generation``. Async work (a queued microphone callback, a
  transcription, a model reply, a speech-finished notification) captures
  ``controller.token()`` when it starts and reports back through the event
  methods below with that token. Once the generation moves on, a late callback is
  ignored, so it can never revive listening, restart work, or move the assistant
  out of OFFLINE.
* **Every stop is attempted.** ``go_offline`` calls each stop port inside its own
  try/except, so one failing service cannot leave the microphone or speech running,
  and it checks afterwards that the microphone really is inactive.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from .ui.state import AssistantModel, AssistantState, VoxaTask

log = logging.getLogger(__name__)

OFFLINE_CONFIRMED = "Offline — microphone off"
OFFLINE_UNCONFIRMED = "Offline — microphone may still be active"


@dataclass(slots=True)
class ControllerPorts:
    """The side effects the controller may perform, supplied by the window (or a test)."""

    #: Start microphone capture and wake-word monitoring; raise on failure (no microphone...).
    start_listening: Callable[[], None]
    #: Stop microphone capture, wake-word monitoring and any dictation.
    stop_listening: Callable[[], None]
    stop_speech: Callable[[], None]
    #: Abort the in-flight model request, where supported.
    cancel_inference: Callable[[], None]
    #: True while any microphone capture is still running.
    microphone_active: Callable[[], bool]


class AssistantController:
    def __init__(self, model: AssistantModel, ports: ControllerPorts) -> None:
        self.model = model
        self.ports = ports

    # ------------------------------------------------------------ session tokens

    def token(self) -> int:
        """Capture this when starting async work; hand it back with the result."""
        return self.model.generation

    def accepts(self, token: int) -> bool:
        """True only while ACTIVE and only for work started in the current generation."""
        return token == self.model.generation and self.model.state is not AssistantState.OFFLINE

    @property
    def is_active(self) -> bool:
        return self.model.state is not AssistantState.OFFLINE

    # ------------------------------------------------------------- ACTIVE / OFFLINE

    def activate(self) -> bool:
        """ACTIVE: start listening. Returns False (and ends up OFFLINE) if that fails."""
        if self.is_active:
            return True
        generation = self.model.bump_generation()
        try:
            self.ports.start_listening()
        except Exception as exc:  # noqa: BLE001 - platform boundary (audio, whisper, ...)
            log.warning("could not start listening: %s", exc)
            self._stop_quietly(self.ports.stop_listening, "stop_listening")
            self.model.set_state(AssistantState.ERROR, str(exc) or "Could not start listening", generation=generation)
            self.model.set_state(AssistantState.OFFLINE, "Offline", generation=generation)
            return False
        return self.model.set_state(AssistantState.READY, "", generation=generation)

    def go_offline(self) -> bool:
        """OFFLINE: the kill switch. Safe in any state, any number of times.

        Returns True when the microphone is confirmed inactive afterwards.
        """
        # First: everything already in flight becomes stale.
        self.model.bump_generation()
        self._stop_quietly(self.ports.stop_listening, "stop_listening")
        self._stop_quietly(self.ports.stop_speech, "stop_speech")
        self._stop_quietly(self.ports.cancel_inference, "cancel_inference")
        self._cancel_tasks()

        confirmed = not self._microphone_active()
        if not confirmed:
            self._stop_quietly(self.ports.stop_listening, "stop_listening (retry)")
            confirmed = not self._microphone_active()
        if not confirmed:
            log.warning("OFFLINE: the microphone still reports as active")
        self.model.set_state(AssistantState.OFFLINE, OFFLINE_CONFIRMED if confirmed else OFFLINE_UNCONFIRMED)
        return confirmed

    # ---------------------------------------------------------------- pipeline events
    # Each takes the token captured when the work started and returns False when the
    # event was ignored (stale generation, OFFLINE, or a transition the policy forbids).

    def wake(self, token: int) -> bool:
        return self._apply(token, AssistantState.LISTENING)

    def prompt_accepted(self, token: int, detail: str = "") -> bool:
        return self._apply(token, AssistantState.THINKING, detail)

    def reply_started(self, token: int) -> bool:
        return self._apply(token, AssistantState.SPEAKING)

    def reply_finished(self, token: int, waiting_for_prompt: bool = False) -> bool:
        return self._apply(token, AssistantState.LISTENING if waiting_for_prompt else AssistantState.READY)

    def barge_in(self, token: int) -> bool:
        """The user talked over Voxa: SPEAKING -> LISTENING."""
        if self.model.state is not AssistantState.SPEAKING:
            return False
        return self._apply(token, AssistantState.LISTENING)

    def failed(self, token: int, detail: str) -> bool:
        return self._apply(token, AssistantState.ERROR, detail)

    def recover(self, token: int) -> bool:
        """ERROR -> READY. A no-op when the assistant is not in ERROR or has gone OFFLINE."""
        if self.model.state is not AssistantState.ERROR:
            return False
        return self._apply(token, AssistantState.READY)

    # ------------------------------------------------------------------------ tasks

    def begin_task(self, title: str, detail: str = "") -> VoxaTask | None:
        """Start visible work. Refused (None) while OFFLINE: no new work begins offline."""
        if not self.is_active:
            return None
        task = self.model.add_task(title, detail)
        return self.model.start_task(task.id)

    # ---------------------------------------------------------------------- internals

    def _apply(self, token: int, state: AssistantState, detail: str = "") -> bool:
        if not self.accepts(token):
            return False
        return self.model.set_state(state, detail, generation=token)

    def _cancel_tasks(self) -> None:
        for task in self.model.active_tasks() + self.model.todo_tasks():
            try:
                self.model.cancel_task(task.id)
            except (KeyError, ValueError):  # finished or removed in the meantime
                continue

    def _microphone_active(self) -> bool:
        try:
            return bool(self.ports.microphone_active())
        except Exception:  # noqa: BLE001 - an unreadable probe must not look like "confirmed off"
            log.exception("microphone_active probe failed")
            return True

    @staticmethod
    def _stop_quietly(action: Callable[[], None], name: str) -> None:
        try:
            action()
        except Exception:  # noqa: BLE001 - one failing service must not stop the others
            log.exception("OFFLINE: %s failed", name)
