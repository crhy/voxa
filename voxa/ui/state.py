"""UI-independent assistant state and task model for Voxa.

This module deliberately has no GTK (``gi``) dependency so it can be unit
tested and used by headless workers. Assistant state and task lifecycle are
owned here; the UI only renders from them.

Threading contract: mutations can come from any worker thread (transcription,
agent execution, TTS). ``on_state_changed`` and ``on_tasks_changed`` may
therefore be called from any thread, and the GTK layer must marshal them with
``GLib.idle_add`` before touching widgets. Callbacks are invoked with copies of
the model data *after* the internal lock is released, so a callback may safely
call back into the model.

Transition policy
-----------------
``AssistantState`` is the only high-level presentation state, and ``set_state``
enforces this policy (anything not listed is rejected)::

    OFFLINE   -> READY
    READY     -> LISTENING | THINKING | WORKING
    LISTENING -> THINKING | READY
    THINKING  -> SPEAKING | READY | WORKING | WAITING
    SPEAKING  -> READY | LISTENING | THINKING
    WORKING   -> WAITING | READY | THINKING | SPEAKING
    WAITING   -> WORKING | READY
    ERROR     -> READY | OFFLINE
    (any state) -> OFFLINE and ERROR;  staying in the same state is always fine

``THINKING`` may follow ``READY`` directly because a wake phrase can already
carry its request. ``SPEAKING -> LISTENING`` is barge-in: the user talks over
Voxa, so speech stops and Voxa listens.

Generations
-----------
``AssistantModel.generation`` is bumped whenever Voxa is activated or goes OFFLINE.
Async work captures the generation when it starts and passes it back to
``set_state(..., generation=...)``; a result from an older generation is ignored,
so a stale callback can never move the assistant backward or revive an OFFLINE
assistant.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto

log = logging.getLogger(__name__)


class AssistantState(Enum):
    """High-level assistant state driving the avatar, caption and controls."""

    OFFLINE = auto()
    READY = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    WORKING = auto()
    WAITING = auto()
    ERROR = auto()


class TaskState(Enum):
    """Lifecycle of a single visible Voxa task."""

    QUEUED = auto()
    RUNNING = auto()
    WAITING = auto()
    DONE = auto()
    FAILED = auto()
    CANCELLED = auto()


_S = AssistantState
_ANYTIME = frozenset({_S.OFFLINE, _S.ERROR})  # every state may go OFFLINE or ERROR

#: The explicit transition policy (see the module docstring).
ALLOWED_TRANSITIONS: dict[AssistantState, frozenset[AssistantState]] = {
    _S.OFFLINE: frozenset({_S.READY}) | _ANYTIME,
    _S.READY: frozenset({_S.LISTENING, _S.THINKING, _S.WORKING}) | _ANYTIME,
    _S.LISTENING: frozenset({_S.THINKING, _S.READY}) | _ANYTIME,
    _S.THINKING: frozenset({_S.SPEAKING, _S.READY, _S.WORKING, _S.WAITING}) | _ANYTIME,
    _S.SPEAKING: frozenset({_S.READY, _S.LISTENING, _S.THINKING}) | _ANYTIME,
    _S.WORKING: frozenset({_S.WAITING, _S.READY, _S.THINKING, _S.SPEAKING}) | _ANYTIME,
    _S.WAITING: frozenset({_S.WORKING, _S.READY}) | _ANYTIME,
    _S.ERROR: frozenset({_S.READY}) | _ANYTIME,
}


def can_transition(current: AssistantState, new: AssistantState) -> bool:
    """Whether the policy allows moving from ``current`` to ``new`` (same state always may)."""
    return current is new or new in ALLOWED_TRANSITIONS[current]


#: States a task can never leave.
TERMINAL_TASK_STATES = frozenset({TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED})


@dataclass(slots=True)
class VoxaTask:
    """A unit of assistant work shown in the task panel."""

    title: str
    detail: str = ""
    state: TaskState = TaskState.QUEUED
    progress: float | None = None
    choices: list[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def requires_user_input(self) -> bool:
        """True when the task is blocked on an explicit user choice."""
        return self.state is TaskState.WAITING and bool(self.choices)


def _copy_task(task: VoxaTask) -> VoxaTask:
    """An independent copy, so readers on another thread never see a half-updated task."""
    return replace(task, choices=list(task.choices))


class AssistantModel:
    """Thread-safe assistant state plus task store with change callbacks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.state = AssistantState.OFFLINE
        self.detail = ""
        self._generation = 0
        self.tasks: dict[str, VoxaTask] = {}

        self.on_state_changed: Callable[[AssistantState, str], None] | None = None
        self.on_tasks_changed: Callable[[list[VoxaTask]], None] | None = None

    # ------------------------------------------------------------------ state

    @property
    def generation(self) -> int:
        """Bumped on every activation and every OFFLINE; see the module docstring."""
        return self._generation

    def bump_generation(self) -> int:
        """Invalidate every callback captured under the previous generation."""
        with self._lock:
            self._generation += 1
            return self._generation

    def set_state(self, state: AssistantState, detail: str = "", *, generation: int | None = None) -> bool:
        """Apply a new assistant state; returns whether it was applied.

        Rejected (False, no callback) when ``generation`` is given and is not the current
        one - a stale callback - or when the transition policy forbids the move. Asking for
        the state and detail already in place succeeds without firing the callback.
        """
        with self._lock:
            if generation is not None and generation != self._generation:
                log.debug("ignored stale state change to %s (generation %s, current %s)", state.name, generation, self._generation)
                return False
            if self.state is state and self.detail == detail:
                return True
            if not can_transition(self.state, state):
                log.debug("ignored illegal state change %s -> %s", self.state.name, state.name)
                return False
            self.state = state
            self.detail = detail

        callback = self.on_state_changed
        if callback is not None:
            callback(state, detail)
        return True

    # ------------------------------------------------------------------ tasks

    def add_task(self, title: str, detail: str = "") -> VoxaTask:
        """Create a queued task and return it."""
        created = time.time()
        task = VoxaTask(
            title=title,
            detail=detail,
            state=TaskState.QUEUED,
            created_at=created,
            updated_at=created,
        )
        with self._lock:
            self.tasks[task.id] = task

        self._emit_tasks()
        return task

    def update_task(
        self,
        task_id: str,
        *,
        detail: str | None = None,
        state: TaskState | None = None,
        progress: float | None = None,
    ) -> None:
        """Apply partial updates to a task, rejecting terminal transitions."""
        with self._lock:
            task = self._require_task(task_id)
            if state is not None:
                self._transition(task, state)
            if detail is not None:
                task.detail = detail
            if progress is not None:
                task.progress = max(0.0, min(1.0, progress))
            task.updated_at = time.time()

        self._emit_tasks()

    def start_task(self, task_id: str, detail: str | None = None) -> VoxaTask:
        """Move a task to RUNNING."""
        with self._lock:
            task = self._require_task(task_id)
            self._transition(task, TaskState.RUNNING)
            if detail is not None:
                task.detail = detail
            task.updated_at = time.time()

        self._emit_tasks()
        return task

    def complete_task(self, task_id: str, result: str = "") -> VoxaTask:
        """Finish a task with progress 1.0 and an optional result summary."""
        with self._lock:
            task = self._require_task(task_id)
            self._transition(task, TaskState.DONE)
            task.progress = 1.0
            task.result = result
            task.updated_at = time.time()

        self._emit_tasks()
        return task

    def fail_task(self, task_id: str, error: str) -> VoxaTask:
        """Mark a task FAILED with a human-readable error."""
        with self._lock:
            task = self._require_task(task_id)
            self._transition(task, TaskState.FAILED)
            task.error = error
            task.updated_at = time.time()

        self._emit_tasks()
        return task

    def cancel_task(self, task_id: str) -> VoxaTask:
        """Mark a task CANCELLED."""
        with self._lock:
            task = self._require_task(task_id)
            self._transition(task, TaskState.CANCELLED)
            task.updated_at = time.time()

        self._emit_tasks()
        return task

    def request_choice(self, task_id: str, choices: list[str]) -> None:
        """Put a task in WAITING with the offered choices."""
        with self._lock:
            task = self._require_task(task_id)
            self._transition(task, TaskState.WAITING)
            task.choices = list(choices)
            task.updated_at = time.time()

        self._emit_tasks()

    def resolve_choice(self, task_id: str, choice: str) -> str:
        """Answer a choice request, resume the task, and return the choice."""
        with self._lock:
            task = self._require_task(task_id)
            if choice not in task.choices:
                raise ValueError(f"unknown choice {choice!r} for task {task.title!r}")
            self._transition(task, TaskState.RUNNING)
            task.choices = []
            task.updated_at = time.time()

        self._emit_tasks()
        return choice

    # ------------------------------------------------------------- queries

    def active_tasks(self) -> list[VoxaTask]:
        """Tasks currently RUNNING or WAITING, in creation order."""
        with self._lock:
            return [
                _copy_task(task)
                for task in self.tasks.values()
                if task.state in (TaskState.RUNNING, TaskState.WAITING)
            ]

    def todo_tasks(self) -> list[VoxaTask]:
        """Queued tasks, in creation order."""
        with self._lock:
            return [_copy_task(t) for t in self.tasks.values() if t.state is TaskState.QUEUED]

    # ------------------------------------------------------------ internals

    def _require_task(self, task_id: str) -> VoxaTask:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task id {task_id!r}")
        return task

    def _transition(self, task: VoxaTask, state: TaskState) -> None:
        if task.state in TERMINAL_TASK_STATES and task.state is not state:
            raise ValueError(
                f"task {task.title!r} is {task.state.name} and cannot move to {state.name}"
            )
        task.state = state

    def _emit_tasks(self) -> None:
        callback = self.on_tasks_changed
        if callback is None:
            return
        callback(self._task_snapshot())

    def _task_snapshot(self) -> list[VoxaTask]:
        with self._lock:
            return [_copy_task(task) for task in self.tasks.values()]
