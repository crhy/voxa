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
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum, auto


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
        self.tasks: dict[str, VoxaTask] = {}

        self.on_state_changed: Callable[[AssistantState, str], None] | None = None
        self.on_tasks_changed: Callable[[list[VoxaTask]], None] | None = None

    # ------------------------------------------------------------------ state

    def set_state(self, state: AssistantState, detail: str = "") -> None:
        """Record a new assistant state; does nothing if nothing changed."""
        with self._lock:
            if self.state is state and self.detail == detail:
                return
            self.state = state
            self.detail = detail

        callback = self.on_state_changed
        if callback is not None:
            callback(state, detail)

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
