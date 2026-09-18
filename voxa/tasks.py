"""In-memory task model for the task panel (issue #5).

Tasks are the future bridge between agent/tool execution and the UI. The
store is thread-safe: backend threads update it while the UI reads it.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class TaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"  # blocked on a user choice
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED})


@dataclass
class Task:
    id: int
    title: str
    detail: str = ""
    state: TaskState = TaskState.QUEUED
    progress: float | None = None  # 0.0..1.0 when known
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    requires_user_input: bool = False
    choices: tuple[str, ...] = ()


class TaskStore:
    """Thread-safe in-memory task list."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._tasks: dict[int, Task] = {}

    def add_task(self, title: str, detail: str = "") -> Task:
        with self._lock:
            task = Task(id=next(self._ids), title=title, detail=detail)
            self._tasks[task.id] = task
            return task

    def update_task(
        self,
        task_id: int,
        *,
        title: str | None = None,
        detail: str | None = None,
        progress: float | None = None,
        state: TaskState | None = None,
    ) -> Task:
        with self._lock:
            task = self._get(task_id)
            if title is not None:
                task.title = title
            if detail is not None:
                task.detail = detail
            if progress is not None:
                task.progress = min(1.0, max(0.0, progress))
            if state is not None:
                task.state = state
                if state != TaskState.WAITING:
                    task.requires_user_input = False
            return task

    def complete_task(self, task_id: int, detail: str = "") -> Task:
        with self._lock:
            task = self._get(task_id)
            task.state = TaskState.DONE
            task.requires_user_input = False
            if detail:
                task.detail = detail
            return task

    def fail_task(self, task_id: int, detail: str = "") -> Task:
        with self._lock:
            task = self._get(task_id)
            task.state = TaskState.FAILED
            task.requires_user_input = False
            if detail:
                task.detail = detail
            return task

    def cancel_task(self, task_id: int) -> Task:
        with self._lock:
            task = self._get(task_id)
            task.state = TaskState.CANCELLED
            task.requires_user_input = False
            return task

    def request_choice(self, task_id: int, choices: tuple[str, ...] | list[str]) -> Task:
        """Park a task until the user picks one of ``choices``."""
        if not choices:
            raise ValueError("request_choice needs at least one choice.")
        with self._lock:
            task = self._get(task_id)
            task.state = TaskState.WAITING
            task.requires_user_input = True
            task.choices = tuple(choices)
            return task

    def resolve_choice(self, task_id: int, choice: str) -> Task:
        """Answer a pending choice; the task resumes as RUNNING."""
        with self._lock:
            task = self._get(task_id)
            if not task.requires_user_input or choice not in task.choices:
                raise ValueError(f"No pending choice {choice!r} on task {task_id}.")
            task.detail = choice
            task.choices = ()
            task.requires_user_input = False
            task.state = TaskState.RUNNING
            return task

    def tasks(self) -> list[Task]:
        with self._lock:
            return [self._tasks[key] for key in sorted(self._tasks)]

    def active_tasks(self) -> list[Task]:
        return [task for task in self.tasks() if task.state not in TERMINAL_STATES]

    def _get(self, task_id: int) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(f"Unknown task id: {task_id}") from None
