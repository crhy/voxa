"""Tests for voxa.tasks (issue #5 task model)."""

from __future__ import annotations

import pytest

from voxa.tasks import TERMINAL_STATES, TaskState, TaskStore


def test_add_and_list_in_order() -> None:
    store = TaskStore()
    first = store.add_task("Reply to John")
    second = store.add_task("Check backup", detail="nightly")
    assert (first.id, second.id) == (1, 2)
    assert [task.title for task in store.tasks()] == ["Reply to John", "Check backup"]
    assert first.state == TaskState.QUEUED
    assert second.detail == "nightly"


def test_update_clamps_progress_and_clears_input_flag() -> None:
    store = TaskStore()
    task = store.add_task("Updating Spaced Linux")
    store.request_choice(task.id, ("Retry", "Skip"))
    updated = store.update_task(task.id, progress=1.5, state=TaskState.RUNNING)
    assert updated.progress == 1.0
    assert updated.state == TaskState.RUNNING
    assert updated.requires_user_input is False
    assert store.update_task(task.id, progress=-2.0).progress == 0.0


def test_complete_fail_cancel_are_terminal() -> None:
    store = TaskStore()
    done = store.add_task("a")
    failed = store.add_task("b")
    cancelled = store.add_task("c")
    store.add_task("d")
    store.complete_task(done.id, detail="sent")
    store.fail_task(failed.id, detail="boom")
    store.cancel_task(cancelled.id)
    assert store.tasks()[0].detail == "sent"
    assert store.tasks()[1].detail == "boom"
    assert {task.state for task in store.tasks()[:3]} <= TERMINAL_STATES
    assert [task.title for task in store.active_tasks()] == ["d"]


def test_choice_round_trip() -> None:
    store = TaskStore()
    task = store.add_task("Publish release notes")
    store.request_choice(task.id, ["Now", "Later"])
    parked = store.tasks()[0]
    assert parked.state == TaskState.WAITING
    assert parked.requires_user_input is True
    assert parked.choices == ("Now", "Later")
    resumed = store.resolve_choice(task.id, "Later")
    assert resumed.state == TaskState.RUNNING
    assert resumed.detail == "Later"
    assert resumed.requires_user_input is False


def test_choice_errors() -> None:
    store = TaskStore()
    task = store.add_task("a")
    with pytest.raises(ValueError):
        store.request_choice(task.id, [])
    with pytest.raises(ValueError):
        store.resolve_choice(task.id, "nope")
    with pytest.raises(KeyError):
        store.complete_task(999)
