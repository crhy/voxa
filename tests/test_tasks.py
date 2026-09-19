from __future__ import annotations

import time
from functools import partial

from voxa.ui.state import TERMINAL_TASK_STATES, AssistantModel, TaskState, VoxaTask


def _collect_tasks(model: AssistantModel) -> list[list[VoxaTask]]:
    events: list[list[VoxaTask]] = []
    model.on_tasks_changed = lambda tasks: events.append(tasks)
    return events


def test_add_start_update_complete_lifecycle() -> None:
    model = AssistantModel()
    task = model.add_task("Update website", detail="finding project")

    assert task.state is TaskState.QUEUED
    assert task.progress is None
    assert task.result == ""
    assert task.error == ""
    assert task.created_at <= task.updated_at

    model.start_task(task.id, detail="editing index.html")
    assert task.state is TaskState.RUNNING
    assert task.detail == "editing index.html"

    model.update_task(task.id, detail="running checks", progress=0.5)
    assert task.detail == "running checks"
    assert task.progress == 0.5

    model.complete_task(task.id, result="website updated")
    assert task.state is TaskState.DONE
    assert task.progress == 1.0
    assert task.result == "website updated"


def test_fail_and_cancel_lifecycle() -> None:
    model = AssistantModel()
    failed = model.add_task("Install Blender")
    cancelled = model.add_task("Organise Downloads")

    model.start_task(failed.id)
    model.fail_task(failed.id, error="no permission to install packages")
    assert failed.state is TaskState.FAILED
    assert failed.error == "no permission to install packages"

    model.cancel_task(cancelled.id)
    assert cancelled.state is TaskState.CANCELLED


def test_updated_at_refreshes_on_every_change() -> None:
    model = AssistantModel()
    task = model.add_task("Check backup")
    first = task.updated_at

    time.sleep(0.01)
    model.start_task(task.id)
    model.update_task(task.id, progress=0.25)
    model.complete_task(task.id, result="ok")

    assert task.updated_at > first
    assert task.created_at <= task.updated_at


def test_progress_is_clamped_to_zero_through_one() -> None:
    model = AssistantModel()
    task = model.add_task("Organise Downloads")

    model.update_task(task.id, progress=-3.0)
    assert task.progress == 0.0

    model.update_task(task.id, progress=12.5)
    assert task.progress == 1.0


def test_request_choice_waits_for_user_input() -> None:
    model = AssistantModel()
    task = model.add_task("Find ISO", detail="three matches")
    model.start_task(task.id)

    model.request_choice(task.id, ["newest ISO", "release ISO", "cancel"])

    assert task.state is TaskState.WAITING
    assert task.requires_user_input is True
    assert task.choices == ["newest ISO", "release ISO", "cancel"]
    assert task in model.active_tasks()


def test_resolve_choice_returns_choice_and_resumes_task() -> None:
    model = AssistantModel()
    task = model.add_task("Find ISO")
    model.start_task(task.id)
    model.request_choice(task.id, ["newest ISO", "release ISO"])

    chosen = model.resolve_choice(task.id, "newest ISO")

    assert chosen == "newest ISO"
    assert task.choices == []
    assert task.state is TaskState.RUNNING
    assert task.requires_user_input is False


def test_resolve_choice_rejects_unknown_choice() -> None:
    model = AssistantModel()
    task = model.add_task("Find ISO")
    model.request_choice(task.id, ["newest ISO"])

    try:
        model.resolve_choice(task.id, "release ISO")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown choice should raise ValueError")

    assert task.state is TaskState.WAITING
    assert task.choices == ["newest ISO"]


def _reach_terminal(model: AssistantModel, task_id: str, terminal: TaskState) -> None:
    if terminal is TaskState.DONE:
        model.complete_task(task_id)
    elif terminal is TaskState.FAILED:
        model.fail_task(task_id, error="boom")
    else:
        model.cancel_task(task_id)


def test_terminal_tasks_reject_further_changes() -> None:
    for terminal in TERMINAL_TASK_STATES:
        model = AssistantModel()
        task = model.add_task("Finished thing")
        _reach_terminal(model, task.id, terminal)

        attempts = [
            partial(model.start_task, task.id),
            partial(model.update_task, task.id, state=TaskState.RUNNING),
            partial(model.request_choice, task.id, ["ok"]),
            partial(model.resolve_choice, task.id, "ok"),
        ]
        if terminal is not TaskState.DONE:
            attempts.append(partial(model.complete_task, task.id))
        if terminal is not TaskState.FAILED:
            attempts.append(partial(model.fail_task, task.id, error="too late"))
        if terminal is not TaskState.CANCELLED:
            attempts.append(partial(model.cancel_task, task.id))

        for attempt in attempts:
            try:
                attempt()
            except ValueError:
                continue
            raise AssertionError(f"terminal state {terminal.name} should reject a change")

        assert task.state is terminal

        model.update_task(task.id, detail="final note")
        assert task.detail == "final note"
        model.update_task(task.id, state=terminal)
        assert task.state is terminal


def test_active_and_todo_tasks_partition_in_creation_order() -> None:
    model = AssistantModel()
    queued_one = model.add_task("Publish release")
    queued_two = model.add_task("Review website")
    running = model.add_task("Build interface")
    waiting = model.add_task("Install VLC")
    finished = model.add_task("Check backup")
    stopped = model.add_task("Old migration")

    model.start_task(running.id, detail="running tests")
    model.start_task(waiting.id)
    model.request_choice(waiting.id, ["Continue", "Cancel"])
    model.complete_task(finished.id, result="backup verified")
    model.cancel_task(stopped.id)

    assert model.active_tasks() == [running, waiting]
    assert model.todo_tasks() == [queued_one, queued_two]
    assert len(model.active_tasks()) + len(model.todo_tasks()) + 2 == len(model.tasks)


def test_tasks_callback_receives_a_copy_of_the_list() -> None:
    model = AssistantModel()
    events = _collect_tasks(model)

    first = model.add_task("Reply to John")
    model.start_task(first.id)
    model.update_task(first.id, progress=0.1)
    model.complete_task(first.id, result="sent")
    model.add_task("Check backup")

    assert len(events) == 5
    snapshots = [list(snapshot) for snapshot in events]
    for snapshot in events:
        assert isinstance(snapshot, list)
        snapshot.append(object())
        snapshot.clear()

    assert len(model.tasks) == 2
    assert len(snapshots[-1]) == 2


def test_unknown_task_id_raises_key_error() -> None:
    model = AssistantModel()

    for attempt in (
        lambda: model.update_task("missing", detail="x"),
        lambda: model.start_task("missing"),
        lambda: model.complete_task("missing"),
        lambda: model.fail_task("missing", error="boom"),
        lambda: model.cancel_task("missing"),
        lambda: model.request_choice("missing", ["yes"]),
        lambda: model.resolve_choice("missing", "yes"),
    ):
        try:
            attempt()
        except KeyError:
            continue
        raise AssertionError("unknown task id should raise KeyError")


def test_snapshots_are_independent_copies_of_the_tasks() -> None:
    model = AssistantModel()
    task = model.add_task("Organize Downloads")
    model.start_task(task.id)
    model.request_choice(task.id, ["a", "b"])

    seen: list[list] = []
    model.on_tasks_changed = seen.append
    model.update_task(task.id, detail="37 / 112 files")

    snapshot = seen[-1][0]
    snapshot.detail = "tampered"
    snapshot.choices.append("c")
    assert model.tasks[task.id].detail == "37 / 112 files"
    assert model.tasks[task.id].choices == ["a", "b"]

    active = model.active_tasks()[0]
    active.detail = "tampered again"
    assert model.tasks[task.id].detail == "37 / 112 files"
