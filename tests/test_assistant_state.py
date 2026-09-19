from __future__ import annotations

import threading

from voxa.ui.state import AssistantModel, AssistantState, TaskState


def _collect(model: AssistantModel) -> list[tuple[AssistantState, str]]:
    events: list[tuple[AssistantState, str]] = []
    model.on_state_changed = lambda state, detail: events.append((state, detail))
    return events


def test_default_state_is_offline() -> None:
    model = AssistantModel()
    assert model.state is AssistantState.OFFLINE
    assert model.detail == ""


def test_assistant_state_has_waiting_after_working() -> None:
    names = [state.name for state in AssistantState]
    assert names == [
        "OFFLINE",
        "READY",
        "LISTENING",
        "THINKING",
        "SPEAKING",
        "WORKING",
        "WAITING",
        "ERROR",
    ]


def test_state_sequence_calls_callback_in_order_with_details() -> None:
    model = AssistantModel()
    events = _collect(model)

    for state, detail in (
        (AssistantState.READY, "Ready"),
        (AssistantState.LISTENING, "Listening for the wake word"),
        (AssistantState.THINKING, "Asking local model"),
        (AssistantState.SPEAKING, "Reading response"),
        (AssistantState.READY, ""),
    ):
        model.set_state(state, detail)

    assert events == [
        (AssistantState.READY, "Ready"),
        (AssistantState.LISTENING, "Listening for the wake word"),
        (AssistantState.THINKING, "Asking local model"),
        (AssistantState.SPEAKING, "Reading response"),
        (AssistantState.READY, ""),
    ]
    assert model.state is AssistantState.READY
    assert model.detail == ""


def test_offline_can_be_entered_from_every_other_state() -> None:
    for state in AssistantState:
        if state is AssistantState.OFFLINE:
            continue
        model = AssistantModel()
        events = _collect(model)
        model.set_state(state, "busy")
        model.set_state(AssistantState.OFFLINE, "disabled")
        assert model.state is AssistantState.OFFLINE
        assert events[-1] == (AssistantState.OFFLINE, "disabled")


def test_repeated_identical_set_state_does_not_fire_callback() -> None:
    model = AssistantModel()
    events = _collect(model)

    model.set_state(AssistantState.READY, "Ready")
    model.set_state(AssistantState.READY, "Ready")
    model.set_state(AssistantState.READY, "Ready")

    assert events == [(AssistantState.READY, "Ready")]

    model.set_state(AssistantState.READY, "Ready, idle")
    assert events[-1] == (AssistantState.READY, "Ready, idle")


def test_callback_can_call_back_into_the_model_without_deadlocking() -> None:
    model = AssistantModel()
    seen: list[AssistantState] = []

    def on_state_changed(state: AssistantState, detail: str) -> None:
        seen.append(state)
        if state is AssistantState.THINKING and len(seen) == 1:
            task = model.add_task("Ask model", detail="queued inference")
            model.start_task(task.id, detail="running")
            model.set_state(AssistantState.WORKING, "task running")

    model.on_state_changed = on_state_changed
    model.set_state(AssistantState.THINKING, "thinking")

    assert seen == [AssistantState.THINKING, AssistantState.WORKING]
    assert model.state is AssistantState.WORKING
    assert len(model.active_tasks()) == 1


def test_concurrent_state_updates_are_serialised() -> None:
    model = AssistantModel()
    barrier = threading.Barrier(8)
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            barrier.wait()
            for _ in range(50):
                model.set_state(AssistantState.WORKING, f"worker {index}")
                task = model.add_task(f"task {index}")
                model.complete_task(task.id, result="done")
        except Exception as exc:  # pragma: no cover - only if locking is broken
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(model.tasks) == 400
    assert all(task.state is TaskState.DONE for task in model.tasks.values())
