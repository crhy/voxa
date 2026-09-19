from __future__ import annotations

import threading

import pytest

from voxa.ui.state import AssistantModel, AssistantState, TaskState, can_transition


def _collect(model: AssistantModel) -> list[tuple[AssistantState, str]]:
    events: list[tuple[AssistantState, str]] = []
    model.on_state_changed = lambda state, detail: events.append((state, detail))
    return events


#: The transition table as documented in voxa/ui/state.py, written out here by
#: hand so the test checks the policy instead of echoing ALLOWED_TRANSITIONS.
#: Each set lists every state reachable from the key, including staying put.
EXPECTED_TRANSITIONS: dict[AssistantState, frozenset[AssistantState]] = {
    AssistantState.OFFLINE: {AssistantState.OFFLINE, AssistantState.READY, AssistantState.ERROR},
    AssistantState.READY: {
        AssistantState.READY,
        AssistantState.LISTENING,
        AssistantState.THINKING,
        AssistantState.WORKING,
        AssistantState.OFFLINE,
        AssistantState.ERROR,
    },
    AssistantState.LISTENING: {
        AssistantState.LISTENING,
        AssistantState.THINKING,
        AssistantState.READY,
        AssistantState.OFFLINE,
        AssistantState.ERROR,
    },
    AssistantState.THINKING: {
        AssistantState.THINKING,
        AssistantState.SPEAKING,
        AssistantState.READY,
        AssistantState.WORKING,
        AssistantState.WAITING,
        AssistantState.OFFLINE,
        AssistantState.ERROR,
    },
    AssistantState.SPEAKING: {
        AssistantState.SPEAKING,
        AssistantState.READY,
        AssistantState.LISTENING,
        AssistantState.THINKING,
        AssistantState.OFFLINE,
        AssistantState.ERROR,
    },
    AssistantState.WORKING: {
        AssistantState.WORKING,
        AssistantState.WAITING,
        AssistantState.READY,
        AssistantState.THINKING,
        AssistantState.SPEAKING,
        AssistantState.OFFLINE,
        AssistantState.ERROR,
    },
    AssistantState.WAITING: {
        AssistantState.WAITING,
        AssistantState.WORKING,
        AssistantState.READY,
        AssistantState.OFFLINE,
        AssistantState.ERROR,
    },
    AssistantState.ERROR: {
        AssistantState.ERROR,
        AssistantState.READY,
        AssistantState.OFFLINE,
    },
}

#: A legal walk from OFFLINE to each state, so tests never start from an
#: unreachable state.
REACHABLE: dict[AssistantState, tuple[AssistantState, ...]] = {
    AssistantState.OFFLINE: (),
    AssistantState.READY: (AssistantState.READY,),
    AssistantState.LISTENING: (AssistantState.READY, AssistantState.LISTENING),
    AssistantState.THINKING: (AssistantState.READY, AssistantState.THINKING),
    AssistantState.SPEAKING: (AssistantState.READY, AssistantState.THINKING, AssistantState.SPEAKING),
    AssistantState.WORKING: (AssistantState.READY, AssistantState.WORKING),
    AssistantState.WAITING: (AssistantState.READY, AssistantState.THINKING, AssistantState.WAITING),
    AssistantState.ERROR: (AssistantState.READY, AssistantState.ERROR),
}


def _reach(model: AssistantModel, state: AssistantState) -> None:
    """Put the model in ``state`` by following only allowed transitions."""
    for step in REACHABLE[state]:
        assert model.set_state(step, f"reach {step.name}")
    assert model.state is state


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
        _reach(model, state)
        assert model.set_state(AssistantState.OFFLINE, "disabled")
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
        if state is AssistantState.THINKING and AssistantState.WORKING not in seen:
            task = model.add_task("Ask model", detail="queued inference")
            model.start_task(task.id, detail="running")
            model.set_state(AssistantState.WORKING, "task running")

    model.on_state_changed = on_state_changed
    model.set_state(AssistantState.READY, "activated")
    model.set_state(AssistantState.THINKING, "thinking")

    assert seen == [AssistantState.READY, AssistantState.THINKING, AssistantState.WORKING]
    assert model.state is AssistantState.WORKING
    assert len(model.active_tasks()) == 1


def test_concurrent_state_updates_are_serialised() -> None:
    model = AssistantModel()
    model.set_state(AssistantState.READY, "activated")
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


@pytest.mark.parametrize("current", list(AssistantState))
@pytest.mark.parametrize("new", list(AssistantState))
def test_can_transition_matches_documented_table(current: AssistantState, new: AssistantState) -> None:
    assert can_transition(current, new) is (new in EXPECTED_TRANSITIONS[current])


@pytest.mark.parametrize("current", list(AssistantState))
def test_every_state_may_go_offline_or_error(current: AssistantState) -> None:
    assert can_transition(current, AssistantState.OFFLINE)
    assert can_transition(current, AssistantState.ERROR)

    model = AssistantModel()
    events = _collect(model)
    _reach(model, current)
    assert model.set_state(AssistantState.OFFLINE, "shutting down")
    assert model.state is AssistantState.OFFLINE
    assert events[-1] == (AssistantState.OFFLINE, "shutting down")
    assert model.set_state(AssistantState.ERROR, "failed")
    assert model.state is AssistantState.ERROR
    assert events[-1] == (AssistantState.ERROR, "failed")


@pytest.mark.parametrize("current", (AssistantState.OFFLINE, AssistantState.ERROR))
def test_offline_and_error_cannot_listen_directly(current: AssistantState) -> None:
    assert not can_transition(current, AssistantState.LISTENING)

    model = AssistantModel()
    events = _collect(model)
    _reach(model, current)
    assert not model.set_state(AssistantState.LISTENING, "wake word")
    assert model.state is current
    assert not any(state is AssistantState.LISTENING for state, _ in events)


def test_speaking_may_barge_in_to_listening() -> None:
    assert can_transition(AssistantState.SPEAKING, AssistantState.LISTENING)

    model = AssistantModel()
    events = _collect(model)
    _reach(model, AssistantState.SPEAKING)
    assert model.set_state(AssistantState.LISTENING, "user interrupted")
    assert model.state is AssistantState.LISTENING
    assert events[-1] == (AssistantState.LISTENING, "user interrupted")


def test_rejected_transition_returns_false_and_fires_no_callback() -> None:
    model = AssistantModel()
    events = _collect(model)

    assert not model.set_state(AssistantState.WORKING, "never started")
    assert model.state is AssistantState.OFFLINE
    assert model.detail == ""
    assert not events

    _reach(model, AssistantState.LISTENING)
    assert not model.set_state(AssistantState.WAITING, "nothing to wait for")
    assert model.state is AssistantState.LISTENING
    assert model.detail == "reach LISTENING"
    assert len(events) == len(REACHABLE[AssistantState.LISTENING])


def test_generation_starts_at_zero_and_bumps_on_activation() -> None:
    model = AssistantModel()
    assert model.generation == 0
    assert model.bump_generation() == 1
    assert model.generation == 1
    assert model.bump_generation() == 2


def test_stale_generation_is_rejected_and_current_one_applied() -> None:
    model = AssistantModel()
    events = _collect(model)

    token = model.bump_generation()
    assert model.set_state(AssistantState.READY, "activated", generation=token)

    model.bump_generation()  # a new activation invalidates the old token
    assert not model.set_state(AssistantState.LISTENING, "late wake word", generation=token)
    assert model.state is AssistantState.READY
    assert model.detail == "activated"
    assert events == [(AssistantState.READY, "activated")]

    fresh = model.generation
    assert model.set_state(AssistantState.LISTENING, "wake word", generation=fresh)
    assert model.state is AssistantState.LISTENING
    assert events[-1] == (AssistantState.LISTENING, "wake word")


def test_stale_generation_cannot_revive_an_offline_assistant() -> None:
    model = AssistantModel()
    events = _collect(model)

    token = model.bump_generation()
    model.set_state(AssistantState.READY, "activated", generation=token)
    model.set_state(AssistantState.OFFLINE, "offline", generation=token)
    new_token = model.bump_generation()

    assert not model.set_state(AssistantState.THINKING, "late result", generation=token)
    assert model.state is AssistantState.OFFLINE
    assert not any(state is AssistantState.THINKING for state, _ in events)
    assert model.set_state(AssistantState.READY, "back online", generation=new_token)
    assert model.state is AssistantState.READY


def test_bump_generation_is_thread_safe() -> None:
    model = AssistantModel()
    barrier = threading.Barrier(2)
    results: list[int] = []

    def worker() -> None:
        barrier.wait()
        for _ in range(1000):
            results.append(model.bump_generation())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert model.generation == 2000
    assert len(results) == 2000
    assert sorted(results) == list(range(1, 2001))
