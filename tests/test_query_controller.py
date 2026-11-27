"""The query controller alone (#138 part 1, crhy/voxa#138).

``QueryController`` runs the whole AI request with nothing but callbacks:
these tests drive it with a fake backend, fake hooks and an injected clock -
no GTK, no display, no real AI server anywhere. The window is only the glue
that wires its hooks up (covered by tests/test_window_tools.py); what the
controller itself must do - stream, wait, restart, apologize, record,
respect the generation tag - is checked here.
"""

from __future__ import annotations

import threading
import time

from voxa.backend import OUT_OF_MEMORY, BackendError
from voxa.conversation import ConversationHistory
from voxa.query_controller import FALSE_CLAIM_REPLY, QueryController
from voxa.ui.state import AssistantState

AI_OUT_OF_MEMORY = "The AI server ran out of memory while it was answering."


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeHooks:
    """Every hook the controller may reach, recorded for the assertions."""

    def __init__(self, generation: int = 1) -> None:
        self.calls: list[tuple] = []
        self.generation = generation
        self.web_enabled_flag = False
        self.managed = False
        self.conversation = False
        self.restarted = 0
        self._ticker_sources = iter(range(1, 1000))

    # marshalling: the test IS the UI thread, so callbacks run at once
    def schedule(self, fn, *args):
        return fn(*args)

    def current(self, generation, cancel_event):
        return generation == self.generation

    def query_generation(self):
        return self.generation

    def current_cancel(self):
        return threading.Event()

    def new_query(self, cancel_previous=True):
        self.generation += 1
        return self.generation, threading.Event()

    def status(self, text):
        self.calls.append(("status", text))

    def status_busy(self, text):
        self.calls.append(("status_busy", text))

    def toast(self, text):
        self.calls.append(("toast", text))

    def assistant_state(self, state, detail):
        self.calls.append(("state", state, detail))

    def idle_status(self):
        return "Ready."

    def set_busy(self, sensitive):
        self.calls.append(("set_busy", sensitive))

    def show_reply(self, text):
        self.calls.append(("show_reply", text))

    def show_answer(self, text):
        self.calls.append(("answer", text))

    def append_batch(self, batch, generation, cancel_event):
        self.calls.append(("batch", batch))

    def task_end(self, *args):
        self.calls.append(("task_end", *args))

    def assistant_active(self):
        return True

    def reply_finished(self, waiting):
        self.calls.append(("reply_finished", waiting))

    def fail_assistant(self, reason):
        self.calls.append(("fail", reason))

    def conversation_active(self):
        return self.conversation

    def waiting_for_prompt(self):
        return False

    def speak(self, text):
        self.calls.append(("speak", text))

    def early_speech_finish(self, spoken, generation):
        return False

    def reset_early_speech(self):
        self.calls.append(("reset_early",))

    def auto_speak_enabled(self):
        return False

    def auto_speak(self):
        self.calls.append(("auto_speak",))

    def log_action(self, *args, **kwargs):
        self.calls.append(("log", args, kwargs))

    def cmd_timings(self, when):
        return {}

    def tool_ran(self):
        return False

    def web_enabled(self):
        return self.web_enabled_flag

    def web_mode(self):
        return "auto"

    def web_search_started(self, query, generation, cancel_event):
        self.calls.append(("web", query))

    def web_unavailable(self, generation, cancel_event):
        self.calls.append(("web_unavailable",))

    def ticker_start(self, callback):
        return next(self._ticker_sources)

    def ticker_stop(self, source):
        self.calls.append(("ticker_stop", source))

    def managed_backend(self):
        return self.managed

    def restart_backend(self):
        self.restarted += 1
        self.calls.append(("restart",))

    def models_refreshed(self):
        self.calls.append(("models",))

    def backend_label(self):
        return "Ollama"

    def chat_model(self):
        return "m"

    def tools(self):
        return None

    def browser_step(self, detail, generation, cancel_event):
        self.calls.append(("browser_step", detail))

    def open_url(self, url, original):
        self.calls.append(("open_url", url, original))


class FakeClient:
    """One backend: streams chunks, then returns the whole answer."""

    def __init__(self, chunks=(), answer="the answer", loaded=True, busy=False) -> None:
        self.chunks = list(chunks)
        self.answer = answer
        self.loaded = loaded
        self.busy = busy

    def is_busy(self) -> bool:
        return self.busy

    def is_model_loaded(self, model: str = ""):
        return self.loaded

    def generate_stream(self, *, model, prompt, cancel_event, on_chunk, messages=None) -> str:
        for chunk in self.chunks:
            on_chunk(chunk)
        return self.answer


class ExplodingClient(FakeClient):
    def generate_stream(self, **kwargs):
        raise BackendError("ollama died", kind=OUT_OF_MEMORY)


class RestartingClient(FakeClient):
    """A managed backend that was stopped under us and answers after a restart."""

    def __init__(self) -> None:
        super().__init__(answer="back on")
        self.checks = 0

    def is_model_loaded(self, model: str = ""):
        self.checks += 1
        # the first probe finds it stopped; after the controller restarts it, it is back
        return None if self.checks == 1 else True


def make_controller(hooks, client, clock=None, history=None) -> QueryController:
    return QueryController(
        backend=lambda: client,
        ui=hooks,
        clock=clock or (lambda: 0.0),
        history=history,
    )


def ask(controller, hooks, generation=1, cancel_event=None, prompt="what is the capital of France"):
    controller.ask(
        model="m",
        prompt=prompt,
        messages=[],
        heard=prompt,
        generation=generation,
        cancel_event=cancel_event or threading.Event(),
    )


def test_a_heard_question_streams_its_answer_and_reports_done_once() -> None:
    hooks = FakeHooks(generation=3)
    history = ConversationHistory("system", 8)
    controller = make_controller(hooks, FakeClient(chunks=["th", "e answer"]), history=history)
    ask(controller, hooks, generation=3)
    assert _wait_for(lambda: ("task_end", "done") in hooks.calls)
    assert ("batch", "the answer") in hooks.calls  # 2 chunks, one batch
    assert ("answer", "the answer") in hooks.calls
    assert sum(call[0] == "task_end" for call in hooks.calls) == 1
    assert any(m["content"] == "the answer" for m in history.messages())  # the turn is kept
    # the pre-call record AND the final record with this request's speech
    logs = [call for call in hooks.calls if call[0] == "log"]
    assert any(call[2].get("speech") == "the answer" for call in logs)


def test_a_cancelled_request_drops_the_unanswered_turn_and_says_stopped() -> None:
    hooks = FakeHooks(generation=2)
    history = ConversationHistory("system", 8)
    history.add_user("what is the capital of France")
    controller = make_controller(hooks, FakeClient(answer="late"), history=history)
    controller.pending_user_generation = 2
    cancel = threading.Event()
    cancel.set()  # the request was retired while its worker was still streaming
    ask(controller, hooks, generation=2, cancel_event=cancel)
    _wait_for(lambda: ("task_end", "cancelled") in hooks.calls)
    # the heard question leaves the model thread too
    assert all(m["content"] != "what is the capital of France" for m in history.messages())
    assert controller.pending_user_generation is None
    assert ("status", "AI request stopped.") in hooks.calls


def test_a_backend_failure_is_felt_out_loud_with_the_kinds_sentence() -> None:
    hooks = FakeHooks(generation=1)
    hooks.conversation = True
    controller = make_controller(hooks, ExplodingClient())
    ask(controller, hooks)
    assert _wait_for(lambda: any(c[0] == "speak" for c in hooks.calls))
    spoken = [c for c in hooks.calls if c[0] == "speak"][0][1]
    assert "Sorry, I could not get an answer" in spoken
    assert AI_OUT_OF_MEMORY in spoken  # the kind's sentence rides along (#138)


def test_a_stopped_managed_backend_is_restarted_and_waited_for_with_the_injected_clock() -> None:
    hooks = FakeHooks(generation=1)
    hooks.managed = True
    client = RestartingClient()
    controller = make_controller(hooks, client)
    # the injected clock never moves: the bounded wait is the controller's own
    ask(controller, hooks)
    assert _wait_for(lambda: ("answer", "back on") in hooks.calls)
    assert hooks.restarted == 1
    assert ("state", AssistantState.WORKING, "Starting Ollama…") in hooks.calls
    assert ("models",) in hooks.calls  # the picker stops saying "not running"


def test_a_cold_model_shows_the_loading_note_and_its_ticker_is_stopped_at_the_answer() -> None:
    hooks = FakeHooks(generation=1)
    controller = make_controller(hooks, FakeClient(loaded=False))
    ask(controller, hooks)
    assert _wait_for(lambda: ("answer", "the answer") in hooks.calls)
    assert ("status_busy", "Loading m… 0s") in hooks.calls
    # the one-second note stops the moment the answer is in
    assert ("ticker_stop", 1) in hooks.calls
    assert controller.loading_model_since is None


def test_a_stale_completion_reports_nothing_at_all() -> None:
    hooks = FakeHooks(generation=9)
    controller = make_controller(hooks, FakeClient())
    assert controller._on_query_finished("old", 1, threading.Event()) is False
    assert controller._on_query_error("old", 1, threading.Event()) is False
    assert hooks.calls == []


def test_a_reply_that_claims_an_action_that_never_ran_answers_honestly() -> None:
    hooks = FakeHooks(generation=1)
    controller = make_controller(hooks, FakeClient(answer="Done. Brutal Chess is closed."))
    ask(controller, hooks)
    assert _wait_for(lambda: ("answer", FALSE_CLAIM_REPLY) in hooks.calls)
    assert any(call[0] == "log" and call[1][1] == "false_claim" for call in hooks.calls)


def test_a_server_that_is_busy_with_a_agent_says_so_before_the_answer() -> None:
    hooks = FakeHooks(generation=1)
    controller = make_controller(hooks, FakeClient(busy=True))
    ask(controller, hooks)
    assert _wait_for(lambda: ("answer", "the answer") in hooks.calls)
    assert any("busy with another request" in str(call) for call in hooks.calls)
