"""The AI query pipeline, outside the window (crhy/voxa#138 part 1).

The rule: UI -> controller -> services. MainWindow keeps the GTK glue - the
status line, the model picker, the toasts, the buttons, the response buffer -
and drives every model request through this controller. The controller owns
the request itself: it runs the bounded worker thread, waits for and restarts
a managed backend that was stopped under us, batches the streamed answer,
writes the action record every heard request must leave behind, and reports
finished/error/cancelled through callbacks tagged with the generation the
request started under. It never imports Gtk, gi or voxa.window.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from . import websearch
from .agent import intents, planner
from .agent.claims import claims_action, first_sentences
from .agent.result import ToolResult
from .agent.tools.web import get_session
from .backend import BackendError
from .conversation import ConversationHistory
from .ollama import OllamaError, strip_reasoning
from .spellings import quote as spelling_quote
from .ui.state import AssistantState

# A heard question is worth waiting briefly for a stopped managed backend
# (ollama serve / llama-server) to come back before the request fails.
AI_SERVER_WAIT_SECONDS = 30.0
AI_SERVER_WAIT_DELAY = 0.5
# Hands-free users cannot see the toast, so a failed request must also say why out loud.
AI_APOLOGY = "Sorry, I could not get an answer from the model."
FALSE_CLAIM_REPLY = (
    "I couldn't do that. Try saying it as a direct command, like “close Brutal Chess”."
)
# Batch streamed chunks so the main loop schedules at most one idle callback
# per batch instead of one per token chunk.
CHUNK_BATCH_SIZE = 8
# How long one heard question may still reuse the last web search.
SEARCH_REUSE_SECONDS = 180.0
# "Open the website" after a web search: the site of the last searched URL, remembered this long.
WEBSITE_MEMORY_SECONDS = 600.0


def site_of_url(url: str) -> str:
    """Scheme and host only: https://spacedlinux.com/posts/x -> https://spacedlinux.com."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    return f"{scheme}{sep}{rest.split('/', 1)[0]}"


def website_fresh(stored_at: float, now: float) -> bool:
    """True while a website URL stored at stored_at is still remembered at now."""
    return now - stored_at <= WEBSITE_MEMORY_SECONDS


def short_model_name(model: str) -> str:
    """The part after the last '/', truncated to 28 characters with an ellipsis."""
    name = model.rsplit("/", 1)[-1]
    if len(name) > 28:
        return name[:27] + "…"
    return name


@dataclass(frozen=True)
class QueryHooks:
    """What MainWindow hands the controller: the glue the controller must not do itself.

    The completion notes (status/toast/button/buffer/picker) run on the UI thread;
    the worker reaches them through ``schedule`` exactly like ``idle`` did.
    """

    # marshalling and the query-generation registry (which OFFLINE and the other
    # request paths share, so MainWindow owns it)
    schedule: Callable[..., None]
    current: Callable[[int, threading.Event], bool]
    query_generation: Callable[[], int]
    current_cancel: Callable[[], threading.Event]
    new_query: Callable[[bool], tuple[int, threading.Event]]
    # status line, toasts, the assistant state model
    status: Callable[[str], None]
    status_busy: Callable[[str], None]
    toast: Callable[[str], None]
    assistant_state: Callable[[AssistantState, str], None]
    idle_status: Callable[[], str]
    # the legacy UI's query controls and the response buffer
    set_busy: Callable[[bool], None]
    show_reply: Callable[[str], None]
    show_answer: Callable[[str], None]
    append_batch: Callable[[str, int, threading.Event], None]
    # the task card and the assistant controller
    task_end: Callable[[str, str], None]
    assistant_active: Callable[[], bool]
    reply_finished: Callable[[bool], None]
    fail_assistant: Callable[[str], None]
    # speech-facing callbacks: how an answer or an apology reaches the ear
    conversation_active: Callable[[], bool]
    waiting_for_prompt: Callable[[], bool]
    speak: Callable[[str], None]
    early_speech_finish: Callable[[str, int], bool]
    reset_early_speech: Callable[[], None]
    auto_speak_enabled: Callable[[], bool]
    auto_speak: Callable[[], None]
    # the action log the request must leave a record in
    log_action: Callable[..., None]
    cmd_timings: Callable[[float | None], dict[str, int]]
    tool_ran: Callable[[], bool]
    # web search augmentation and the model-loading note's one-second ticker
    web_enabled: Callable[[], bool]
    web_mode: Callable[[], str]
    web_search_started: Callable[[str, int, threading.Event], None]
    web_unavailable: Callable[[int, threading.Event], None]
    ticker_start: Callable[[Callable[[], bool]], int]
    ticker_stop: Callable[[int], None]
    # a managed backend stopped under us: restart it, then refill the picker
    managed_backend: Callable[[], bool]
    restart_backend: Callable[[], None]
    models_refreshed: Callable[[], None]
    backend_label: Callable[[], str]
    # the page-aware browser task: the chat model, the tool registry, the step caption
    chat_model: Callable[[], str]
    tools: Callable[[], object]
    browser_step: Callable[[str, int, threading.Event], None]
    # opening the site of the last searched web result
    open_url: Callable[[str, str], None]


class QueryController:
    """Runs one AI query start to finish: wait/restart, stream, record, report."""

    def __init__(
        self,
        backend: Callable[[], object],
        ui: QueryHooks,
        clock: Callable[[], float] = time.monotonic,
        history: ConversationHistory | None = None,
    ) -> None:
        self._backend = backend
        self._ui = ui
        self._clock = clock
        self._history = history
        # The conversation turn whose user message has not been answered yet, so
        # a stale finish/error callback cannot untangle a newer turn.
        self.pending_user_generation: int | None = None
        # Web search bookkeeping: what the last request searched for, what it
        # found, and the site it should not have to search for again.
        self.last_search_query: str | None = None
        self.last_search_at = 0.0
        self.search_record: dict | None = None
        self.search_source = ""
        self.search_results: list = []
        self.last_site_url: tuple[float, str] | None = None
        # heard/route/monotonic-start of the model request whose pre-call action
        # record was written, so its final record can carry ok/speech/ms.
        self._action_pending: tuple[str, str, float, int] | None = None
        # The "Loading tiny… 3s" note while a cold model warms up.
        self._loading_model_since: float | None = None
        self._loading_model_name: str = ""
        self._loading_source: int = 0
        self._loading_generation: int = -1
        self._loading_cancel: threading.Event | None = None

    # ---- the heard question itself -------------------------------------------------

    def ask(self, *, model: str, prompt: str, messages: list[dict[str, str]],
            heard: str, generation: int, cancel_event: threading.Event) -> None:
        """One heard question: keep one bounded thread for it and report at the end.

        Typed questions and spoken conversation use the same model, so
        follow-ups must see the same earlier turns regardless of how the user
        entered them.
        """
        self.search_record = None
        self.search_source = ""
        self.search_results = []

        pending_chunks: list[str] = []

        def flush_chunks() -> None:
            if pending_chunks:
                batch = "".join(pending_chunks)
                pending_chunks.clear()
                self._ui.schedule(self._ui.append_batch, batch, generation, cancel_event)

        def on_chunk(chunk: str) -> None:
            pending_chunks.append(chunk)
            if len(pending_chunks) >= CHUNK_BATCH_SIZE:
                flush_chunks()

        def worker() -> None:
            try:
                client = self._backend()
                nonlocal messages, prompt
                # Web search only augments the question of a real AI backend:
                # the contract marks the real ones with capabilities(); a
                # client that predates the contract gets the question as
                # asked, exactly like the old isinstance guard did (#138).
                if self._ui.web_enabled() and hasattr(client, "capabilities"):
                    context = self.web_context(prompt, generation, cancel_event)
                    if context:
                        if messages:
                            messages = [*messages[:-1], {**messages[-1], "content": f"{context}\n\n{messages[-1]['content']}"}]
                        else:
                            prompt = f"{context}\n\nQuestion: {prompt}"
                rec = self.search_record
                if rec is not None:
                    route = rec["route"]
                    self._ui.log_action(heard, route, args=rec["args"], detail=rec["detail"], timings=self._ui.cmd_timings(None))
                else:
                    route = "model"
                    self._ui.log_action(heard, route, timings=self._ui.cmd_timings(None))
                # Behavior differs: only llama.cpp/Strata servers serve one request
                # at a time, so a heard question can queue behind a coding agent;
                # Ollama answers both at once and must not pay for that probe.
                if hasattr(client, "is_busy") and client.is_busy():
                    self._ui.schedule(self._on_server_busy, generation, cancel_event)
                loaded = client.is_model_loaded(model) if hasattr(client, "is_model_loaded") else None
                if loaded is False:
                    # A reachable backend that has not got the model in memory: say so
                    # ("Loading tiny… 3s") while the first request warms it up.
                    self._ui.schedule(self._on_model_loading, model, generation, cancel_event)
                # Behavior differs: only the backends Voxa manages itself can have
                # been stopped under us; a heard question is worth restarting those
                # and waiting briefly for them instead of silently answering with
                # nothing. (Strata is the user's server: Voxa never starts it.)
                if loaded is None and hasattr(client, "is_model_loaded") and self._ui.managed_backend():
                    # The managed backend was stopped (or never started) under us: a
                    # heard question is worth restarting it and waiting briefly for
                    # it instead of silently answering with nothing.
                    self._ui.schedule(self._on_ai_server_starting, generation, cancel_event)
                    self._ui.restart_backend()
                    deadline = self._clock() + AI_SERVER_WAIT_SECONDS
                    while (
                        self._clock() < deadline
                        and self._ui.current(generation, cancel_event)
                        and not cancel_event.is_set()
                    ):
                        loaded = client.is_model_loaded(model)
                        if loaded is not None:
                            break
                        time.sleep(AI_SERVER_WAIT_DELAY)
                    if loaded is not None:
                        # The restarted backend is reachable again: fill the picker with
                        # the real model list so it stops saying "not running".
                        self._ui.schedule(self._ui.models_refreshed)
                    if loaded is False:
                        self._ui.schedule(self._on_model_loading, model, generation, cancel_event)
                if self._ui.current(generation, cancel_event):
                    # vf04: the record carries its query generation. A slow worker
                    # from an old query used to overwrite (or get read as) the
                    # pending record of the query that replaced it.
                    self._action_pending = (heard, route, self._clock(), generation)
                answer = client.generate_stream(
                    model=model,
                    prompt=prompt,
                    cancel_event=cancel_event,
                    on_chunk=on_chunk,
                    messages=messages,
                )
                flush_chunks()
                self._ui.schedule(self._on_query_finished, answer, generation, cancel_event, heard)
            except BackendError as exc:
                # vf02/vf04: every client now says the same normalized thing for a
                # dead backend, a missing model, an out-of-memory or a too-long
                # question (#138); the kind's sentence is what gets heard aloud.
                flush_chunks()
                self._ui.schedule(self._on_query_error, str(exc), generation, cancel_event, exc.message)
            except Exception as exc:  # noqa: BLE001 - a dead query thread must never be silent
                flush_chunks()
                logging.exception("the AI query thread crashed")
                self._ui.schedule(self._on_query_error, f"{type(exc).__name__}: {exc}", generation, cancel_event)

        threading.Thread(target=worker, name=f"ollama-query-{generation}", daemon=True).start()

    def web_context(self, prompt: str, generation: int, cancel_event: threading.Event) -> str:
        self.search_record = None
        self.search_source = ""
        self.search_results = []
        previous = self.last_search_query
        if previous is not None and time.time() - self.last_search_at > SEARCH_REUSE_SECONDS:
            previous = None
        query = websearch.search_query_for(prompt, self._ui.web_mode(), previous)
        if query:
            query = spelling_quote(query)
        if not query:
            return ""
        self.last_search_query = query
        self.last_search_at = time.time()
        self._ui.schedule(self._ui.web_search_started, query, generation, cancel_event)
        try:
            results = websearch.search(query)
        except Exception:  # noqa: BLE001 - offline or blocked: answer without the web
            self._ui.schedule(self._ui.web_unavailable, generation, cancel_event)
            self.search_record = websearch.search_record_fields(query, [], True)
            return ""
        self.search_record = websearch.search_record_fields(query, results, False)
        if results:
            self.last_site_url = (time.time(), results[0].url)
            self.search_source = websearch.source_line(results)
            self.search_results = results
            return websearch.format_for_prompt(query, results)
        return ""

    # ---- one-shot model requests whose result is an action -------------------------

    def draft(self, *, cold_check: str, model: str, user_prompt: str, messages: list[dict[str, str]],
              finish: Callable[[str], str], generation: int, cancel_event: threading.Event) -> None:
        def worker() -> None:
            try:
                client = self._backend()
                if hasattr(client, "is_model_loaded") and client.is_model_loaded(cold_check) is False:
                    self._ui.schedule(self._on_model_loading, cold_check, generation, cancel_event)
                answer = client.generate_stream(
                    model=model,
                    prompt=user_prompt,
                    cancel_event=cancel_event,
                    on_chunk=lambda chunk: None,
                    messages=messages,
                )
                self._ui.schedule(self.draft_finished, answer, generation, cancel_event, finish)
            except OllamaError as exc:
                self._ui.schedule(self.draft_error, str(exc), generation, cancel_event)
            except Exception as exc:  # noqa: BLE001 - a dead draft thread must never be silent either
                logging.exception("the AI draft thread crashed")
                self._ui.schedule(self.draft_error, f"{type(exc).__name__}: {exc}", generation, cancel_event)

        threading.Thread(target=worker, name=f"draft-{generation}", daemon=True).start()

    def draft_finished(self, answer: str, generation: int, cancel_event: threading.Event,
                           finish: Callable[[str], str]) -> bool:
        if not self._ui.current(generation, cancel_event):
            return False
        self._ui.set_busy(True)
        if cancel_event.is_set():
            self._ui.task_end("cancelled")
            self._ui.status("AI request stopped.")
            return False
        try:
            reply = finish(strip_reasoning(answer))
        except Exception as exc:  # noqa: BLE001 - the mail client or LibreOffice may not be installed
            self._ui.task_end("failed", str(exc))
            logging.warning("AI draft failed: %s", exc)
            self._ui.toast(f"Could not complete that: {exc}")
            self._ui.status("Draft failed.")
            return False
        self._ui.task_end("done")
        display = f"{reply}\n{self.search_source}" if self.search_source else reply
        self._ui.show_answer(display)
        if not self._ui.assistant_active():
            self._ui.toast(reply)
        elif self._ui.conversation_active():
            self._ui.speak(reply)
        else:
            self._ui.reply_finished(False)
            self._ui.toast(reply)
        self._ui.status(reply)
        return False

    def draft_error(self, error: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._ui.current(generation, cancel_event) or cancel_event.is_set():
            return False
        self._ui.set_busy(True)
        self._ui.task_end("failed", error)
        logging.warning("AI request failed: %s", error)
        if self._ui.conversation_active():
            self._ui.fail_assistant("The AI request failed")
        self._ui.toast(error)
        self._ui.status(self._ui.idle_status() if self._ui.conversation_active() else "AI request failed.")
        return False

    # ---- the page-aware browser task --------------------------------------------------

    def browse(self, prompt: str, generation: int, cancel_event: threading.Event) -> None:
        """Run the page-aware planner in one worker thread and speak the result."""

        def on_step(call: intents.ToolCall, result: ToolResult, ms: int) -> None:
            self._ui.log_action(
                prompt,
                "browser",
                tool=call.tool,
                args=call.args,
                ok=result.ok,
                speech=result.speech,
                detail=result.detail,
                ms=ms,
            )

        def on_caption(call: intents.ToolCall) -> None:
            if call.tool == "click_on":
                detail = f"Clicking {call.args.get('text', '')}…"
            else:
                detail = "Reading the page…"
            self._ui.schedule(self._ui.browser_step, detail, generation, cancel_event)

        def worker() -> None:
            try:
                def ask_model(messages: list[dict]) -> str:
                    return self._backend().generate_stream(
                        model=self._ui.chat_model(),
                        prompt=messages[-1]["content"],
                        messages=messages,
                        cancel_event=cancel_event,
                        on_chunk=lambda chunk: None,
                    )

                answer = planner.run_browser_task(
                    prompt,
                    ask_model,
                    get_session(),
                    self._ui.tools(),
                    on_step=on_step,
                    should_stop=cancel_event.is_set,
                    on_caption=on_caption,
                )
                self._ui.schedule(self.draft_finished, answer, generation, cancel_event, lambda text: text)
            except OllamaError as exc:
                self._ui.schedule(self.draft_error, str(exc), generation, cancel_event)
            except Exception:
                self._ui.schedule(
                    self.draft_error,
                    "I lost contact with my browser. Say that again and I'll reopen it.",
                    generation,
                    cancel_event,
                )

        threading.Thread(target=worker, name=f"browse-{generation}", daemon=True).start()

    # ---- "open the website" after a web search --------------------------------------------

    def open_website(self, kind: str, words: str | None, original: str) -> None:
        """Open the site of the last searched URL; with words, search for them first."""
        if words:
            threading.Thread(
                target=self.search_for_website,
                args=(kind, words, original, self._ui.query_generation(), self._ui.current_cancel()),
                name="website-search",
                daemon=True,
            ).start()
            return
        self.open_stored_website(kind, original)

    def search_for_website(self, kind: str, words: str, original: str,
                           generation: int, cancel_event: threading.Event) -> None:
        query = spelling_quote(websearch.search_query_for(words, self._ui.web_mode(), None) or words)
        try:
            results = websearch.search(query)
        except Exception:  # noqa: BLE001 - offline or blocked: fall back to what is stored
            results = []
        if results:
            self.last_site_url = (time.time(), results[0].url)
        self._ui.schedule(self.open_stored_website, kind, original, generation, cancel_event)

    def open_stored_website(self, kind: str, original: str, generation: int | None = None,
                            cancel_event: threading.Event | None = None) -> None:
        if generation is not None and not self._ui.current(generation, cancel_event):
            return
        stored = self.last_site_url
        if stored is None or not website_fresh(stored[0], time.time()):
            self._ui.toast("Which website? Tell me its name.")
            return
        url = site_of_url(stored[1]) if kind == "site" else stored[1]
        self._ui.open_url(url, original)

    # ---- warming a picked model up --------------------------------------------------

    def warm_up(self, model: str) -> None:
        # Loading a model into memory does not need the microphone, so this runs
        # whether the assistant is ACTIVE or OFFLINE. A 1-token request makes the
        # cold-start wait happen the moment the user picks the model.
        if not model:
            return
        client = self._backend()
        if not hasattr(client, "is_model_loaded") or client.is_model_loaded(model) is not False:
            return
        generation, cancel_event = self._ui.new_query(False)
        self._ui.schedule(self._on_model_loading, model, generation, cancel_event)

        def worker() -> None:
            try:
                client.generate_stream(
                    model=model,
                    prompt="hi",
                    cancel_event=cancel_event,
                    on_chunk=lambda chunk: None,
                    num_predict=1,
                )
                if not cancel_event.is_set():
                    self._ui.schedule(self._warm_up_finished, generation, cancel_event)
            except OllamaError as exc:
                if not cancel_event.is_set():
                    self._ui.schedule(self._warm_up_failed, str(exc), generation, cancel_event)
            except Exception as exc:  # noqa: BLE001 - the loading spinner must not stick forever
                logging.exception("the AI warm-up thread crashed")
                if not cancel_event.is_set():
                    self._ui.schedule(self._warm_up_failed, f"{type(exc).__name__}: {exc}", generation, cancel_event)

        threading.Thread(target=worker, name=f"warm-up-{generation}", daemon=True).start()

    def _warm_up_finished(self, generation: int, cancel_event: threading.Event) -> bool:
        if not self._ui.current(generation, cancel_event):
            return False
        self.stop_model_loading()
        self._ui.status("Ready")
        return False

    def _warm_up_failed(self, error: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._ui.current(generation, cancel_event):
            return False
        self.stop_model_loading()
        self._ui.toast(error)
        self._ui.status("Ready")
        return False

    # ---- the notes the worker sends while it waits ----------------------------------

    def _on_server_busy(self, generation: int, cancel_event: threading.Event) -> bool:
        if self._ui.current(generation, cancel_event) and not cancel_event.is_set():
            note = "The AI server is busy with another request. Voxa will answer as soon as it is free."
            self._ui.status_busy(note)
            self._ui.toast(note)
            self._ui.assistant_state(AssistantState.THINKING, "AI server busy — waiting in line")
        return False

    def _on_model_loading(self, model: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._ui.current(generation, cancel_event) or cancel_event.is_set():
            return False
        self._loading_model_since = self._clock()
        self._loading_model_name = short_model_name(model)
        self._loading_generation = generation
        self._loading_cancel = cancel_event
        self.show_model_loading()
        self._loading_source = self._ui.ticker_start(self._model_loading_tick)
        return False

    @property
    def loading_model_since(self) -> float | None:
        """When the current "Loading … Ns" note started, or None if none is showing."""
        return self._loading_model_since

    def show_model_loading(self) -> None:
        seconds = int(self._clock() - self._loading_model_since) if self._loading_model_since is not None else 0
        detail = f"Loading {self._loading_model_name}… {seconds}s"
        self._ui.status_busy(detail)
        self._ui.assistant_state(AssistantState.WORKING, detail)

    def _model_loading_tick(self) -> bool:
        if self._loading_model_since is None:
            return False
        if self._loading_cancel is not None and self._loading_cancel.is_set():
            self.stop_model_loading()
            return False
        if self._loading_generation != self._ui.query_generation():
            self.stop_model_loading()
            return False
        self.show_model_loading()
        return True

    def stop_model_loading(self) -> None:
        if self._loading_source:
            self._ui.ticker_stop(self._loading_source)
            self._loading_source = 0
        self._loading_model_since = None
        self._loading_model_name = ""
        self._loading_generation = -1
        self._loading_cancel = None

    def _on_ai_server_starting(self, generation: int, cancel_event: threading.Event) -> bool:
        """A stopped managed backend is being restarted: say so instead of failing silent."""
        if not self._ui.current(generation, cancel_event) or cancel_event.is_set():
            return False
        detail = f"Starting {self._ui.backend_label()}…"
        self._ui.status_busy(detail)
        self._ui.assistant_state(AssistantState.WORKING, detail)
        return False

    # ---- the generation-tagged reports MainWindow reacts to -------------------------

    def _on_query_finished(self, answer: str, generation: int, cancel_event: threading.Event,
                           heard: str = "") -> bool:
        if not self._ui.current(generation, cancel_event):
            return False
        self.stop_model_loading()
        self._ui.set_busy(True)
        if cancel_event.is_set():
            if self._history and self.pending_user_generation == generation:
                self._history.drop_last()
            self.pending_user_generation = None
            self._ui.task_end("cancelled")
            self.finish_action_record(generation, ok=False, detail="cancelled")
            self._ui.status("AI request stopped.")
            return False
        self._ui.task_end("done")
        self._ui.status("AI response complete.")
        # A thinking model streams its scratchpad in with the reply. Replace
        # what was streamed with just the answer, so the chain of thought is
        # neither left on screen nor read aloud in conversation mode.
        spoken = strip_reasoning(answer)
        if claims_action(spoken) and not self._ui.tool_ran():
            # The model talked as if it had acted, but no tool ran: answer honestly.
            self._ui.log_action(heard, "false_claim", detail=first_sentences(answer, 1))
            spoken = FALSE_CLAIM_REPLY
        if spoken and spoken != FALSE_CLAIM_REPLY and self.search_results:
            query = ((self.search_record or {}).get("args") or {}).get("query", "")
            spoken = websearch.grounded_answer(spoken, self.search_results, query)
        if spoken != answer:
            self._ui.show_reply(spoken)
        self._ui.show_answer(spoken)
        replaced = spoken == FALSE_CLAIM_REPLY
        if spoken:
            self._history.add_assistant(spoken)
        elif self._history and self.pending_user_generation == generation:
            # Do not leave a question without an answer in the model thread.
            self._history.drop_last()
        self.pending_user_generation = None
        if spoken and self._ui.conversation_active():
            if replaced:
                self._ui.reset_early_speech()  # what was said early is being corrected: say the honest answer in full
                self._ui.speak(spoken)
            elif not self._ui.early_speech_finish(spoken, generation):
                self._ui.speak(spoken)
        else:
            if self._ui.conversation_active():  # nothing to say: back to waiting
                self._ui.reply_finished(self._ui.waiting_for_prompt())
            if answer and self._ui.auto_speak_enabled():
                self._ui.auto_speak()
        self.finish_action_record(generation, speech=spoken)
        return False

    def _on_query_error(self, error: str, generation: int, cancel_event: threading.Event,
                        spoken: str = "") -> bool:
        if not self._ui.current(generation, cancel_event) or cancel_event.is_set():
            return False
        self.stop_model_loading()
        self._ui.set_busy(True)
        if self._history and self.pending_user_generation == generation:
            # The turn produced no usable answer, so it must not sit in the
            # conversation history as an unanswered question with no reply.
            self._history.drop_last()
        self.pending_user_generation = None
        self._ui.task_end("failed", error)
        self.finish_action_record(generation, ok=False, detail=error)
        logging.warning("AI request failed: %s", error)
        if self._ui.conversation_active():
            self._ui.fail_assistant("The AI request failed")
            self._ui.speak(self.apology(spoken))
        self._ui.toast(error)
        self._ui.status(self._ui.idle_status() if self._ui.conversation_active() else "AI request failed.")
        return False

    def apology(self, kind_message: str = "") -> str:
        """Hands-free users cannot see the toast: the failed request must be heard too.

        vf04: it goes through the SAME path as a normal reply. A bare
        ``speech.speak`` never mutes the conversation and own_speech never
        hears it, so inside an open follow-up window Voxa took her own
        "Sorry, I could not get an answer" as a new request and apologised
        on and on.

        #138: when the failure is a normalized BackendError, its kind's plain
        sentence rides along ("... The AI server ran out of memory while it
        was answering."), so the user hears WHY and not just the old blanket
        apology for every failure.
        """
        return f"{AI_APOLOGY} {kind_message}" if kind_message else AI_APOLOGY

    def finish_action_record(self, generation: int, ok: bool = True, speech: str = "", detail: str = "") -> None:
        """Write the final record of a model request the pre-call record was logged for.

        Without it the action log only ever shows the pre-call line (ok null, empty
        speech, ms 0) for every heard request, healthy or broken alike. vf04: only a
        record written by THIS query generation is finished; a stale worker's tuple
        is dropped instead of stamping the wrong request.
        """
        pending, self._action_pending = self._action_pending, None
        if pending is None or pending[3] != generation:
            return
        heard, route, started = pending[:3]
        self._ui.log_action(
            heard,
            route,
            ok=ok,
            speech=speech,
            detail=detail,
            ms=max(0, int((self._clock() - started) * 1000)),
            timings=self._ui.cmd_timings(self._clock()),
        )
