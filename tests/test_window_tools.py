"""The real window routes recognised commands to tools instead of the model."""

from __future__ import annotations

import threading

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")
Adw.init()

from voxa import apps  # noqa: E402
from voxa.agent import intents  # noqa: E402
from voxa.agent.actionlog import ActionLog  # noqa: E402
from voxa.agent.policy import RiskLevel  # noqa: E402
from voxa.agent.registry import Tool, ToolRegistry  # noqa: E402
from voxa.agent.result import ToolResult  # noqa: E402
from voxa.agent.tools import typing as typing_mod  # noqa: E402
from voxa.apps import DesktopApp  # noqa: E402
from voxa.audio import AudioDevice  # noqa: E402
from voxa.hardware import MODEL_CATALOG  # noqa: E402
from voxa.ollama import OllamaError  # noqa: E402
from voxa.window import MainWindow, short_model_name  # noqa: E402


def _pump(iterations: int = 150) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


def _settle(win, predicate, limit: int = 3000) -> bool:
    ctx = GLib.MainContext.default()
    for _ in range(limit):
        ctx.iteration(False)
        if predicate():
            return True
    return False


class FakeAudio:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, device_id, on_audio, on_level, on_error) -> None:
        self.calls.append("start")
        self._active = True

    def stop(self) -> None:
        self.calls.append("stop")
        self._active = False


class FakeSpeech:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text, rate, voice, on_started=None, on_done=None, on_error=None) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        pass


class FakeClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def is_busy(self) -> bool:
        return False

    def is_model_loaded(self, model: str = "") -> bool:
        return True

    def generate_stream(self, model, prompt, cancel_event, on_chunk, messages=None) -> str:
        self.queries.append(prompt)
        return "Paris."


class LoadingClient:
    def __init__(self, gate: threading.Event, answer: str = "Paris.") -> None:
        self.gate = gate
        self.answer = answer
        self.queries: list[str] = []

    def is_busy(self) -> bool:
        return False

    def is_model_loaded(self, model: str) -> bool:
        return False

    def generate_stream(self, model, prompt, cancel_event, on_chunk, messages=None) -> str:
        self.queries.append(prompt)
        self.gate.wait()
        on_chunk(self.answer)
        return self.answer


class LoadedClient:
    def __init__(self, answer: str = "Paris.") -> None:
        self.answer = answer
        self.queries: list[str] = []

    def is_busy(self) -> bool:
        return False

    def is_model_loaded(self, model: str) -> bool:
        return True

    def generate_stream(self, model, prompt, cancel_event, on_chunk, messages=None) -> str:
        self.queries.append(prompt)
        on_chunk(self.answer)
        return self.answer


class FakeRegistry:
    def __init__(self, speech: str = "Done.") -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.speech = speech

    def call(self, name: str, args: dict[str, str]) -> ToolResult:
        self.calls.append((name, dict(args)))
        return ToolResult.success(self.speech)


def _plan_registry() -> tuple[ToolRegistry, list[tuple[str, dict[str, str]]]]:
    calls: list[tuple[str, dict[str, str]]] = []

    def handler(name: str):
        def run(args: dict[str, str]) -> ToolResult:
            calls.append((name, dict(args)))
            return ToolResult.success("")
        return run

    registry = ToolRegistry()
    registry.register(
        Tool(name="open_site", description="Open a website", parameters={"name": "site"},
             risk=RiskLevel.REVERSIBLE, handler=handler("open_site"), required=("name",))
    )
    registry.register(
        Tool(name="compose_gmail", description="Start a new Gmail message", parameters={},
             risk=RiskLevel.REVERSIBLE, handler=handler("compose_gmail"))
    )
    registry.register(
        Tool(name="press_key", description="Press a key", parameters={"key": "key"},
             risk=RiskLevel.REVERSIBLE, handler=handler("press_key"), required=("key",))
    )
    return registry, calls


class FakePlanClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.queries: list[str] = []

    def is_busy(self) -> bool:
        return False

    def is_model_loaded(self, model: str = "") -> bool:
        return True

    def generate_stream(self, model, prompt, cancel_event, on_chunk, messages=None) -> str:
        self.queries.append(prompt)
        return self.answer


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.tooltest")
    app.register(None)
    return app


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "_load_whisper",
        "_load_wake_whisper",
        "_detect_hardware_async",
        "_refresh_ollama_models",
        "_refresh_devices",
        "_start_ai_server_async",
        "_restart_ai_server_async",
        "_stop_ai_server_async",
    ):
        monkeypatch.setattr(MainWindow, name, lambda *args, **kwargs: None)
    client = FakeClient()
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    win = MainWindow(application)
    win._fake_client = client
    win.action_log = ActionLog(tmp_path / "actions.jsonl")
    win.audio = FakeAudio()
    win.speech = FakeSpeech()
    win.whisper._model = object()
    win.devices = [AudioDevice(identifier="mic", name="Microphone", device=None)]
    win.settings.ollama_model = "tiny"
    win.present()
    _pump()
    yield win
    win._closing = True
    win.assistant.go_offline()
    win.destroy()
    _pump(20)


def test_open_gmail_goes_to_the_registry_never_to_the_model(window) -> None:
    registry = FakeRegistry()
    window.tools = registry
    window.ask_ai("open gmail")
    assert _settle(window, lambda: registry.calls)
    assert registry.calls == [("open_site", {"name": "gmail"})]
    assert window._fake_client.queries == []
    assert _settle(window, lambda: window._query_task_id is None)


def test_open_gmail_is_logged_as_a_tool_call(window) -> None:
    registry = FakeRegistry()
    window.tools = registry
    window.ask_ai("open gmail")
    assert _settle(window, lambda: registry.calls)
    records = window.action_log.read()
    assert len(records) == 1
    assert records[0]["route"] == "tool"
    assert records[0]["tool"] == "open_site"
    assert records[0]["heard"] == "open gmail"
    assert records[0]["ok"] is True


def test_a_model_question_is_logged_as_unrouted(window) -> None:
    registry = FakeRegistry()
    window.tools = registry
    window.ask_ai("what is the capital of France")
    assert _settle(window, lambda: window._fake_client.queries)
    records = window.action_log.read()
    assert len(records) == 1
    assert records[0]["route"] == "model"
    assert records[0]["heard"] == "what is the capital of France"
    assert records[0]["ok"] is None


def test_send_it_calls_send_gmail_at_once(window) -> None:
    registry = FakeRegistry(speech="")
    window.tools = registry
    window.ask_ai("send it")
    assert _settle(window, lambda: registry.calls)
    assert registry.calls == [("send_gmail", {})]
    assert window._fake_client.queries == []
    assert window.speech.spoken == []


def test_a_real_question_still_asks_the_model(window) -> None:
    registry = FakeRegistry()
    window.tools = registry
    window.ask_ai("what is the capital of France")
    assert _settle(window, lambda: window._fake_client.queries)
    assert registry.calls == []
    assert window._fake_client.queries == ["what is the capital of France"]


def test_open_files_launches_an_app(window, monkeypatch) -> None:
    launched: list[str] = []
    monkeypatch.setattr(
        apps,
        "list_apps",
        lambda: [DesktopApp(name="Files", path="/usr/bin/nautilus")],
    )
    monkeypatch.setattr(apps, "launch", lambda app: launched.append(app.name))
    window.ask_ai("open files")
    assert _settle(window, lambda: launched)
    assert launched == ["Files"]


def test_dictation_types_into_the_window_without_the_model(window, monkeypatch) -> None:
    typed: list[str] = []
    erased: list[int] = []
    monkeypatch.setattr(typing_mod, "type_text", lambda text: typed.append(text))
    monkeypatch.setattr(typing_mod, "erase", lambda count: erased.append(count))

    window.ask_ai("start dictating")
    assert window._external_dictation
    assert window._fake_client.queries == []

    window.ask_ai("hello comma world period")
    assert _settle(window, lambda: typed)
    assert typed == ["hello, world. "]
    assert window._fake_client.queries == []

    window.ask_ai("scratch that")
    assert _settle(window, lambda: erased)
    assert erased == [14]
    assert window._last_dictated == ""

    window.ask_ai("stop dictating")
    assert not window._external_dictation
    assert window._fake_client.queries == []


def test_going_offline_ends_dictation(window, monkeypatch) -> None:
    typed: list[str] = []
    monkeypatch.setattr(typing_mod, "type_text", lambda text: typed.append(text))
    monkeypatch.setattr(typing_mod, "erase", lambda count: None)
    window.ask_ai("dictate")
    assert window._external_dictation
    window.stop_current_work()
    assert not window._external_dictation


def test_unrouted_command_is_planned_and_run_in_order(window, monkeypatch) -> None:
    request = "open gmail and start a new message"
    assert intents.route(request) is None
    registry, calls = _plan_registry()
    window.tools = registry
    client = FakePlanClient(
        '{"steps": [{"tool": "open_site", "args": {"name": "gmail"}}, '
        '{"tool": "compose_gmail", "args": {}}], '
        '"say": "Opening Gmail and starting a message."}'
    )
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai(request)
    assert _settle(window, lambda: calls)
    assert calls == [("open_site", {"name": "gmail"}), ("compose_gmail", {})]
    assert _settle(window, lambda: window.status_label.get_text() == "Opening Gmail and starting a message.")
    records = window.action_log.read()
    plan_records = [record for record in records if record["route"] == "plan"]
    assert len(plan_records) == 2
    assert [record["tool"] for record in plan_records] == ["open_site", "compose_gmail"]
    assert all(record["ok"] is True and record["heard"] == request for record in plan_records)


def test_a_non_json_plan_reply_fails_cleanly(window, monkeypatch) -> None:
    request = "open gmail and start a new message"
    registry, calls = _plan_registry()
    window.tools = registry
    client = FakePlanClient("I am not sure how to do that.")
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai(request)
    assert _settle(window, lambda: window.status_label.get_text() == "I couldn't work out how to do that.")
    assert calls == []
    records = window.action_log.read()
    plan_records = [record for record in records if record["route"] == "plan"]
    assert len(plan_records) == 1
    assert plan_records[0]["ok"] is False
    assert plan_records[0]["heard"] == request


def test_short_model_name_strips_prefix_and_truncates() -> None:
    assert short_model_name("tiny") == "tiny"
    assert short_model_name("vendor/deep/llama-3-8b") == "llama-3-8b"
    assert short_model_name("x" * 40) == "x" * 27 + "…"


def test_a_cold_model_shows_a_loading_caption(window, monkeypatch) -> None:
    gate = threading.Event()
    client = LoadingClient(gate)
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai("what is the capital of France")
    assert _settle(window, lambda: window.status_label.get_text().startswith("Loading "))
    assert "tiny" in window.status_label.get_text()
    assert window._loading_model_since is not None


def test_releasing_the_gate_and_a_chunk_clears_the_caption(window, monkeypatch) -> None:
    gate = threading.Event()
    client = LoadingClient(gate)
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai("what is the capital of France")
    assert _settle(window, lambda: window.status_label.get_text().startswith("Loading "))
    gate.set()
    assert _settle(window, lambda: window._loading_model_since is None)


def test_a_loaded_model_never_shows_a_loading_caption(window, monkeypatch) -> None:
    client = LoadedClient()
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai("what is the capital of France")
    assert _settle(window, lambda: client.queries)
    assert window._loading_model_since is None
    assert not window.status_label.get_text().startswith("Loading ")


def test_a_long_way_round_offers_one_coaching_suggestion(window) -> None:
    registry = FakeRegistry()
    window.tools = registry
    window.ask_ai("open brave")
    assert _settle(window, lambda: len(registry.calls) == 1)
    window.ask_ai("open gmail")
    assert _settle(window, lambda: len(registry.calls) == 2)
    assert _settle(window, lambda: any(r["route"] == "suggestion" for r in window.action_log.read()))
    suggestions = [r for r in window.action_log.read() if r["route"] == "suggestion"]
    assert len(suggestions) == 1
    assert suggestions[0]["detail"] == "direct:open_app>open_site"
    assert suggestions[0]["ok"] is None


def test_coaching_is_off_when_suggestions_are_disabled(window) -> None:
    registry = FakeRegistry()
    window.tools = registry
    window.settings.suggestions_enabled = False
    window.ask_ai("open brave")
    assert _settle(window, lambda: len(registry.calls) == 1)
    window.ask_ai("open gmail")
    assert _settle(window, lambda: len(registry.calls) == 2)
    assert _settle(window, lambda: len(window.action_log.read()) >= 2)
    assert not any(r["route"] == "suggestion" for r in window.action_log.read())


def test_a_browser_request_runs_the_page_aware_planner(window, monkeypatch) -> None:
    request = "find the opening hours on the luigis.com website"
    calls: list[tuple[str, dict[str, str]]] = []

    def handler(name: str):
        def run(args: dict[str, str]) -> ToolResult:
            calls.append((name, dict(args)))
            return ToolResult.success("")
        return run

    registry = ToolRegistry()
    registry.register(
        Tool(name="click_on", description="Click an element", parameters={"text": "text"},
             risk=RiskLevel.REVERSIBLE, handler=handler("click_on"), required=("text",))
    )
    window.tools = registry

    class Session:
        def outline(self):
            return [{"n": 1, "kind": "link", "text": "Opening hours", "placeholder": "", "href": "/hours"}]

        def title(self):
            return "Luigi's"

        def url(self):
            return "https://luigis.com/"

    monkeypatch.setattr("voxa.window.get_session", lambda: Session())

    class BrowserClient:
        def __init__(self) -> None:
            self.answers = iter(
                [
                    '{"tool": "click_on", "args": {"text": "Opening hours"}}',
                    '{"done": true, "say": "Opening hours are 9 to 5."}',
                ]
            )

        def is_busy(self) -> bool:
            return False

        def is_model_loaded(self, model: str = "") -> bool:
            return True

        def generate_stream(self, model, prompt, cancel_event, on_chunk, messages=None) -> str:
            return next(self.answers)

    client = BrowserClient()
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai(request)
    assert _settle(window, lambda: calls)
    assert calls == [("click_on", {"text": "Opening hours"})]
    assert _settle(window, lambda: window.status_label.get_text() == "Opening hours are 9 to 5.")
    records = window.action_log.read()
    browser_records = [record for record in records if record["route"] == "browser"]
    assert len(browser_records) == 1
    assert browser_records[0]["tool"] == "click_on"
    assert browser_records[0]["ok"] is True
    assert browser_records[0]["heard"] == request


def test_cancelling_stops_the_browser_task_after_the_current_step(window, monkeypatch) -> None:
    request = "find the opening hours on the luigis.com website"
    calls: list[tuple[str, dict[str, str]]] = []

    def handler(name: str):
        def run(args: dict[str, str]) -> ToolResult:
            calls.append((name, dict(args)))
            return ToolResult.success("")
        return run

    registry = ToolRegistry()
    registry.register(
        Tool(name="click_on", description="Click an element", parameters={"text": "text"},
             risk=RiskLevel.REVERSIBLE, handler=handler("click_on"), required=("text",))
    )
    window.tools = registry

    class Session:
        def outline(self):
            return [{"n": 1, "kind": "link", "text": "Opening hours", "placeholder": "", "href": "/hours"}]

        def title(self):
            return "Luigi's"

        def url(self):
            return "https://luigis.com/"

    monkeypatch.setattr("voxa.window.get_session", lambda: Session())

    class CancelClient:
        def __init__(self) -> None:
            self.n = 0

        def is_busy(self) -> bool:
            return False

        def is_model_loaded(self, model: str = "") -> bool:
            return True

        def generate_stream(self, model, prompt, cancel_event, on_chunk, messages=None) -> str:
            self.n += 1
            if self.n == 1:
                return '{"tool": "click_on", "args": {"text": "Opening hours"}}'
            cancel_event.wait()
            return '{"done": true, "say": "Never reported."}'

    client = CancelClient()
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai(request)
    window.query_cancel.set()
    assert _settle(window, lambda: calls)
    assert calls == [("click_on", {"text": "Opening hours"})]
    assert _settle(window, lambda: client.n == 1)
    assert window.status_label.get_text() != "Never reported."
    records = window.action_log.read()
    assert len([record for record in records if record["route"] == "browser"]) == 1


def test_a_model_claim_of_an_action_is_replaced(window, monkeypatch) -> None:
    client = LoadedClient("Done. Brutal Chess is closed.")
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai("what is the capital of France")
    assert _settle(window, lambda: client.queries)
    assert _settle(window, lambda: any(r["route"] == "false_claim" for r in window.action_log.read()))
    assert window._get_text(window.response_view) == (
        "I couldn't do that. Try saying it as a direct command, like “close Brutal Chess”."
    )
    records = window.action_log.read()
    claims = [record for record in records if record["route"] == "false_claim"]
    assert len(claims) == 1
    assert claims[0]["heard"] == "what is the capital of France"
    assert claims[0]["detail"] == "Done."


def test_a_normal_model_answer_passes_through_unchanged(window, monkeypatch) -> None:
    client = LoadedClient("The capital of France is Paris.")
    monkeypatch.setattr(MainWindow, "_ai_client", lambda self: client)
    window.ask_ai("what is the capital of France")
    assert _settle(window, lambda: client.queries)
    assert _settle(window, lambda: window._get_text(window.response_view) != "")
    assert window._get_text(window.response_view) == "The capital of France is Paris."
    records = window.action_log.read()
    assert not [record for record in records if record["route"] == "false_claim"]


def test_reminders_tick_delivers_due_items_as_toasts(window, monkeypatch) -> None:
    from datetime import datetime, timedelta

    from voxa.agent.reminders import Reminder

    toasts: list[str] = []
    monkeypatch.setattr(window.shell, "show_notice", lambda text: toasts.append(text))
    window.reminder_store.add(
        Reminder(id="t1", due=datetime.now() - timedelta(seconds=1), text="10 minutes", kind="timer")
    )
    window.reminder_store.add(
        Reminder(id="r1", due=datetime.now() - timedelta(seconds=1), text="stretch", kind="reminder")
    )
    window._reminders_tick()
    assert len(toasts) == 2
    assert window.reminder_store.upcoming() == []


def test_a_two_step_routine_runs_both_steps_in_order(window) -> None:
    registry = FakeRegistry()
    window.tools = registry
    window.ask_ai("when I say coffee time, open gmail and play music")
    assert _settle(window, lambda: window.routine_store.all())
    window.ask_ai("coffee time")
    assert _settle(window, lambda: len(registry.calls) == 2)
    assert registry.calls == [("open_site", {"name": "gmail"}), ("play_music", {"query": "music"})]
    records = window.action_log.read()
    assert any(r["route"] == "routine" for r in records)


def _children(widget):
    if hasattr(widget, "get_child"):
        child = widget.get_child()
        if child is not None:
            return child
    if hasattr(widget, "get_first_child"):
        return widget.get_first_child()
    return None


def _walk(widget):
    child = _children(widget)
    while child is not None:
        nxt = child.get_next_sibling()
        yield child
        yield from _walk(child)
        child = nxt


def _row_titles(widget):
    return [w.get_title() for w in _walk(widget) if isinstance(w, Adw.ActionRow)]


def _has_label(widget, text):
    return any(isinstance(w, Gtk.Label) and w.get_label() == text for w in _walk(widget))


def _find_entry(widget):
    return next(w for w in _walk(widget) if isinstance(w, Adw.EntryRow))


def _find_pull_button(widget):
    return next(w for w in _walk(widget) if isinstance(w, Gtk.Button) and w.get_label() == "Pull")


def _capture_dialogs(monkeypatch):
    dialogs: list[Adw.Dialog] = []
    real_present = Adw.Dialog.present
    monkeypatch.setattr(
        Adw.Dialog,
        "present",
        lambda dialog, opener: (dialogs.append(dialog), real_present(dialog, opener))[1],
    )
    return dialogs


class FlakyClient:
    def __init__(self) -> None:
        self.calls = 0

    def list_models(self):
        self.calls += 1
        if self.calls == 1:
            raise OllamaError("server not up")
        return ["alpha", "beta"]

    def list_models_detailed(self):
        return []

    def pull_model(self, model, *, cancel_event, on_progress):
        on_progress("downloading", 0, 0)

    def delete_model(self, model):
        pass


class PullClient:
    def __init__(self) -> None:
        self.pulled = False

    def list_models(self):
        return ["newmodel"] if self.pulled else []

    def list_models_detailed(self):
        return []

    def pull_model(self, model, *, cancel_event, on_progress):
        self.pulled = True
        on_progress("downloading", 0, 0)
        on_progress("downloading", 50, 100)

    def delete_model(self, model):
        pass


def _fast_model_wait(monkeypatch):
    monkeypatch.setattr("voxa.window.MODEL_WAIT_ATTEMPTS", 2)
    monkeypatch.setattr("voxa.window.MODEL_WAIT_DELAY", 0.01)


def test_dialog_shows_starting_then_the_installed_models(window, monkeypatch) -> None:
    _fast_model_wait(monkeypatch)
    shared = FlakyClient()
    monkeypatch.setattr("voxa.window.OllamaClient", lambda *a, **k: shared)
    dialogs = _capture_dialogs(monkeypatch)
    window._show_model_manager()
    dialog = dialogs[0]
    assert _settle(window, lambda: "Starting Ollama…" in _row_titles(dialog))
    assert "alpha" not in _row_titles(dialog)
    assert _settle(window, lambda: "alpha" in _row_titles(dialog))
    assert "Starting Ollama…" not in _row_titles(dialog)


def test_pull_keeps_the_dialog_open_and_lists_the_model(window, monkeypatch) -> None:
    _fast_model_wait(monkeypatch)
    shared = PullClient()
    monkeypatch.setattr("voxa.window.OllamaClient", lambda *a, **k: shared)
    window.settings.ollama_model = ""
    window.ollama_models = []
    dialogs = _capture_dialogs(monkeypatch)
    window._show_model_manager()
    dialog = dialogs[0]
    _find_entry(dialog).set_text("newmodel")
    _find_pull_button(dialog).emit("clicked")
    assert _settle(window, lambda: "newmodel" in _row_titles(dialog))
    assert window.settings.ollama_model == "newmodel"
    assert _find_entry(dialog).get_text() == ""


def test_smallest_chip_present_when_not_installed(window, monkeypatch) -> None:
    smallest = MODEL_CATALOG[0].name
    window._suggested_models = ["gemma3:1b"]
    window.ollama_models = []
    dialogs = _capture_dialogs(monkeypatch)
    window._show_model_manager()
    assert _has_label(dialogs[0], "Smallest — runs anywhere")
    assert any(isinstance(w, Gtk.Button) and w.get_label() == smallest for w in _walk(dialogs[0]))


def test_smallest_chip_absent_when_already_installed(window, monkeypatch) -> None:
    smallest = MODEL_CATALOG[0].name
    window._suggested_models = ["gemma3:1b"]
    window.ollama_models = [smallest]
    dialogs = _capture_dialogs(monkeypatch)
    window._show_model_manager()
    assert not _has_label(dialogs[0], "Smallest — runs anywhere")
