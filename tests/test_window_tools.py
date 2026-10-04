"""The real window routes recognised commands to tools instead of the model."""

from __future__ import annotations

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
from voxa.window import MainWindow  # noqa: E402


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

    def generate_stream(self, model, prompt, cancel_event, on_chunk, messages=None) -> str:
        self.queries.append(prompt)
        return "Paris."


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
