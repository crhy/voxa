"""OFFLINE must be a real kill switch in the production window (issue #7 section 3).

These drive the real MainWindow, with fake audio, speech and Whisper, through ACTIVE and
OFFLINE from every interesting state. A late callback that was queued before OFFLINE (a heard
prompt, the end of a spoken reply, a microphone error) must never restart listening or work.
"""

from __future__ import annotations

import time

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")
Adw.init()

from voxa.audio import AudioDevice  # noqa: E402
from voxa.ollama import OllamaError  # noqa: E402
from voxa.ui.state import AssistantState, TaskState  # noqa: E402
from voxa.window import MainWindow  # noqa: E402


def _pump(iterations: int = 60) -> None:
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


class FakeAudio:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._active = False
        self.stuck = False  # when True, stop() does not actually release the microphone

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self, device_id, on_audio, on_level, on_error) -> None:
        self.calls.append("start")
        self._active = True

    def stop(self) -> None:
        self.calls.append("stop")
        if not self.stuck:
            self._active = False


class FakeSpeech:
    def __init__(self) -> None:
        self.stops = 0
        self.callbacks: dict[str, object] = {}

    def speak(self, text, rate, voice, on_started=None, on_done=None, on_error=None) -> None:
        self.callbacks = {"started": on_started, "done": on_done, "error": on_error}

    def stop(self) -> None:
        self.stops += 1


@pytest.fixture(scope="module")
def application():
    app = Adw.Application(application_id="io.github.crhy.voxa.offlinetest")
    app.register(None)
    return app


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    for var, sub in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("_load_whisper", "_load_wake_whisper", "_detect_hardware_async", "_refresh_ollama_models", "_refresh_devices"):
        monkeypatch.setattr(MainWindow, name, lambda *args, **kwargs: None)
    win = MainWindow(application)
    win.audio = FakeAudio()
    win.speech = FakeSpeech()
    win.whisper._model = object()  # WhisperService.ready
    win.devices = [AudioDevice(id="mic", name="Microphone")] if "id" in AudioDevice.__dataclass_fields__ else [object()]
    win.present()
    _pump()
    yield win
    win._closing = True
    win.assistant.go_offline()
    win.destroy()
    _pump(20)


def _state(win) -> AssistantState:
    return win.assistant_model.state


def test_active_starts_listening_and_shows_ready(window) -> None:
    assert window.assistant.activate() is True
    assert _state(window) is AssistantState.READY
    assert window.audio.calls == ["start"] and window.audio.is_active
    assert window.conversation_active and window.conversation is not None


def test_active_failure_ends_offline_and_leaves_the_microphone_alone(window) -> None:
    window.devices = []
    assert window.assistant.activate() is False
    assert _state(window) is AssistantState.OFFLINE
    # a failed start still runs the stop cleanup, but never starts the microphone
    assert "start" not in window.audio.calls and not window.audio.is_active


def test_offline_stops_the_microphone_wake_monitor_speech_and_says_so(window) -> None:
    window.assistant.activate()
    confirmed = window.assistant.go_offline()
    assert confirmed is True
    assert _state(window) is AssistantState.OFFLINE
    assert "microphone off" in window.assistant_model.detail
    assert not window.audio.is_active
    assert window.conversation is None and not window.conversation_active
    assert window.speech.stops >= 1


def test_a_stuck_microphone_is_reported_not_hidden(window) -> None:
    window.assistant.activate()
    window.audio.stuck = True
    assert window.assistant.go_offline() is False
    assert _state(window) is AssistantState.OFFLINE
    assert "may still be active" in window.assistant_model.detail


def test_prompt_queued_before_offline_never_reaches_the_model(window, monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(window, "ask_ai", lambda text=None: asked.append(text))
    window.assistant.activate()
    token = window.assistant.token()
    queued_prompt = window._for_session(token, window._on_conversation_prompt)

    queued_prompt("what time is it")
    assert asked == ["what time is it"]  # while ACTIVE the very same callback works

    window.assistant.go_offline()
    queued_prompt("delete everything")  # was already queued when OFFLINE was pressed
    assert asked == ["what time is it"]
    assert _state(window) is AssistantState.OFFLINE


def test_speech_finishing_after_offline_cannot_revive_the_state(window) -> None:
    window.assistant.activate()
    window.assistant.wake(window.assistant.token())
    window.assistant.prompt_accepted(window.assistant.token())
    window._conversation_speak("Hello there")
    assert _state(window) is AssistantState.SPEAKING
    done = window.speech.callbacks["done"]

    window.assistant.go_offline()
    done()  # the "speech finished" notification arriving late
    _pump()
    assert _state(window) is AssistantState.OFFLINE


def test_microphone_error_from_an_old_session_is_ignored(window) -> None:
    window.assistant.activate()
    stale = window._for_session(window.assistant.token(), window._conversation_capture_error)
    window.assistant.go_offline()
    window.assistant.activate()  # a new session
    stale("device unplugged")  # belongs to the old session
    assert _state(window) is AssistantState.READY
    assert window.audio.is_active


def test_offline_cancels_running_tasks_and_blocks_new_work(window) -> None:
    window.assistant.activate()
    task = window.assistant.begin_task("Organize Downloads")
    assert task is not None and window.assistant_model.tasks[task.id].state is TaskState.RUNNING

    window.assistant.go_offline()
    assert window.assistant_model.tasks[task.id].state is TaskState.CANCELLED
    assert window.assistant.begin_task("Another job") is None


def test_offline_while_waiting_for_a_choice_cancels_it(window) -> None:
    window.assistant.activate()
    task = window.assistant.begin_task("Install package")
    window.assistant_model.request_choice(task.id, ["Allow once", "Cancel"])
    assert window.assistant_model.tasks[task.id].requires_user_input

    window.assistant.go_offline()
    assert not window.assistant_model.tasks[task.id].requires_user_input
    assert window.assistant_model.tasks[task.id].state is TaskState.CANCELLED


def test_old_conversation_button_and_shortcut_use_the_same_controller(window) -> None:
    window.toggle_conversation()
    _pump()
    assert _state(window) is AssistantState.READY and window.audio.is_active
    window.toggle_conversation()
    _pump()
    assert _state(window) is AssistantState.OFFLINE and not window.audio.is_active


def test_starting_dictation_takes_the_assistant_offline(window) -> None:
    window.assistant.activate()
    window.start_recording()
    assert _state(window) is AssistantState.OFFLINE
    assert window.listening
    window.stop_recording()


def test_the_stop_button_is_the_same_kill_switch(window) -> None:
    window.assistant.activate()
    window.stop_current_work()
    assert _state(window) is AssistantState.OFFLINE and not window.audio.is_active


def test_go_offline_is_harmless_when_already_offline(window) -> None:
    assert window.assistant.go_offline() is True
    assert window.assistant.go_offline() is True
    assert _state(window) is AssistantState.OFFLINE


class FakeClient:
    def __init__(self, text: str = "It is noon.", error: Exception | None = None) -> None:
        self.text, self.error = text, error

    def generate_stream(self, *, model, prompt, cancel_event, on_chunk, messages=None, **_kwargs) -> str:
        if self.error is not None:
            raise self.error
        on_chunk(self.text)
        return self.text


def _wait_for(condition, timeout: float = 4.0) -> bool:
    ctx = GLib.MainContext.default()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        while ctx.iteration(False):
            pass
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_a_hands_free_turn_flows_through_the_states_and_the_task_list(window, monkeypatch) -> None:
    monkeypatch.setattr(window, "_ai_client", lambda: FakeClient("It is noon."))
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    states: list[AssistantState] = []
    window.assistant_model.on_state_changed = lambda state, _detail: states.append(state)

    window._on_conversation_woken()
    assert _state(window) is AssistantState.LISTENING
    window._on_conversation_prompt("what time is it")
    assert _state(window) is AssistantState.THINKING
    (task,) = window.assistant_model.active_tasks()
    assert task.title.startswith("Answering: what time is it")

    assert _wait_for(lambda: _state(window) is AssistantState.SPEAKING)
    assert window.assistant_model.tasks[task.id].state is TaskState.DONE
    window.speech.callbacks["done"]()
    # after the reply Voxa keeps listening for a follow-up without the wake word
    assert _wait_for(lambda: _state(window) is AssistantState.LISTENING)
    assert window.conversation.waiting_for_prompt
    assert window.shell.exchange_panel.get_visible()
    window._follow_up_expired(window._follow_up_token)
    assert _state(window) is AssistantState.READY
    assert not window.conversation.waiting_for_prompt
    assert states[:3] == [AssistantState.LISTENING, AssistantState.THINKING, AssistantState.SPEAKING]
    assert states[-2:] == [AssistantState.LISTENING, AssistantState.READY]


def test_a_failed_request_fails_its_task_and_briefly_shows_an_error(window, monkeypatch) -> None:
    monkeypatch.setattr(window, "_ai_client", lambda: FakeClient(error=OllamaError("Could not connect")))
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    window._on_conversation_prompt("hello")

    assert _wait_for(lambda: _state(window) is AssistantState.ERROR)
    failed = [t for t in window.assistant_model.tasks.values() if t.state is TaskState.FAILED]
    assert len(failed) == 1 and "Could not connect" in failed[0].error
    # one failed request is not a global outage: after the pause Voxa is READY again
    window._recover_assistant(window.assistant.token())
    assert _state(window) is AssistantState.READY


def test_a_spoken_cancel_returns_to_ready_instead_of_sticking_on_thinking(window, monkeypatch) -> None:
    monkeypatch.setattr(window, "_ai_client", lambda: FakeClient("never delivered"))
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    window.assistant.wake(window.assistant.token())
    window.assistant.prompt_accepted(window.assistant.token())
    task = window.assistant.begin_task("Answering: something long")
    window._query_task_id = task.id  # as ask_ai does for the request being cancelled

    window._on_conversation_exit("cancel")
    assert _state(window) is AssistantState.READY  # the assistant is still ACTIVE, just idle
    assert window.assistant_model.tasks[task.id].state is TaskState.CANCELLED
    assert window._query_task_id is None
    assert window.audio.is_active  # "cancel" does not switch the microphone off


def test_a_spoken_stop_blanks_the_interface_but_stays_active(window) -> None:
    window.assistant.activate()
    window.shell.exchange_panel.show_question("what time is it")
    window.shell.exchange_panel.show_answer("Noon.")
    window._set_text(window.transcript_view, "what time is it")
    window._set_text(window.response_view, "Noon.")

    window._on_conversation_exit("stop")
    assert not window.shell.exchange_panel.get_visible()
    assert window._get_text(window.transcript_view) == "" and window._get_text(window.response_view) == ""
    assert _state(window) is AssistantState.READY and window.audio.is_active


def test_a_spoken_goodbye_goes_offline(window) -> None:
    window.assistant.activate()
    window._on_conversation_exit("goodbye")
    assert _state(window) is AssistantState.OFFLINE and not window.audio.is_active


def test_speech_finishing_after_a_barge_in_cannot_cut_short_the_next_request(window) -> None:
    window.assistant.activate()
    token = window.assistant.token()
    window.assistant.wake(token)
    window.assistant.prompt_accepted(token)
    window._conversation_speak("A long answer")
    assert _state(window) is AssistantState.SPEAKING
    old_done = window.speech.callbacks["done"]

    assert window.assistant.barge_in(token)  # the user talks over Voxa
    window.assistant.prompt_accepted(token)  # ...and the interruption becomes the next request
    assert _state(window) is AssistantState.THINKING

    old_done()  # the old speech's completion notification arrives late
    _pump()
    assert _state(window) is AssistantState.THINKING


def test_a_spoken_cancel_after_the_wake_word_returns_to_ready(window) -> None:
    window.assistant.activate()
    window.assistant.wake(window.assistant.token())
    assert _state(window) is AssistantState.LISTENING
    window._on_conversation_exit("cancel")
    assert _state(window) is AssistantState.READY


def test_open_command_launches_the_matching_menu_app_and_confirms(window, monkeypatch) -> None:
    from voxa import apps

    launched = []
    menu = [apps.DesktopApp("Brutal Chess", "/x/io.github.crhy.BrutalChess.desktop"), apps.DesktopApp("Brave", "/x/brave.desktop")]
    monkeypatch.setattr(apps, "list_apps", lambda: menu)
    monkeypatch.setattr(apps, "launch", lambda app: launched.append(app.name))
    monkeypatch.setattr(window, "_ai_client", lambda: FakeClient("must not be asked"))
    window.settings.ollama_model = "test-model"
    window.assistant.activate()
    window.assistant.wake(window.assistant.token())

    window.ask_ai("Open Brutal Chess.")
    assert _wait_for(lambda: launched == ["Brutal Chess"])
    assert _wait_for(lambda: window.shell.exchange_panel.answer.get_text() == "Opening Brutal Chess.")
    assert window._get_text(window.response_view) == ""  # the AI was never asked
