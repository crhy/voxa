"""Issue #138 part 2: every worker completion carries its generation.

MainWindow starts dozens of ``threading.Thread`` workers. Each one captures the
token its work started under, and the completion that reports back (through
``idle``) must do nothing when the token is stale — OFFLINE, a window close, or
a newer run of the same work retired it. These tests drive the completions
directly (no threads, no display): a stale token must leave every attribute,
toast, dialog and setting exactly as it was.

Run headless too: when PyGObject is not installed (e.g. the sandbox runtime's
python), a minimal ``gi`` stub is registered so ``voxa.window`` imports and the
contract above can still be checked. With real gi the whole file behaves the
same; the real-window OFFLINE kill-switch effects live in
tests/test_window_offline.py.
"""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# gi: the real thing when it is installed, a minimal stub otherwise.
# ---------------------------------------------------------------------------

_REAL_GI = True
try:  # noqa: E402 - deliberate import dance: stub gi only when it is missing
    import gi  # noqa: F401
except ImportError:
    _REAL_GI = False

    class _StubMeta(type):
        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            sub = _StubMeta(name, (cls,), {})
            setattr(cls, name, sub)
            return sub

    class _StubBase(metaclass=_StubMeta):
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *args, **kwargs: None

    gi = types.ModuleType("gi")
    gi.require_version = lambda *args: None
    gi.require_foreign = lambda *args: True
    _repo = types.ModuleType("gi.repository")

    def _repo_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = _StubMeta(name, (_StubBase,), {})
        _repo.__dict__[name] = cls
        return cls

    _repo.__getattr__ = _repo_getattr
    gi.repository = _repo
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = _repo
    sys.modules.setdefault("cairo", types.ModuleType("cairo"))

from voxa.generation import Generation  # noqa: E402
from voxa.window import MainWindow  # noqa: E402


def _fake_window() -> MagicMock:
    """A stand-in self that records everything a completion touches."""
    window = MagicMock(name="window")
    window._gpu_generation = Generation()
    window._hardware_generation = Generation()
    window._install_generation = Generation()
    window._welcome_generation = Generation()
    window._whisper_generation = Generation()
    window._dictation_generation = Generation()
    window._model_manager_generation = Generation()
    window._models_generation = 0
    window._live_generation = 0
    window._closing = False
    # The reused guard must be the real one, not the mock that auto-appears.
    window._query_is_current = MainWindow._query_is_current.__get__(window)
    return window


# ---------------------------------------------------------------------------
# The helper itself.
# ---------------------------------------------------------------------------


def test_generation_tokens_retire_on_bump():
    generation = Generation()
    token = generation.value
    assert generation.is_current(token)
    assert generation.is_current(None)  # direct UI-thread calls are always current
    newer = generation.bump()
    assert newer != token
    assert not generation.is_current(token)
    assert generation.is_current(newer)


# ---------------------------------------------------------------------------
# Model list workers (models_generation).
# ---------------------------------------------------------------------------


def test_apply_ollama_models_stale_changes_nothing():
    window = _fake_window()
    window.ollama_models = ["keep"]
    window._apply_model_combo = MagicMock()
    assert MainWindow._apply_ollama_models(window, ["new"], 99) is False
    window._apply_model_combo.assert_not_called()
    assert window.ollama_models == ["keep"]
    # Current token still works (this is how existing tests call it without one).
    window._models_generation = 4
    window.settings.ai_backend = "ollama"
    window.settings.ollama_model = "keep"
    window.ai_server.health.return_value.status = "ready"
    assert MainWindow._apply_ollama_models(window, ["new"], 4) is False
    window._apply_model_combo.assert_called_once_with(["new"])
    assert MainWindow._apply_ollama_models(window, ["newer"]) is False


def test_backend_unreachable_stale_leaves_picker_alone():
    window = _fake_window()
    assert MainWindow._on_backend_unreachable(window, "Ollama", 99) is False
    window._set_status.assert_not_called()
    window._show_no_models.assert_not_called()
    assert MainWindow._on_backend_unreachable(window, "Ollama") is False
    window._set_status.assert_called_once()


def test_show_no_models_stale_leaves_picker_alone():
    window = _fake_window()
    window.ollama_models = ["keep"]
    assert MainWindow._show_no_models(window, "Ollama", 99) is False
    assert window.ollama_models == ["keep"]
    window._apply_model_combo.assert_not_called()
    assert MainWindow._show_no_models(window, "Ollama", 0) is False
    window._apply_model_combo.assert_called_once()


# ---------------------------------------------------------------------------
# Hardware / GPU workers.
# ---------------------------------------------------------------------------


def test_hardware_summary_stale_keeps_old_suggestions():
    window = _fake_window()
    window._hardware_summary = "old"
    window._suggested_models = ["old"]
    token = window._hardware_generation.bump()
    stale = token - 1
    assert MainWindow._apply_hardware_summary(window, 8.0, "RAM", [], 0.0, True, stale) is False
    assert window._hardware_summary == "old"
    assert window._suggested_models == ["old"]
    window._toast.assert_not_called()
    # Current token still shows the fit warning.
    assert MainWindow._apply_hardware_summary(window, 8.0, "RAM", [], 0.0, True, token) is False
    window._toast.assert_called_once()


def test_gpu_usage_stale_leaves_meters_alone():
    window = _fake_window()
    usage = types.SimpleNamespace(utilization_percent=42, memory_used_gb=1.0, memory_total_gb=8.0)
    token = window._gpu_generation.bump()
    assert MainWindow._apply_gpu_usage(window, usage, token - 1) is False
    window.gpu_level.set_value.assert_not_called()
    assert MainWindow._apply_gpu_usage(window, usage, token) is False
    window.gpu_level.set_value.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# Install / whisper / welcome workers.
# ---------------------------------------------------------------------------


def test_install_finished_stale_keeps_the_newer_run_owning_the_state():
    window = _fake_window()
    window._installing = "owned by the newer install"
    window.settings.ai_backend = "openai"
    window._welcome_followup = False
    dialog = MagicMock()
    token = window._install_generation.bump()
    assert MainWindow._on_install_finished(window, dialog, True, "done", token - 1) is False
    assert window._installing == "owned by the newer install"
    dialog.close.assert_not_called()
    window._toast.assert_not_called()
    assert MainWindow._on_install_finished(window, dialog, True, "done", token) is False
    assert window._installing is False
    window._toast.assert_called_once_with("done")
    # Old test-style call without a token still works.
    assert MainWindow._on_install_finished(window, dialog, False, "failed") is False


def test_whisper_ready_stale_never_saves_or_speaks():
    window = _fake_window()
    token = window._whisper_generation.bump()
    assert MainWindow._on_whisper_ready(window, "large-v3", "cpp", token - 1) is False
    window.config_store.save.assert_not_called()
    window._set_status.assert_not_called()
    assert MainWindow._on_whisper_error(window, "boom", token - 1) is False
    window._toast.assert_not_called()


def test_welcome_download_done_stale_only_clears_the_flag():
    window = _fake_window()
    window._welcome_downloading = True
    token = window._welcome_generation.bump()
    assert MainWindow._on_welcome_download_done(window, True, "done", token - 1) is False
    assert window._welcome_downloading is False
    window._toast.assert_not_called()
    window._refresh_ollama_models.assert_not_called()
    assert MainWindow._on_welcome_download_done(window, True, "done", token) is False
    window._toast.assert_called_once_with("done")
    window._refresh_ollama_models.assert_called_once()


def test_benchmark_choice_stale_never_changes_the_model():
    window = _fake_window()
    window.settings.whisper_model = "tiny"
    token = window._welcome_generation.bump()
    assert MainWindow._on_benchmark_choice(window, "base", 0.3, token - 1) is False
    assert window.settings.whisper_model == "tiny"
    window._load_whisper.assert_not_called()


# ---------------------------------------------------------------------------
# Model manager.
# ---------------------------------------------------------------------------


def test_model_deleted_stale_leaves_the_open_dialog_alone():
    window = _fake_window()
    dialog = MagicMock()
    token = window._model_manager_generation.bump()
    assert MainWindow._on_model_deleted(window, dialog, True, "deleted", token - 1) is False
    dialog.close.assert_not_called()
    window._toast.assert_not_called()
    window._show_model_manager.assert_not_called()
    assert MainWindow._on_model_deleted(window, dialog, True, "deleted", token) is False
    dialog.close.assert_called_once_with()
    window._show_model_manager.assert_called_once()


# ---------------------------------------------------------------------------
# Dictation (start_recording / external dictation).
# ---------------------------------------------------------------------------


def test_dictation_completions_stale_transcribe_and_note_nothing():
    window = _fake_window()
    token = window._dictation_generation.bump()
    stale = token - 1

    MainWindow._append_transcript(window, "hello", stale)
    window.transcript_view.get_buffer.assert_not_called()
    MainWindow._on_dictation_status(window, "Listening…", stale)
    window._set_status.assert_not_called()
    window.listening = True
    MainWindow._auto_stop_recording(window, stale)
    window.stop_recording.assert_not_called()
    MainWindow._capture_error(window, "mic died", stale)
    window._toast.assert_not_called()
    MainWindow._dictation_note(window, "I can't type here.", stale)
    window._toast.assert_not_called()

    # The current token still transcribes and toasts.
    buffer = MagicMock()
    buffer.get_char_count.return_value = 0
    window.transcript_view.get_buffer.return_value = buffer
    assert MainWindow._append_transcript(window, "hello", token) is False
    buffer.insert.assert_called_once_with(buffer.get_end_iter.return_value, "hello")
    assert MainWindow._dictation_note(window, "noted", token) is False
    window._toast.assert_called_once_with("noted")


def test_stopping_external_dictation_retires_later_toasts():
    window = _fake_window()
    captured = window._dictation_generation.value
    MainWindow._stop_external_dictation(window)
    assert not window._dictation_generation.is_current(captured)
    assert MainWindow._dictation_note(window, "late toast", captured) is False
    window._toast.assert_not_called()


# ---------------------------------------------------------------------------
# Reused query-generation guard (_query_is_current).
# ---------------------------------------------------------------------------


def test_suggestion_stale_never_interrupts_the_new_request():
    import threading

    from voxa.agent.suggest import Suggestion

    window = _fake_window()
    window.query_cancel = threading.Event()
    window._query_generation = 5
    window.conversation_active = False
    suggestion = Suggestion(key="copy", text="Try saying", kind="command")
    assert MainWindow._on_suggestion(window, suggestion, "heard", 4, window.query_cancel) is False
    window._toast.assert_not_called()
    assert MainWindow._on_suggestion(window, suggestion, "heard", 5, window.query_cancel) is False
    window._toast.assert_called_once_with("Try saying")
    # OFFLINE retires it as well (stop_current_work sets the event and bumps
    # the generation).
    window.query_cancel.set()
    window._query_generation += 1
    assert MainWindow._on_suggestion(window, suggestion, "heard", 5, window.query_cancel) is False
    window._toast.assert_called_once_with("Try saying")
    # Old call style (no tokens) still works.
    assert MainWindow._on_suggestion(window, suggestion, "heard") is False


def test_issue_flow_stale_never_asks_for_a_title():
    window = _fake_window()
    window.query_cancel = MagicMock()
    window._query_generation = 5
    assert MainWindow._begin_issue_flow(window, "voxa", "crhy/voxa", 4, window.query_cancel) is False
    window._on_tool_finished.assert_not_called()
    # Current token proceeds (repo unknown -> the "not found" question).
    assert MainWindow._begin_issue_flow(window, "voxa", None, 5, window.query_cancel) is False
    window._on_tool_finished.assert_called_once()
    # Old call style from the existing wiring test still works.
    assert MainWindow._begin_issue_flow(window, "voxa", "crhy/voxa") is False


def test_website_search_stale_never_opens_the_page():
    # the flow lives in the query controller now (g138d): drive it with a live
    # open_stored_website and fake hooks, so the token check itself is tested.
    from voxa.query_controller import QueryController

    opened: list[str] = []
    toasted: list[str] = []
    cancel = MagicMock()
    fake = QueryController.__new__(QueryController)
    fake.last_site_url = (time.time(), "https://example.com/page")
    fake._ui = types.SimpleNamespace(
        current=lambda generation, cancel_event: generation == 5,
        toast=toasted.append,
        open_url=lambda url, original: opened.append(url),
    )
    assert QueryController.open_stored_website(fake, "site", "open it", 4, cancel) is None
    assert opened == []
    assert toasted == []
    assert QueryController.open_stored_website(fake, "site", "open it", 5, cancel) is None
    assert opened == ["https://example.com"]


# ---------------------------------------------------------------------------
# Live face.
# ---------------------------------------------------------------------------


def test_recover_stalled_live_face_stale_leaves_the_new_renderer_alone():
    window = _fake_window()
    client = MagicMock()
    window._live_client = client
    window._live_generation = 7
    assert MainWindow._recover_stalled_live_face(window, client, 6) is False
    window._toast.assert_not_called()
    window.shell.show_live_frames.assert_not_called()
    assert MainWindow._recover_stalled_live_face(window, client, 7) is False
    assert window._live_generation == 8
    window._toast.assert_called_once()


def test_face_manifest_failure_stale_never_downgrades_the_face():
    window = _fake_window()
    assert MainWindow._on_face_manifest_failed(window, 99) is False
    window._toast.assert_not_called()
    window._go_back_to_medium.assert_not_called()
    assert MainWindow._on_face_manifest_failed(window, window._live_generation) is False
    window._go_back_to_medium.assert_called_once()


# ---------------------------------------------------------------------------
# OFFLINE is the mechanical kill switch (effects tested on fake and real
# windows; the full ACTIVE/OFFLINE matrix lives in test_window_offline.py).
# ---------------------------------------------------------------------------


def test_offline_stops_capture_retires_dictation_and_cancels_welcome(monkeypatch):
    import threading

    import voxa.window as window_module

    class _NoSpin:
        stopped = False

        def stop_spin(self) -> None:
            type(self).stopped = True

    monkeypatch.setattr(window_module, "compiz", _NoSpin())
    window = _fake_window()
    welcome_cancel = threading.Event()
    window._welcome_cancel = welcome_cancel
    window.listening = True
    window._installing = False

    MainWindow.stop_current_work(window)

    window.assistant.go_offline.assert_called_once_with()
    window.stop_recording.assert_called_once_with()  # microphone capture stopped
    window._stop_external_dictation.assert_called_once_with()
    window._stop_ai_server_async.assert_called_once_with()
    window._live_reset.assert_called_once_with()
    assert welcome_cancel.is_set()  # a welcome download stops at once
    assert _NoSpin.stopped
    # And the dictation generation was retired, so late dictation toasts no-op:
    stale = window._dictation_generation.value - 1
    assert not window._dictation_generation.is_current(stale - 1)


# The real-window OFFLINE matrix (microphone, wake monitor, speech, late
# callbacks) lives in tests/test_window_offline.py; the Dictate-session effect
# of stop_current_work is tested there too, because constructing a real
# MainWindow changes the global CSS metrics and would skew the layout tests
# that run between this file and that one.
