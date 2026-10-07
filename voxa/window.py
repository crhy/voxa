from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

from . import apps, documents, mail, websearch, welcome  # noqa: E402
from .agent import hearing, host, intents, issueflow, mailflow, planner, repeat, ui  # noqa: E402
from .agent.actionlog import ActionLog, ActionRecord  # noqa: E402
from .agent.claims import claims_action, first_sentences  # noqa: E402
from .agent.host import host_command  # noqa: E402
from .agent.host import spawn as host_spawn  # noqa: E402
from .agent.player import active_player  # noqa: E402
from .agent.progress import working_caption  # noqa: E402
from .agent.registry import ToolError  # noqa: E402
from .agent.reminders import ReminderStore, reminder_phrase, timer_phrase  # noqa: E402
from .agent.result import ToolResult  # noqa: E402
from .agent.routines import Routine, RoutineStore, parse_create, parse_delete, parse_list  # noqa: E402
from .agent.spoken_text import format_dictation, parse_dictation_control  # noqa: E402
from .agent.suggest import Suggestion, SuggestionState, suggest  # noqa: E402
from .agent.tools import contacts, default_registry, github, media, textedit, typing, windows  # noqa: E402
from .agent.tools.web import get_session  # noqa: E402
from .anc import EchoCanceller as AncCanceller  # noqa: E402
from .audio import AudioCapture, AudioDevice  # noqa: E402
from .bargein import BargeInGate, is_own_voice  # noqa: E402
from .catalog import CatalogUnavailable, load_catalog, refresh_and_cache, refresh_due  # noqa: E402
from .config import ConfigStore  # noqa: E402
from .controller import AssistantController, ControllerPorts  # noqa: E402
from .conversation import ConversationController, ConversationHistory, strip_wake_word  # noqa: E402
from .dictation import DictationController  # noqa: E402
from .echo import (  # noqa: E402
    default_monitor_source,
    remove_leftover_devices,
)
from .hardware import (  # noqa: E402
    MODEL_CATALOG,
    GpuUsage,
    detect_available_model_memory_gb,
    reserved_gpu_gb,
    sample_gpu_usage,
    suggest_models,
)
from .installer import InstallerError, install_ollama, ollama_installed  # noqa: E402
from .llamacpp import LlamaCppClient, StrataClient  # noqa: E402
from .modelwait import wait_for_models  # noqa: E402
from .ollama import OllamaClient, OllamaError, strip_reasoning  # noqa: E402
from .pausewords import is_pause_request, resume_request  # noqa: E402
from .replylang import LANGUAGES, is_only_a_language_request, requested_language  # noqa: E402
from .server import SERVER_FAILED, SERVER_STARTING, SERVER_UNAVAILABLE, AiServerManager  # noqa: E402
from .speakstream import SentenceFeeder, is_thinking_model, looks_like_reasoning  # noqa: E402
from .speech import SpeechService  # noqa: E402
from .theme import host_theme_is_dark  # noqa: E402
from .transcription import WhisperService  # noqa: E402
from .ui import clipboard  # noqa: E402
from .ui.avatars import character_choices, get_avatar  # noqa: E402
from .ui.focus_window import overlay_supported  # noqa: E402
from .ui.legacy_view import LegacyCallbacks, LegacyView  # noqa: E402
from .ui.live_face import LiveFaceClient, decode_audio_to_pcm16k  # noqa: E402
from .ui.shell import AssistantShell, build_header  # noqa: E402
from .ui.state import AssistantModel, AssistantState  # noqa: E402
from .ui.styles import install_styles  # noqa: E402
from .ui.welcome_dialog import WelcomeDialog  # noqa: E402

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
# Conversation mode's wake-word phase runs continuously in the background, so
# it always uses this small model instead of whichever (possibly much
# larger) model the user picked for real dictation — that one only has to
# run once per turn, after the wake word is actually heard.
WAKE_WHISPER_MODEL = "tiny"
GPU_POLL_INTERVAL_SECONDS = 2.0
APPEARANCE_VALUES = ["system", "light", "dark"]
APPEARANCE_LABELS = ["System", "Light", "Dark"]
WEB_SEARCH_VALUES = ["auto", "always", "never"]
WEB_SEARCH_LABELS = ["Automatically when needed", "Always search", "Never search"]
TTS_VOICES = [
    ("Aria — US female", "en-US-AriaNeural"),
    ("Jenny — US female", "en-US-JennyNeural"),
    ("Guy — US male", "en-US-GuyNeural"),
    ("Connor — Ireland male", "en-IE-ConnorNeural"),
    ("Emily — Ireland female", "en-IE-EmilyNeural"),
    ("Ryan — UK male", "en-GB-RyanNeural"),
    ("Sonia — UK female", "en-GB-SoniaNeural"),
    ("Jorge — Mexico male", "es-MX-JorgeNeural"),
    ("Dalia — Mexico female", "es-MX-DaliaNeural"),
    ("Henri — France male", "fr-FR-HenriNeural"),
    ("Denise — France female", "fr-FR-DeniseNeural"),
    ("Conrad — Germany male", "de-DE-ConradNeural"),
    ("Katja — Germany female", "de-DE-KatjaNeural"),
    ("Keita — Japan male", "ja-JP-KeitaNeural"),
    ("Nanami — Japan female", "ja-JP-NanamiNeural"),
]
# A few turns of history keeps the model aware of what was just said without
# letting a long hands-free session grow the prompt unboundedly.
CONVERSATION_HISTORY_MESSAGES = 24
CONVERSATION_SYSTEM_PROMPT = (
    "You are Voxa, a hands-free voice assistant on the user's desktop. Keep "
    "answers short and conversational — one or two sentences — since they are spoken aloud. "
    "You cannot perform actions in this reply: never say you opened, closed, sent, played, "
    "saved, deleted, booked or changed anything. If the user asked for an action you cannot "
    "perform, say you could not do it and suggest how to phrase it as a direct command."
)
FALSE_CLAIM_REPLY = (
    "I couldn't do that. Try saying it as a direct command, like “close Brutal Chess”."
)
# Barge-in: loud sustained speech while a reply is being read interrupts it.
# A short grace period ignores the TTS itself starting (speaker onset is the
# loudest moment); the level gate in .bargein decides when the mic is loud
# enough, and the words heard decide whether it was really the user.
BARGE_IN_GRACE_SECONDS = 0.4
# How long the high-resolution face server may take to load its model before Voxa gives up on it.
FACE_SERVER_START_SECONDS = 90.0


def face_server_command() -> list[str] | None:
    """The command that starts the face server on the host, or None when it is not installed.

    $VOXA_FACE_SERVER overrides the search. The script is looked for on the HOST (Voxa may be sandboxed).
    """
    override = os.environ.get("VOXA_FACE_SERVER", "").strip()
    home = os.path.expanduser("~")
    candidates = [override] if override else [
        f"{home}/.local/share/voxa/face-server/run-face-server.sh",
        f"{home}/voxa-neural/run-face-server.sh",
    ]
    for path in candidates:
        try:
            found = subprocess.run(host_command(["test", "-x", path]), check=False, timeout=5).returncode == 0
        except (OSError, subprocess.SubprocessError):
            found = False
        if found:
            return [path, "--steps", "10"]
    return None


# High facial quality: how much video must be buffered before the voice starts, and the longest it may wait.
LIVE_FACE_HEAD_START = 0.2
LIVE_FACE_MAX_HOLD = 2.5


# How long the model list waits for an AI server that Voxa has only just started.
MODEL_WAIT_ATTEMPTS = 30
MODEL_WAIT_DELAY = 1.0


def idle(callback: Callable, *args) -> None:
    # Default priority, not idle priority: an animating widget can keep the frame clock busy enough that
    # idle-priority callbacks never run, which left the model list and pull progress stuck.
    # Always one-shot, whatever the callback returns: a repeating default-priority source would starve drawing.
    def run_once() -> bool:
        callback(*args)
        return False

    GLib.idle_add(run_once, priority=GLib.PRIORITY_DEFAULT)


def short_model_name(model: str) -> str:
    """The part after the last '/', truncated to 28 characters with an ellipsis."""
    name = model.rsplit("/", 1)[-1]
    if len(name) > 28:
        return name[:27] + "…"
    return name


def string_item_factory(*, wrap: bool, width_chars: int) -> Gtk.SignalListItemFactory:
    """Create readable labels for long Gtk.StringList entries."""
    factory = Gtk.SignalListItemFactory()

    def setup(_factory, list_item) -> None:
        label = Gtk.Label(xalign=0)
        label.set_hexpand(True)
        label.set_halign(Gtk.Align.FILL)
        label.set_width_chars(width_chars)
        label.set_max_width_chars(90)
        label.set_wrap(wrap)
        if wrap:
            label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        else:
            label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        list_item.set_child(label)

    def bind(_factory, list_item) -> None:
        item = list_item.get_item()
        label = list_item.get_child()
        text = item.get_string() if item is not None else ""
        label.set_text(text)
        label.set_tooltip_text(text)

    factory.connect("setup", setup)
    factory.connect("bind", bind)
    return factory


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, application: Adw.Application) -> None:
        super().__init__(application=application)
        self.set_title("Voxa")
        self.set_default_size(1200, 760)
        self.set_size_request(850, 600)

        self.config_store = ConfigStore()
        self.settings = self.config_store.load()
        server_logs = Path(GLib.get_user_cache_dir()) / "voxa" / "server"
        self.ai_server = AiServerManager(self.settings, log_dir=server_logs)
        self._server_action_lock = threading.Lock()
        self._server_generation = 0
        self._last_search_query: str | None = None
        self._last_search_at = 0.0
        self._search_record: dict | None = None
        self._search_source = ""
        self.style_manager = Adw.StyleManager.get_default()
        self._apply_appearance()
        self.whisper = WhisperService()
        self.wake_whisper = WhisperService()
        self.audio = AudioCapture()
        self.speech = SpeechService()
        self.dictation: DictationController | None = None
        self.listening = False
        self.conversation: ConversationController | None = None
        self.conversation_active = False
        # Earlier turns of the current hands-free conversation, sent to the
        # model as a chat message list so it can follow the thread.
        self._conversation_history = ConversationHistory(
            CONVERSATION_SYSTEM_PROMPT, CONVERSATION_HISTORY_MESSAGES
        )
        # Barge-in bookkeeping: when the reply started being spoken (0 = not
        # speaking), the level gate that decides when the mic is loud enough,
        # and the text being spoken now / the piece before it (own-voice check).
        self._speaking_since = 0.0
        self._barge_gate = BargeInGate()
        self._speaking_text = ""
        self._recent_spoken = ""
        # The conversation turn whose user message was just pushed, so a stale
        # finish/error callback can't untangle history built by a newer turn.
        self._pending_user_generation: int | None = None
        # Language the current session's answers must be in (None = the
        # character's own language); set by a "…in German" style request.
        self._reply_language: str | None = None
        # The last question actually asked, so a bare "in German" can re-answer it.
        self._last_user_prompt: str | None = None
        self.query_cancel = threading.Event()
        self.devices: list[AudioDevice] = []
        self.ollama_models: list[str] = []
        self._progress_source = 0
        self._level_source = 0
        self._latest_level = 0.0
        self._query_generation = 0
        self._loading_model_since: float | None = None
        self._loading_model_name: str = ""
        self._loading_source: int = 0
        self._loading_generation: int = -1
        self._loading_cancel: threading.Event | None = None
        self._hardware_summary = "Detecting your hardware…"
        self._suggested_models: list[str] = []
        self._install_cancel = threading.Event()
        self._installing = False
        self._has_gpu = False
        self._gpu_poll_stop: threading.Event | None = None
        self._model_combo_updating = False
        # Set when the window starts closing, so callbacks that arrive afterwards do nothing.
        self._closing = False
        self._follow_up_token: int | None = None
        self._app_cache: list[apps.DesktopApp] | None = None
        # Tools the router can call directly, without ever asking the model.
        self.tools = default_registry()
        windows.ON_LOCK = self._pause_now
        textedit.ask_model = lambda messages: self._ai_client().generate_stream(model=self.settings.ollama_model, prompt=messages[-1]["content"], cancel_event=threading.Event(), on_chunk=lambda chunk: None, messages=messages)
        media.get_my_channel = lambda: self.settings.youtube_channel
        media.set_my_channel = self._save_youtube_channel
        github.set_owner = self._save_github_owner
        ui.say = lambda text: idle(self._say_notice, text)
        contacts.get_contacts = lambda: self.settings.contacts
        contacts.save_contacts = self._save_contacts
        textedit.read_clipboard = clipboard.read_text
        textedit.write_clipboard = clipboard.write_text
        # Every command Voxa hears is logged here, so we can see what works.
        self.action_log = ActionLog()
        # True once a tool or app has run for the request being answered; the
        # ordinary chat path uses it to avoid rewriting a reply that followed a
        # real action.
        self._tool_ran_for_request = False
        # Coaching suggestions: remembers what was offered and dismissed.
        self.suggestion_state = SuggestionState(self.action_log.path.parent / "suggestions.json")
        # Timers and reminders: kept in a JSON file next to the action log so they
        # survive a restart; a 5-second tick delivers the ones whose time has come.
        self.reminder_store = ReminderStore(self.action_log.path.parent / "reminders.json")
        self._reminders_source = GLib.timeout_add_seconds(5, self._reminders_tick)
        # Routines: one phrase runs several commands, kept in a JSON file next to
        # the action log. A matched routine queues its steps; each runs through the
        # normal router, drained one at a time as the previous step finishes.
        self.routine_store = RoutineStore(self.action_log.path.parent / "routines.json")
        self._routine_queue: list[str] = []
        self._routine_active = False
        # Dictation mode: utterances go straight into the focused window instead
        # of the model, until the user says "stop dictating".
        self._external_dictation = False
        self._issue_flow: issueflow.IssueFlow | None = None
        self._mail_flow: mailflow.MailFlow | None = None
        self._last_tool_call = None
        self._welcome_followup = False
        self._last_dictated = ""
        # The single source of truth for what the assistant is doing; the shell renders it.
        self.assistant_model = AssistantModel()
        # ACTIVE / OFFLINE and every stale-callback decision go through this controller.
        self.assistant = AssistantController(
            self.assistant_model,
            ControllerPorts(
                start_listening=self._port_start_listening,
                stop_listening=self._port_stop_listening,
                stop_speech=self._port_stop_speech,
                cancel_inference=self._port_cancel_inference,
                microphone_active=lambda: self.audio.is_active,
            ),
        )
        self._start_failure = ""
        self._query_task_id: str | None = None
        self._announced_ready = False
        self._activate_when_ready = False
        install_styles()

        self._build_ui()
        self._install_actions()
        self._refresh_devices()
        self._refresh_ollama_models()
        self._detect_hardware_async()
        self._load_whisper(self.settings.whisper_model)
        self._load_wake_whisper()
        self._start_ai_server_async()

    def _save_youtube_channel(self, handle: str) -> None:
        self.settings.youtube_channel = handle
        self.config_store.save(self.settings)

    def _save_github_owner(self, owner: str) -> None:
        self.settings.github_owner = owner
        self.config_store.save(self.settings)

    def _save_contacts(self, contacts: dict[str, str]) -> None:
        self.settings.contacts = contacts
        self.config_store.save(self.settings)

    def _load_wake_whisper(self) -> None:
        # Runs quietly in the background: conversation mode falls back to the
        # main model (see start_conversation_mode) if this isn't ready yet,
        # so a slow or failed load here should never block anything.
        self.wake_whisper.load_async(
            WAKE_WHISPER_MODEL,
            lambda _name, _backend: None,
            lambda error: idle(self._toast, f"Wake-word model failed to load: {error}"),
        )

    def _build_ui(self) -> None:
        self.toast_overlay = Adw.ToastOverlay()
        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)

        menu = Gio.Menu()
        menu.append("Preferences", "win.preferences")
        menu.append("Transcript and dictation…", "win.transcript")
        menu.append("Keyboard Shortcuts", "win.shortcuts")
        menu.append("About Voxa", "app.about")
        # The assistant shell is the production interface. Everything it shows is
        # rendered from self.assistant_model; the window only wires callbacks.
        self.shell = AssistantShell(self.assistant_model, self.settings)
        self.shell.on_active = self._on_shell_active
        self.shell.on_offline = self._on_shell_offline
        self.shell.on_pause = self._on_shell_pause
        self.shell.on_model_selected = self._on_shell_model_selected
        self.shell.on_backend_selected = self._on_shell_backend_selected
        self.shell.on_character_selected = self._on_shell_character_selected
        self.shell.on_face_mode_selected = self._on_shell_face_mode_selected
        self.shell.set_characters(self.settings.character_id)
        if overlay_supported():
            self.shell.enable_focus_window(self.present)
        self.shell.face_quality.set_mode(self.settings.face_mode)
        self.shell.face_quality.set_sensitive(bool(self.settings.character_id))
        self._live_client: LiveFaceClient | None = None
        GLib.timeout_add(250, self._sync_listening_caption)
        GLib.timeout_add(500, self._sync_window_focus)
        # High is offered only when the face server answers; the check runs off the GTK thread.
        self._refresh_live_available()
        toolbar.add_top_bar(build_header(menu, self.shell.character_picker))
        # Attachments are a later milestone (issue #7 section 16); until then the paperclip
        # says so instead of silently doing nothing.
        self.shell.attachment_button.set_sensitive(False)
        self.shell.attachment_button.set_tooltip_text("Attaching files is coming in a later update")
        self.toast_overlay.set_child(self.shell)
        toolbar.set_content(self.toast_overlay)

        self._legacy_window: Adw.Window | None = None
        self._build_legacy_ui()

    def _build_legacy_ui(self) -> None:
        """The original dictation/transcript interface, kept fully working.

        It is built but not attached to the main window: the new assistant shell is
        the production view. It is reachable from the menu ("Transcript and
        dictation...") and through the existing keyboard shortcuts, in a secondary
        window, so no capability was lost. The view itself lives in
        :class:`voxa.ui.legacy_view.LegacyView`; every widget it builds is aliased
        back onto this window under the name the rest of the code already uses.
        """
        legacy = LegacyView(
            LegacyCallbacks(
                record_toggled=self._on_record_toggled,
                conversation_toggled=self._on_conversation_toggled,
                copy=self.copy_transcript,
                copy_reply=self.copy_response,
                clear=self.clear_all,
                ask=self.ask_ai,
                speak=self.speak_response,
                stop=self.stop_current_work,
                model_selected=self._on_model_selected,
            )
        )
        self._legacy_view = legacy.view
        for name in LegacyView.WIDGET_NAMES:
            setattr(self, name, getattr(legacy, name))

    def _install_actions(self) -> None:
        actions = {
            "preferences": self.show_preferences,
            "transcript": self.show_transcript_window,
            "shortcuts": self.show_shortcuts,
            "record": self.toggle_recording,
            "conversation": self.toggle_conversation,
            "ask": self.ask_ai,
            "copy": self.copy_transcript,
            "copy-response": self.copy_response,
            "clear": self.clear_all,
        }
        for name, callback in actions.items():
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda _a, _p, cb=callback: cb())
            self.add_action(action)
        app = self.get_application()
        app.set_accels_for_action("win.record", ["<Control>r"])
        app.set_accels_for_action("win.conversation", ["<Control><Shift>r"])
        app.set_accels_for_action("win.ask", ["<Control>Return"])
        app.set_accels_for_action("win.copy", ["<Control><Shift>c"])
        app.set_accels_for_action("win.copy-response", ["<Control><Shift>v"])
        app.set_accels_for_action("win.clear", ["<Control>l"])
        app.set_accels_for_action("win.preferences", ["<Control>comma"])

    # ---------------------------------------------- the assistant shell's controls

    def _on_shell_active(self) -> None:
        self._start_ai_server_async()
        self._activate_when_ready = False
        if self.assistant.activate():
            return
        if self._activate_when_ready:
            # Whisper is still loading: keep waiting and say so under the face (no toast).
            self.assistant_model.set_state(
                AssistantState.OFFLINE,
                "Getting ready… — Loading the speech model — I'll start listening as soon as it's ready",
            )
        elif self._start_failure:
            self.assistant_model.set_state(
                AssistantState.OFFLINE,
                f"Can't start listening — {self._start_failure}",
            )

    def _on_shell_offline(self) -> None:
        self._activate_when_ready = False
        self.stop_current_work()

    def _on_shell_pause(self) -> None:
        if self.assistant.is_paused:
            self.assistant.resume()
        elif self.assistant.is_active:
            self._pause_now()

    def _sync_listening_caption(self) -> bool:
        """Keep "Ready" and "Listening" truthful. Runs four times a second.

        "Listening" is shown exactly while the conversation pipeline is capturing a request (after the wake
        word, or in the follow-up window after a reply) and the microphone is not muted; otherwise "Ready".
        Other states (Thinking, Speaking, Working, Paused, Offline) are left alone.
        """
        if self._closing:
            return False
        conversation = self.conversation
        state = self.assistant_model.state
        if conversation is None or state not in (AssistantState.READY, AssistantState.LISTENING):
            return True
        capturing = bool(conversation.waiting_for_prompt or getattr(conversation, "keep_prompt", False))
        truly_listening = capturing and not conversation.muted
        wanted = AssistantState.LISTENING if truly_listening else AssistantState.READY
        if wanted is not state:
            self.assistant_model.set_state(wanted, "")
        return True

    def _sync_window_focus(self) -> bool:
        """Show the focus pop-up whenever the window manager reports the window unfocused."""
        if self._closing:
            return False
        popup = self.shell.focus_window
        focused = any(w.is_active() for w in Gtk.Window.list_toplevels() if w is not popup)
        self.shell.set_window_focus(focused)
        return True

    def _pause_now(self) -> None:
        """Pause: drop the answer in progress and stop talking, but keep listening for the wake word."""
        self.query_cancel.set()          # a reply still being written must not be spoken after the pause
        self._early = None
        if self.conversation is not None:
            self.conversation.unmute()   # speech may have muted the microphone: the wake word must get through
        self._speaking_since = 0.0
        self.assistant.pause()
        self._set_status(f"Paused — say “{self.settings.wake_word}” to continue")

    def _queue_ai_server_action(self, action: str) -> None:
        """Serialize managed-server changes without ever blocking GTK's main loop."""
        self._server_generation += 1
        generation = self._server_generation

        def work() -> None:
            with self._server_action_lock:
                if generation != self._server_generation:
                    return
                if action == "stop":
                    self.ai_server.stop()
                    return
                if action == "restart":
                    self.ai_server.restart()
                else:
                    self.ai_server.start()

            if self.ai_server.health().status == SERVER_STARTING:
                self.ai_server.wait_until_ready(cancelled=lambda: generation != self._server_generation)
            if generation == self._server_generation and not self._closing:
                idle(self._report_ai_server_health, generation)

        # A stop worker must keep the process alive long enough to reap a managed
        # child when the window is closing.  Start/restart workers remain daemon
        # threads so a slow readiness probe cannot delay application shutdown.
        threading.Thread(
            target=work,
            name=f"voxa-ai-server-{action}",
            daemon=action != "stop",
        ).start()

    def _report_ai_server_health(self, generation: int) -> bool:
        if generation != self._server_generation or self._closing:
            return False
        health = self.ai_server.health()
        if health.status in {SERVER_FAILED, SERVER_UNAVAILABLE} and health.last_error:
            self._toast(health.last_error.splitlines()[0])
        return False

    def _start_ai_server_async(self) -> None:
        self._queue_ai_server_action("start")

    def _restart_ai_server_async(self) -> None:
        self._queue_ai_server_action("restart")

    def _stop_ai_server_async(self) -> None:
        self._queue_ai_server_action("stop")

    # The controller's ports: the real services behind ACTIVE and OFFLINE. Each is
    # idempotent, and the controller calls all of them even if one raises.

    def _port_start_listening(self) -> None:
        self._start_failure = ""
        if self.listening:
            self.stop_recording()
        if not self.start_conversation_mode():
            raise RuntimeError(self._start_failure or "Could not start listening")

    def _port_stop_listening(self) -> None:
        if self.listening:
            self.stop_recording()
        if self.conversation_active or self.conversation is not None:
            self.stop_conversation_mode()
        self.audio.stop()

    def _port_stop_speech(self) -> None:
        self.speech.stop()
        self._live_reset()
        self._speaking_since = 0.0
        self._barge_gate.reset()
        if self.conversation is not None:
            self.conversation.unmute()

    def _port_cancel_inference(self) -> None:
        if self._pending_user_generation is not None:
            self._conversation_history.drop_last()
        self._pending_user_generation = None
        self.query_cancel.set()
        self._query_generation += 1
        self._query_task_id = None  # the controller cancels the task itself
        self.ask_button.set_sensitive(True)

    def _for_session(self, token: int, callback: Callable) -> Callable:
        """Wrap a queued callback so it is dropped if Voxa went OFFLINE (or was re-activated) since.

        Microphone, transcription, model and speech callbacks are queued with GLib.idle_add
        from worker threads; without this a callback queued just before OFFLINE could still
        run afterwards and restart work.
        """

        def run(*args) -> bool:
            if self._closing or not self.assistant.accepts(token):
                return False
            callback(*args)
            return False

        return run

    def _fail_assistant(self, detail: str) -> None:
        """A global assistant failure: show ERROR briefly, then return to READY."""
        token = self.assistant.token()
        if self.assistant.failed(token, detail):
            GLib.timeout_add_seconds(4, self._recover_assistant, token)

    def _recover_assistant(self, token: int) -> bool:
        self.assistant.recover(token)
        return False

    def _end_query_task(self, outcome: str, detail: str = "") -> None:
        task_id, self._query_task_id = self._query_task_id, None
        if task_id is None:
            return
        try:
            if outcome == "done":
                self.assistant_model.complete_task(task_id)
            elif outcome == "failed":
                self.assistant_model.fail_task(task_id, detail or "The request failed.")
            else:
                self.assistant_model.cancel_task(task_id)
        except (KeyError, ValueError):  # already finished or cancelled (for example by OFFLINE)
            pass

    def _on_shell_model_selected(self, name: str) -> None:
        if name and name != self.settings.ollama_model:
            self.settings.ollama_model = name
            self.config_store.save(self.settings)
            self._apply_model_combo(self.ollama_models)  # keep the legacy dropdown in step
            self._warm_up_model(name)

    def _on_shell_backend_selected(self, backend: str) -> None:
        if backend != self.settings.ai_backend:
            self.settings.ai_backend = backend
            self.config_store.save(self.settings)
            self.shell.set_backend(backend)
            self._refresh_ollama_models()
            if backend != "strata":  # Voxa does not manage the Strata server
                self._restart_ai_server_async()

    def _on_shell_character_selected(self, character_id: str) -> None:
        if character_id == self.settings.character_id:
            return
        self.settings.character_id = character_id
        avatar = get_avatar(character_id)
        if avatar is not None:
            self.settings.tts_voice = avatar.voice
        self._reply_language = None
        self.config_store.save(self.settings)
        assistant_view = getattr(self.shell, "assistant_view", None)
        if assistant_view is not None:
            assistant_view.set_character(self.settings.character_id)
        self.shell.set_characters(self.settings.character_id)
        self.shell.face_quality.set_sensitive(bool(self.settings.character_id))

    def _on_shell_face_mode_selected(self, face_mode: str) -> None:
        if face_mode == self.settings.face_mode:
            return
        self.settings.face_mode = face_mode
        self.config_store.save(self.settings)
        apply = getattr(self, "_apply_face_mode", None)
        if apply is not None:
            apply()

    def _apply_face_mode(self) -> None:
        """Push the saved face mode into the assistant view."""
        assistant_view = getattr(self.shell, "assistant_view", None)
        if assistant_view is not None:
            assistant_view.set_face_mode(self.settings.face_mode)
        self._refresh_live_available()
        # The face server's share of the card changes what fits, so the
        # suggestions are re-run for the new mode.
        detect = getattr(self, "_detect_hardware_async", None)
        if detect is not None:
            detect()

    def _refresh_live_available(self) -> None:
        """High is always selectable. When it is the chosen quality, make sure the face server is running
        (starting it if needed) and say what is happening; otherwise let go of a server Voxa started."""
        face_quality = getattr(self.shell, "face_quality", None)
        if face_quality is not None:
            face_quality.set_live_available(True)
        if self.settings.face_mode != "live" or not self.settings.character_id:
            # Leaving High cancels a start that is still in progress.
            self._live_generation = getattr(self, "_live_generation", 0) + 1
            self._live_starting = False
            self._release_face_server()
            if face_quality is not None:
                face_quality.set_status("")
            return
        if self._live_client is not None and self._live_client.available():
            if face_quality is not None:
                face_quality.set_status("")
            return
        if face_quality is not None:
            face_quality.set_status("Loading the high-resolution face…")
        if getattr(self, "_live_starting", False):
            return  # one start at a time: several things ask for this while the window is being built
        self._live_starting = True
        generation = getattr(self, "_live_generation", 0)

        def worker() -> None:
            client = LiveFaceClient()
            started = False
            if not client.available():
                command = face_server_command()
                if command is None:
                    idle(self._on_live_ready, generation, None, False,
                         "The high-resolution face is not installed on this computer")
                    return
                host_spawn(command)
                started = True
                deadline = time.monotonic() + FACE_SERVER_START_SECONDS
                while time.monotonic() < deadline and generation == getattr(self, "_live_generation", 0):
                    if client.available():
                        break
                    time.sleep(1.0)
            if generation != getattr(self, "_live_generation", 0):
                if started and client.available():
                    client.shutdown_server()
                return
            if client.available():
                idle(self._on_live_ready, generation, client, started, "")
            else:
                idle(self._on_live_ready, generation, None, started,
                     "The high-resolution face could not start (is the graphics card full?)")

        threading.Thread(target=worker, name="live-face-start", daemon=True).start()

    def _on_live_ready(self, generation: int, client: LiveFaceClient | None, started: bool, problem: str) -> None:
        if generation != getattr(self, "_live_generation", 0):
            return
        self._live_starting = False
        face_quality = getattr(self.shell, "face_quality", None)
        self._live_client = client
        self._face_server_started = bool(started and client is not None)
        if face_quality is not None:
            face_quality.set_status("")
        if client is not None:
            self._toast("High-resolution face ready.")
            return
        # Could not start: say why and go back to Medium rather than leave High selected and doing nothing.
        self._toast(f"{problem}. Using Medium instead.")
        self.settings.face_mode = "prerendered"
        self.config_store.save(self.settings)
        if face_quality is not None:
            face_quality.set_mode("prerendered")
        assistant_view = getattr(self.shell, "assistant_view", None)
        if assistant_view is not None:
            assistant_view.set_face_mode("prerendered")

    def _prepare_microphone(self) -> None:
        """Before listening: choose how the computer's own sound is removed from the microphone."""
        if not getattr(self, "_echo_leftovers_checked", False):
            self._echo_leftovers_checked = True
            threading.Thread(target=remove_leftover_devices, name="echo-cleanup", daemon=True).start()
        mode = getattr(self.settings, "echo_mode", "voxa")
        if mode == "voxa":
            # Our canceller: record the loopback monitor as the reference and clean the mic against it.
            self.audio.reference_source = default_monitor_source()
            self.audio.pulse_source = None
            self.audio.canceller = AncCanceller()
        else:
            self.audio.pulse_source = None
            self.audio.reference_source = None
            self.audio.canceller = None

    def _release_face_server(self) -> None:
        """Drop the live client; stop the face server if Voxa was the one that started it (frees the GPU)."""
        client, self._live_client = self._live_client, None
        if client is None:
            return
        if getattr(self, "_face_server_started", False):
            self._face_server_started = False
            threading.Thread(target=client.shutdown_server, name="live-face-stop", daemon=True).start()
        else:
            client.close()

    def _live_before_play(self, path: str) -> None:
        """Called in the synthesis worker thread before playback: feed the live face server."""
        if self._live_client is None:
            return
        try:
            pcm = decode_audio_to_pcm16k(path)
        except Exception:
            return
        buffer = self._live_client.start_utterance(
            self.settings.character_id or "", pcm, size=512
        )
        if buffer is None:
            return
        renderer = getattr(self.shell, "assistant_view", None)
        if renderer is not None:
            photo = getattr(renderer, "_photo_renderer", None)
            if photo is not None:
                idle(photo.set_live_frames, buffer)
        # Hold the voice until the first frames are in, so face and sound start together. The server makes
        # frames faster than they are played, so a small head start is enough; never wait long.
        deadline = time.monotonic() + LIVE_FACE_MAX_HOLD

        def flag(value) -> bool:
            return bool(value() if callable(value) else value)

        while time.monotonic() < deadline:
            if buffer.buffered_seconds() >= LIVE_FACE_HEAD_START or flag(buffer.done) or flag(buffer.failed):
                break
            time.sleep(0.02)

    def _live_prepare(self, item) -> None:
        """Worker thread, as soon as a piece's audio exists: start the neural face on it ahead of time."""
        client = self._live_client
        if client is None or self.settings.face_mode != "live":
            return
        pcm = decode_audio_to_pcm16k(item.path)
        item.extra = client.start_utterance(self.settings.character_id or "", pcm, size=512)

    def _live_activate(self, item) -> None:
        """Worker thread, just before a prepared piece plays: show its frames and hold the voice briefly
        if they are not in yet (they usually are: the piece was prepared while the previous one played)."""
        if item.extra is None:
            try:
                self._live_prepare(item)
            except Exception:  # noqa: BLE001 - no neural face for this piece: the pre-rendered mouth is used
                item.extra = None
        buffer = item.extra
        view = getattr(self.shell, "assistant_view", None)
        photo = getattr(view, "_photo_renderer", None) if view is not None else None
        if photo is not None:
            idle(photo.set_live_frames, buffer)
        if buffer is None:
            return
        deadline = time.monotonic() + LIVE_FACE_MAX_HOLD

        def flag(value) -> bool:
            return bool(value() if callable(value) else value)

        while time.monotonic() < deadline:
            if buffer.buffered_seconds() >= LIVE_FACE_HEAD_START or flag(buffer.done) or flag(buffer.failed):
                break
            time.sleep(0.02)

    def _live_reset(self) -> None:
        """Cancel any in-flight live utterance and clear the renderer's buffer."""
        if self._live_client is not None:
            self._live_client.cancel()
        renderer = getattr(self.shell, "assistant_view", None)
        if renderer is not None:
            photo = getattr(renderer, "_photo_renderer", None)
            if photo is not None:
                photo.set_live_frames(None)

    def show_transcript_window(self) -> None:
        """The original dictation and transcript view, in a secondary window."""
        if self._legacy_window is None:
            window = Adw.Window()
            window.set_title("Transcript and dictation")
            window.set_default_size(1000, 640)
            window.set_transient_for(self)
            window.set_hide_on_close(True)
            window.set_content(self._legacy_view)
            self._legacy_window = window
        self._legacy_window.present()

    def _set_status(self, text: str, busy: bool = False) -> None:
        if self.status_label.get_text() != text:
            self.status_label.set_text(text)
        if self.status_spinner.get_visible() != busy:
            self.status_spinner.set_visible(busy)
        self.status_box.queue_draw()

    def _toast(self, text: str) -> None:
        self.shell.show_notice(text)

    def _reminders_tick(self) -> bool:
        """Deliver reminders whose time has come, then reschedule. Returns True to keep the timer."""
        for item in self.reminder_store.due(datetime.now()):
            phrase = timer_phrase(item.text) if item.kind == "timer" else reminder_phrase(item.text)
            self._toast(phrase)
            if self.assistant.is_active:
                self._conversation_speak(phrase)
        return True

    def _start_progress(self) -> None:
        self.progress.set_visible(True)
        if self._progress_source:
            GLib.source_remove(self._progress_source)
        self._progress_source = GLib.timeout_add(100, self._pulse_progress)

    def _pulse_progress(self) -> bool:
        self.progress.pulse()
        return True

    def _stop_progress(self) -> None:
        if self._progress_source:
            GLib.source_remove(self._progress_source)
            self._progress_source = 0
        self.progress.set_fraction(0)
        self.progress.set_visible(False)

    def _refresh_devices(self) -> None:
        try:
            self.devices = self.audio.list_devices()
            if self.devices:
                selected = next(
                    (device for device in self.devices if device.identifier == self.settings.microphone_id),
                    None,
                )
                if selected is None and self.settings.microphone_name:
                    saved_name = self.settings.microphone_name.casefold()
                    selected = next(
                        (device for device in self.devices if device.name.casefold() == saved_name),
                        None,
                    )
                if selected is None:
                    selected = self.devices[0]
                if (
                    self.settings.microphone_id != selected.identifier
                    or self.settings.microphone_name != selected.name
                ):
                    self.settings.microphone_id = selected.identifier
                    self.settings.microphone_name = selected.name
                    self.config_store.save(self.settings)
        except Exception as exc:  # noqa: BLE001 - platform boundary
            self._toast(f"Microphone scan failed: {exc}")

    def _ai_client(self) -> OllamaClient | LlamaCppClient | StrataClient:
        """The Ask AI client for whichever backend the user picked."""
        if self.settings.ai_backend == "ollama":
            return OllamaClient(self.settings.ollama_url)
        if self.settings.ai_backend == "strata":
            return StrataClient(self.settings.strata_url)
        return LlamaCppClient(self.settings.llamacpp_url)

    def _backend_label(self) -> str:
        if self.settings.ai_backend == "ollama":
            return "Ollama"
        if self.settings.ai_backend == "strata":
            return "Strata"
        return "llama.cpp"

    def _refresh_ollama_models(self) -> None:
        """Fetch the model list, waiting for a server that Voxa has only just started."""
        self._models_generation = getattr(self, "_models_generation", 0) + 1
        generation = self._models_generation
        label = self._backend_label()

        if self.settings.ai_backend == "strata":
            # Voxa does not manage Strata: one probe, no waiting, and say so when it is down.
            def strata_worker() -> None:
                client = self._ai_client()
                info = client.server_info()
                if generation != self._models_generation:
                    return
                if info is None:
                    idle(self._show_backend_offline, label)
                    return
                model = info.get("model")
                if model:
                    idle(self._apply_ollama_models, [model])
                else:
                    idle(self._show_no_models)

            threading.Thread(target=strata_worker, name="strata-models", daemon=True).start()
            return

        def list_models() -> list[str]:
            client = self._ai_client()
            try:
                if self.settings.ai_backend == "llamacpp":
                    models_dir = str(Path(self.settings.llamacpp_model).parent) if self.settings.llamacpp_model else None
                    try:
                        return client.list_models(models_dir)
                    except TypeError:
                        return client.list_models()
                return client.list_models()
            except OllamaError:
                raise
            except Exception as exc:  # connection refused and friends are not wrapped by every client
                raise OllamaError(str(exc)) from exc

        def worker() -> None:
            models = wait_for_models(
                list_models,
                attempts=MODEL_WAIT_ATTEMPTS,
                delay=MODEL_WAIT_DELAY,
                should_stop=lambda: generation != self._models_generation,
                on_waiting=lambda attempt: idle(self._show_models_starting, label) if attempt == 1 else None,
            )
            if generation != self._models_generation:
                return
            if models is None:
                # Say so in the model picker too ("No models available - Ollama") instead of
                # leaving it blank, and keep the saved model selection untouched.
                idle(self._show_no_models)
                idle(
                    self._set_status,
                    f"The AI server ({label}) is offline. Dictation is still available.",
                )
                return
            idle(self._apply_ollama_models, models)

        threading.Thread(target=worker, name="ollama-models", daemon=True).start()

    def _show_models_starting(self, label: str) -> bool:
        """The server is still coming up: say so instead of "No models available"."""
        self._set_status(f"Starting {label}…")
        return False

    def _show_no_models(self) -> bool:
        self.ollama_models = []
        self._apply_model_combo([])
        return False

    def _show_backend_offline(self, label: str) -> bool:
        """A backend Voxa does not manage (Strata) is down: name it in the picker."""
        message = f"{label} is not running"
        self.ollama_models = [message]
        self._apply_model_combo([message])
        return False

    def _detect_hardware_async(self) -> None:
        def gpu_fit_warning(available_gb: float, reserved_gb: float) -> bool:
            """True when the current model plus the reservation exceeds the VRAM."""
            if self.settings.face_mode != "live" or not self.settings.ollama_model:
                return False
            client = OllamaClient(self.settings.ollama_url)
            try:
                infos = client.list_models_detailed()
            except OllamaError:
                return False
            for info in infos:
                if info.name == self.settings.ollama_model and info.size_bytes > 0:
                    return info.size_bytes / (1024.0**3) + reserved_gb > available_gb
            return False  # Unknown size: skip silently.

        def worker() -> None:
            available_gb, source = detect_available_model_memory_gb()
            reserved = reserved_gpu_gb(self.settings.face_mode, True) if source == "GPU VRAM" else 0.0
            # Show something immediately from the built-in or cached catalog,
            # so the suggestion never waits on the network.
            catalog = load_catalog()
            idle(
                self._apply_hardware_summary,
                available_gb,
                source,
                suggest_models(available_gb, catalog=catalog, reserved_gb=reserved),
                reserved,
                gpu_fit_warning(available_gb, reserved) if reserved else False,
            )
            if not refresh_due():
                return
            try:
                refreshed = refresh_and_cache()
            except CatalogUnavailable:
                return  # Offline, or the registry is down; the cache still stands.
            if refreshed != catalog:
                idle(
                    self._apply_hardware_summary,
                    available_gb,
                    source,
                    suggest_models(available_gb, catalog=refreshed, reserved_gb=reserved),
                    reserved,
                    False,  # One toast per detection run.
                )

        threading.Thread(target=worker, name="hardware-detect", daemon=True).start()

    def _apply_hardware_summary(self, available_gb: float, source: str, suggestions: list, reserved_gb: float = 0.0, gpu_fit_warning: bool = False) -> bool:
        self._suggested_models = [model.name for model in suggestions]
        names = ", ".join(self._suggested_models)
        self._hardware_summary = f"Suggested for this machine (~{available_gb:.0f} GB {source}): {names}"
        if reserved_gb > 0:
            self._hardware_summary += f" ({reserved_gb:g} GB kept free for the High face and speech recognition)"
        if gpu_fit_warning:
            self._toast(
                "This model and the High face may not fit on the graphics card together. "
                "Try a smaller model or Medium."
            )
        self._has_gpu = source == "GPU VRAM"
        # Keep the gauge live for the app's lifetime once a GPU is detected,
        # not only while an Ollama query is in flight.
        if not self._closing:
            self._start_gpu_monitor()
        return False

    def _start_gpu_monitor(self) -> None:
        # Runs for the window's lifetime once hardware detection finds a
        # GPU; the guard below keeps repeated calls (from both detection
        # and ask_ai) from spawning a second poller.
        if not self._has_gpu or self._gpu_poll_stop is not None:
            return
        stop_event = threading.Event()
        self._gpu_poll_stop = stop_event

        def worker() -> None:
            # Sample immediately (off the UI thread) so the gauge doesn't sit
            # at 0% for a full interval before its first real reading.
            while True:
                usage = sample_gpu_usage()
                if usage is not None and not stop_event.is_set():
                    idle(self._apply_gpu_usage, usage)
                if stop_event.wait(GPU_POLL_INTERVAL_SECONDS):
                    break

        threading.Thread(target=worker, name="gpu-monitor", daemon=True).start()
        self.gpu_box.set_visible(True)

    def _stop_gpu_monitor(self) -> None:
        if self._gpu_poll_stop is not None:
            self._gpu_poll_stop.set()
            self._gpu_poll_stop = None
        self.gpu_box.set_visible(False)
        self.gpu_level.set_value(0)

    def _apply_gpu_usage(self, usage: GpuUsage) -> bool:
        self.gpu_level.set_value(usage.utilization_percent)
        self.gpu_label.set_text(f"{usage.utilization_percent:.0f}%")
        self.gpu_box.set_tooltip_text(
            f"{usage.utilization_percent:.0f}% utilization — "
            f"{usage.memory_used_gb:.1f} / {usage.memory_total_gb:.1f} GB VRAM"
        )
        return False

    def _apply_ollama_models(self, models: list[str]) -> bool:
        if self._welcome_followup and welcome.real_models(models):
            self._welcome_followup = False
            self._say_welcome_how_to()
        self.ollama_models = models
        if models and self.settings.ollama_model not in models:
            self.settings.ollama_model = models[0]
            self.config_store.save(self.settings)
        self._apply_model_combo(models)
        return False

    def _apply_model_combo(self, models: list[str]) -> None:
        self.shell.set_models(models, self.settings.ollama_model, self.settings.ai_backend)
        if not models:
            self.model_combo.set_visible(False)
            return
        selected = (
            models.index(self.settings.ollama_model)
            if self.settings.ollama_model in models
            else 0
        )
        self._model_combo_updating = True
        self.model_combo.set_model(Gtk.StringList.new(models))
        self.model_combo.set_selected(selected)
        self.model_combo.set_visible(True)
        self._model_combo_updating = False

    def _on_model_selected(self, _combo: Gtk.DropDown, _property: str) -> None:
        # Programmatic set_model/set_selected also emit this signal; the flag
        # above skips that pass so only real user picks are persisted.
        if self._model_combo_updating or not self.ollama_models:
            return
        selected = min(self.model_combo.get_selected(), len(self.ollama_models) - 1)
        model = self.ollama_models[selected]
        if model != self.settings.ollama_model:
            self.settings.ollama_model = model
            self.config_store.save(self.settings)
            self._set_status(f"Asking with {model} from now on.")
            self._warm_up_model(model)

    @staticmethod
    def _scroll_to_end(view: Gtk.TextView) -> None:
        buffer = view.get_buffer()
        end_iter = buffer.get_end_iter()
        mark = buffer.create_mark(None, end_iter, False)
        view.scroll_to_mark(mark, 0.0, False, 0.0, 0.0)
        buffer.delete_mark(mark)

    def _start_ollama_install(self) -> None:
        if self._installing:
            self._toast("Ollama installation is already running.")
            return
        self._installing = True
        cancel_event = threading.Event()
        self._install_cancel = cancel_event

        dialog = Adw.Dialog(title="Installing Ollama", content_width=560, content_height=420)
        toolbar = Adw.ToolbarView()
        dialog.set_child(toolbar)
        toolbar.add_top_bar(Adw.HeaderBar(show_end_title_buttons=False))

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        toolbar.set_content(box)

        info = Gtk.Label(
            label="This downloads the official installer from ollama.com and runs it as "
            "root. You'll be asked for your password.",
            wrap=True,
            xalign=0,
        )
        box.append(info)

        log_view = Gtk.TextView(editable=False, cursor_visible=False, monospace=True)
        log_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_vexpand(True)
        scroller.set_child(log_view)
        box.append(scroller)

        cancel_button = Gtk.Button(label="Cancel", halign=Gtk.Align.END)
        box.append(cancel_button)

        def append_log(line: str) -> bool:
            buffer = log_view.get_buffer()
            buffer.insert(buffer.get_end_iter(), line + "\n")
            self._scroll_to_end(log_view)
            return False

        def on_cancel(*_args) -> None:
            cancel_event.set()
            cancel_button.set_sensitive(False)
            cancel_button.set_label("Stopping…")

        cancel_button.connect("clicked", on_cancel)
        dialog.connect("closed", lambda *_: cancel_event.set())
        dialog.present(self)

        def worker() -> None:
            try:
                exit_code = install_ollama(
                    on_output=lambda line: idle(append_log, line),
                    cancel_event=cancel_event,
                )
            except InstallerError as exc:
                idle(self._on_install_finished, dialog, False, str(exc))
                return
            if cancel_event.is_set():
                idle(self._on_install_finished, dialog, False, "Installation cancelled.")
            elif exit_code == 0:
                idle(self._on_install_finished, dialog, True, "Ollama installed successfully.")
            else:
                idle(self._on_install_finished, dialog, False, f"Installer exited with status {exit_code}.")

        threading.Thread(target=worker, name="ollama-install", daemon=True).start()

    def _on_install_finished(self, dialog: Adw.Dialog, success: bool, message: str) -> bool:
        self._installing = False
        dialog.close()
        self._toast(message)
        if success:
            self._refresh_ollama_models()
            if self._welcome_followup:
                self._download_welcome_model()
        return False

    def _show_model_manager(self) -> None:
        dialog = Adw.Dialog(title="Manage Ollama Models", content_width=560, content_height=520)
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        dialog.set_child(root)
        root.append(Adw.HeaderBar())

        page = Adw.PreferencesPage()
        scroller = Gtk.ScrolledWindow(vexpand=True)
        scroller.set_child(page)
        root.append(scroller)

        installed_group = Adw.PreferencesGroup(title="Installed")
        page.add(installed_group)
        installed_rows: dict[str, Adw.ActionRow] = {}
        placeholder_rows: list[Adw.ActionRow] = []

        def add_installed_row(name: str) -> None:
            row = Adw.ActionRow(title=name, subtitle="Calculating size…")
            delete_button = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER)
            delete_button.add_css_class("flat")
            delete_button.set_tooltip_text(f"Remove {name}")
            delete_button.connect(
                "clicked", lambda _btn, model=name: self._confirm_delete_model(dialog, model)
            )
            row.add_suffix(delete_button)
            installed_rows[name] = row
            installed_group.add(row)

        def show_placeholder(title: str) -> None:
            row = Adw.ActionRow(title=title)
            placeholder_rows.append(row)
            installed_group.add(row)

        def build_installed(models: list[str]) -> None:
            for row in list(installed_rows.values()) + placeholder_rows:
                installed_group.remove(row)
            installed_rows.clear()
            placeholder_rows.clear()
            if not models:
                show_placeholder("No models installed yet")
                return
            for name in models:
                add_installed_row(name)

        # The dialog fetches its own list so it can say "Starting Ollama…" while the
        # server Voxa has only just started is still coming up, then rebuild in place.
        dialog_alive = [True]
        dialog.connect("closed", lambda *_: dialog_alive.__setitem__(0, False))

        def list_installed() -> list[str]:
            client = OllamaClient(self.settings.ollama_url)
            try:
                return client.list_models()
            except OllamaError:
                raise
            except Exception as exc:  # connection refused and friends are not wrapped by every client
                raise OllamaError(str(exc)) from exc

        def load_installed() -> None:
            models = wait_for_models(
                list_installed,
                attempts=MODEL_WAIT_ATTEMPTS,
                delay=MODEL_WAIT_DELAY,
                should_stop=lambda: not dialog_alive[0],
                on_waiting=lambda attempt: idle(show_placeholder, "Starting Ollama…") if attempt == 1 else None,
            )
            if not dialog_alive[0]:
                return
            if models is not None:
                idle(build_installed, models)

        threading.Thread(target=load_installed, name="ollama-installed", daemon=True).start()

        pull_group = Adw.PreferencesGroup(
            title="Pull a model",
            description="Enter any Ollama model tag, or pick a suggestion for this machine.",
        )
        page.add(pull_group)

        pull_row = Adw.EntryRow(title="Model name")
        pull_button = Gtk.Button(label="Pull", valign=Gtk.Align.CENTER)
        pull_button.add_css_class("suggested-action")
        pull_row.add_suffix(pull_button)
        pull_group.add(pull_row)

        progress_bar = Gtk.ProgressBar(visible=False, show_text=True)
        pull_group.add(progress_bar)

        if self._suggested_models:
            suggestion_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            suggestion_box.set_margin_top(4)
            suggestion_box.set_margin_bottom(8)
            suggestion_box.set_margin_start(12)
            suggestion_box.set_margin_end(12)
            for name in self._suggested_models:
                chip = Gtk.Button(label=name)
                chip.connect("clicked", lambda _btn, model=name: pull_row.set_text(model))
                suggestion_box.append(chip)
            pull_group.add(suggestion_box)

        smallest = MODEL_CATALOG[0].name
        if smallest not in self._suggested_models and smallest not in self.ollama_models:
            smallest_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            smallest_box.set_margin_top(4)
            smallest_box.set_margin_bottom(8)
            smallest_box.set_margin_start(12)
            smallest_box.set_margin_end(12)
            smallest_box.append(Gtk.Label(label="Smallest — runs anywhere"))
            chip = Gtk.Button(label=smallest)
            chip.connect("clicked", lambda _btn, model=smallest: pull_row.set_text(model))
            smallest_box.append(chip)
            pull_group.add(smallest_box)

        def set_pulling(active: bool) -> None:
            pull_button.set_sensitive(not active)
            pull_row.set_sensitive(not active)
            progress_bar.set_visible(active)
            if not active:
                progress_bar.set_fraction(0)
                progress_bar.set_text("")

        pulling = [False]
        got_total = [False]
        selected_before = [False]

        def on_progress(status: str, completed: int, total: int) -> bool:
            if total > 0:
                got_total[0] = True
                progress_bar.set_fraction(min(1.0, completed / total))
                progress_bar.set_text(f"{status} — {completed / (1024**2):.0f} / {total / (1024**2):.0f} MB")
            else:
                progress_bar.set_fraction(0.0)
                progress_bar.set_text(status)
            return False

        def pulse_tick() -> bool:
            if not pulling[0] or got_total[0]:
                return False
            progress_bar.pulse()
            return True

        def on_pull_finished(success: bool, message: str, name: str) -> bool:
            set_pulling(False)
            self._toast(message)
            if success:
                try:
                    models = OllamaClient(self.settings.ollama_url).list_models()
                except OllamaError:
                    models = list(self.ollama_models)
                    if name not in models:
                        models.append(name)
                build_installed(models)
                if not selected_before[0]:
                    self.settings.ollama_model = name
                    self.config_store.save(self.settings)
                self._apply_ollama_models(models)
            pull_row.set_text("")
            return False

        def start_pull(*_args) -> None:
            name = pull_row.get_text().strip()
            if not name:
                self._toast("Enter a model name first.")
                return
            selected_before[0] = bool(self.settings.ollama_model)
            set_pulling(True)
            pulling[0] = True
            got_total[0] = False
            GLib.timeout_add(150, pulse_tick)
            cancel_event = threading.Event()
            client = OllamaClient(self.settings.ollama_url)

            def worker() -> None:
                try:
                    models = wait_for_models(
                        client.list_models,
                        attempts=MODEL_WAIT_ATTEMPTS,
                        delay=MODEL_WAIT_DELAY,
                        should_stop=cancel_event.is_set,
                        on_waiting=lambda attempt: idle(
                            on_progress, "Starting Ollama…", 0, 0
                        ) if attempt == 1 else None,
                    )
                    if models is None:
                        idle(on_pull_finished, False, "The AI server never came up.", name)
                        return
                    client.pull_model(
                        name,
                        cancel_event=cancel_event,
                        on_progress=lambda status, completed, total: idle(
                            on_progress, status, completed, total
                        ),
                    )
                    try:
                        models = client.list_models()
                    except OllamaError:
                        models = self.ollama_models
                    idle(on_pull_finished, True, f"Pulled {name}.", name)
                except OllamaError as exc:
                    idle(on_pull_finished, False, str(exc), name)

            threading.Thread(target=worker, name="ollama-pull", daemon=True).start()

        pull_button.connect("clicked", start_pull)
        dialog.present(self)

    def _confirm_delete_model(self, parent_dialog: Adw.Dialog, model: str) -> None:
        confirm = Adw.AlertDialog(
            heading=f"Remove {model}?",
            body="This deletes the downloaded model from disk. You can pull it again later.",
        )
        confirm.add_response("cancel", "Cancel")
        confirm.add_response("delete", "Remove")
        confirm.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        confirm.set_default_response("cancel")
        confirm.set_close_response("cancel")

        def on_response(_dialog, response: str) -> None:
            if response != "delete":
                return
            endpoint = self.settings.ollama_url

            def worker() -> None:
                try:
                    OllamaClient(endpoint).delete_model(model)
                    try:
                        models = OllamaClient(endpoint).list_models()
                    except OllamaError:
                        models = [name for name in self.ollama_models if name != model]
                    idle(self._apply_ollama_models, models)
                    idle(self._on_model_deleted, parent_dialog, True, f"Removed {model}.")
                except OllamaError as exc:
                    idle(self._on_model_deleted, parent_dialog, False, str(exc))

            threading.Thread(target=worker, name="ollama-delete", daemon=True).start()

        confirm.connect("response", on_response)
        confirm.present(parent_dialog)

    def _on_model_deleted(self, parent_dialog: Adw.Dialog, success: bool, message: str) -> bool:
        self._toast(message)
        if success:
            parent_dialog.close()
            self._show_model_manager()
        return False

    def _load_whisper(self, model_name: str) -> None:
        self._set_status(f"Loading Whisper {model_name}…", busy=True)
        self._start_progress()
        self.whisper.load_async(
            model_name,
            lambda name, backend: idle(self._on_whisper_ready, name, backend),
            lambda error: idle(self._on_whisper_error, error),
        )

    def _on_whisper_ready(self, name: str, backend: str) -> bool:
        self.settings.whisper_model = name
        self.config_store.save(self.settings)
        self._stop_progress()
        self._set_status(f"Ready — Whisper {name} on {backend}")
        if self._activate_when_ready:
            # ACTIVE was pressed while the model loaded: start listening by itself now.
            self._activate_when_ready = False
            self._on_shell_active()
        elif not self._announced_ready and not self.assistant.is_active:
            # ACTIVE cannot start listening until the speech model has loaded; say when it can.
            self._announced_ready = True
            self._toast("Voxa is ready. Press ACTIVE to start listening.")
        GLib.timeout_add(1500, lambda: (self._maybe_welcome(), False)[1])
        return False

    def _maybe_welcome(self) -> None:
        """First start only: greet the user and offer the one missing piece (Ollama, or a model)."""
        if os.environ.get("VOXA_NO_WELCOME") or self.settings.welcomed or getattr(self, "_closing", False):
            return
        self.settings.welcomed = True
        self.config_store.save(self.settings)

        def worker() -> None:
            step = welcome.probe(
                self.settings.ai_backend,
                ollama_installed,
                OllamaClient(self.settings.ollama_url).list_models,
            )
            idle(self._show_welcome, step)

        threading.Thread(target=worker, name="welcome-probe", daemon=True).start()

    def _show_welcome(self, step: str) -> bool:
        if self._closing:
            return False
        self._welcome_followup = step != "ready"
        name = self._character_name()
        WelcomeDialog(name, step, self._on_welcome_primary, self._on_welcome_tutorial).present(self)
        self.speech.speak(
            welcome.spoken(step, name),
            self.settings.tts_rate,
            self._reply_voice(),
            on_started=lambda: None,
            on_done=lambda: None,
            on_error=lambda error: None,
        )
        return False

    def _say_welcome_how_to(self) -> None:
        self._toast("You're all set. " + welcome.HOW_TO)
        self.speech.speak(
            "You're all set. " + welcome.HOW_TO,
            self.settings.tts_rate,
            self._reply_voice(),
            on_started=lambda: None,
            on_done=lambda: None,
            on_error=lambda error: None,
        )

    def _say_notice(self, text: str) -> bool:
        """Speak a short notice in the middle of a running command (for example: please enter your password)."""
        self._toast(text)
        self.speech.speak(
            text,
            self.settings.tts_rate,
            self._reply_voice(),
            on_started=lambda: None,
            on_done=lambda: None,
            on_error=lambda error: None,
        )
        return False

    def _download_welcome_model(self) -> None:
        """Fetch the smallest model with no further click; the model list refresh then finishes the welcome."""
        if getattr(self, "_welcome_downloading", False):
            return
        self._welcome_downloading = True
        name = welcome.SMALLEST_MODEL
        cancel_event = threading.Event()
        self._welcome_cancel = cancel_event
        self._set_status(welcome.download_status("", 0, 0), busy=True)

        def worker() -> None:
            client = OllamaClient(self.settings.ollama_url)
            try:
                models = wait_for_models(
                    client.list_models,
                    attempts=MODEL_WAIT_ATTEMPTS,
                    delay=MODEL_WAIT_DELAY,
                    should_stop=cancel_event.is_set,
                )
                if models is None:
                    idle(self._on_welcome_download_done, False, "The AI server never came up.")
                    return
                client.pull_model(
                    name,
                    cancel_event=cancel_event,
                    on_progress=lambda status, completed, total: idle(
                        self._set_status, welcome.download_status(status, completed, total), True
                    ),
                )
                idle(self._on_welcome_download_done, True, f"{name} is ready.")
            except OllamaError as exc:
                idle(self._on_welcome_download_done, False, str(exc))

        threading.Thread(target=worker, name="welcome-pull", daemon=True).start()

    def _on_welcome_download_done(self, ok: bool, message: str) -> bool:
        self._welcome_downloading = False
        self._toast(message)
        if ok:
            self._refresh_ollama_models()   # _apply_ollama_models then says "You're all set"
        else:
            self._set_status(message)
            self._show_model_manager()      # let the user retry or pick another model by hand
        return False

    def _on_welcome_primary(self, step: str) -> None:
        self.speech.stop()
        if step == "install":
            self._start_ollama_install()
        elif step == "model":
            self._download_welcome_model()

    def _on_welcome_tutorial(self) -> None:
        host.spawn(["xdg-open", welcome.TUTORIAL_URL])

    def _character_name(self) -> str:
        avatar = get_avatar(self.settings.character_id) if self.settings.character_id else None
        return getattr(avatar, "name", "") or "Voxa"

    def _on_whisper_error(self, error: str) -> bool:
        self._stop_progress()
        self._set_status("Whisper could not be loaded.")
        self._activate_when_ready = False
        self.assistant_model.set_state(AssistantState.OFFLINE, f"Speech model failed — {error}")
        self._toast(error)
        return False

    def _on_record_toggled(self, button: Gtk.ToggleButton) -> None:
        if button.get_active() and not self.listening:
            self.start_recording()
        elif not button.get_active() and self.listening:
            self.stop_recording()

    def toggle_recording(self) -> None:
        self.record_button.set_active(not self.record_button.get_active())

    def start_recording(self) -> None:
        if not self.whisper.ready:
            self.record_button.set_active(False)
            self._toast("Whisper is still loading.")
            return
        if not self.devices:
            self.record_button.set_active(False)
            self._toast("No microphone is available.")
            return
        if self.assistant.is_active:
            self.assistant.go_offline()  # dictation and hands-free mode never share the microphone

        self.dictation = DictationController(
            self.whisper,
            language=self.settings.language,
            threshold=self.settings.voice_threshold,
            silence_ms=self.settings.silence_ms,
            max_segment_seconds=self.settings.max_segment_seconds,
            on_text=lambda text: idle(self._append_transcript, text),
            on_status=lambda text: idle(self._on_dictation_status, text),
            on_auto_stop=lambda: idle(self._auto_stop_recording),
            on_error=lambda text: idle(self._toast, text),
        )
        self.dictation.start()
        try:
            self._prepare_microphone()
            self.audio.start(
                self.settings.microphone_id,
                self.dictation.feed,
                self._queue_level,
                lambda error: idle(self._capture_error, error),
            )
        except Exception as exc:  # noqa: BLE001 - platform boundary
            self.dictation.stop()
            self.dictation = None
            self.record_button.set_active(False)
            self._toast(str(exc))
            return

        self.listening = True
        self._latest_level = 0.0
        self._start_level_updates()
        self.record_button.set_label("Stop")
        self.record_button.add_css_class("destructive-action")
        self.conversation_button.set_sensitive(False)
        self._set_status("Listening…", busy=True)

    def stop_recording(self) -> None:
        self.audio.stop()
        if self.dictation is not None:
            self.dictation.stop()
            self.dictation = None
        self.listening = False
        self._stop_level_updates()
        self.record_button.set_label("Dictate")
        self.record_button.remove_css_class("destructive-action")
        if self.record_button.get_active():
            self.record_button.set_active(False)
        self.conversation_button.set_sensitive(True)
        self.level.set_value(0)
        self._set_status("Ready")

    def _auto_stop_recording(self) -> bool:
        if self.listening:
            self.stop_recording()
        return False

    def _capture_error(self, error: str) -> bool:
        self.stop_recording()
        self._toast(f"Microphone error: {error}")
        return False

    def _on_conversation_toggled(self, button: Gtk.ToggleButton) -> None:
        if button.get_active() and not self.assistant.is_active:
            self.assistant.activate()
        elif not button.get_active() and self.assistant.is_active:
            self.assistant.go_offline()

    def toggle_conversation(self) -> None:
        self.conversation_button.set_active(not self.conversation_button.get_active())

    def start_conversation_mode(self) -> bool:
        """Start the hands-free conversation pipeline; returns False (and says why) if it cannot."""
        if not self.whisper.ready:
            self.conversation_button.set_active(False)
            self._activate_when_ready = True
            self._start_failure = "Whisper is still loading."
            return False
        if not self.devices:
            self.conversation_button.set_active(False)
            self._start_failure = "No microphone is available."
            self._toast(self._start_failure)
            return False
        if self.listening:
            self.stop_recording()

        # Everything the pipeline queues back to the GTK thread is tied to this session:
        # once OFFLINE (or re-activated) a late callback is dropped, never acted on.
        token = self.assistant.token()

        def session(callback: Callable) -> Callable:
            return self._for_session(token, callback)

        self.conversation = ConversationController(
            wake_whisper=self.wake_whisper if self.wake_whisper.ready else self.whisper,
            prompt_whisper=self.whisper,
            language=self.settings.language,
            wake_word=self.settings.wake_word,
            threshold=self.settings.voice_threshold,
            silence_ms=self.settings.silence_ms,
            max_segment_seconds=self.settings.max_segment_seconds,
            on_woken=lambda: idle(session(self._on_conversation_woken)),
            on_prompt=lambda text: idle(session(self._on_conversation_prompt), text),
            on_status=lambda text: idle(session(self._set_status), text, True),
            on_error=lambda text: idle(session(self._toast), text),
            on_exit=lambda kind: idle(session(self._on_conversation_exit), kind),
        )
        self.conversation.start()
        try:
            self._prepare_microphone()
            self.audio.start(
                self.settings.microphone_id,
                self.conversation.feed,
                self._queue_level,
                lambda error: idle(session(self._conversation_capture_error), error),
            )
        except Exception as exc:  # noqa: BLE001 - platform boundary
            self.conversation.stop()
            self.conversation = None
            self.conversation_button.set_active(False)
            self._start_failure = str(exc)
            self._toast(self._start_failure)
            return False

        self.conversation_active = True
        self._latest_level = 0.0
        self._start_level_updates()
        self.conversation_button.set_label("Stop listening")
        self.conversation_button.add_css_class("destructive-action")
        self.record_button.set_sensitive(False)
        self._set_status(f"Conversation mode — say “{self.settings.wake_word}” to begin", busy=True)
        return True

    def stop_conversation_mode(self) -> None:
        self.audio.stop()
        if self.conversation is not None:
            self.conversation.stop()
            self.conversation.unmute()
            self.conversation = None
        self.conversation_active = False
        # A reply may still be playing: speech.stop() fires no callbacks, so
        # the mute/barge-in state is reset explicitly here.
        self.speech.stop()
        self._live_reset()
        self._speaking_since = 0.0
        self._barge_gate.reset()
        if self._pending_user_generation is not None:
            self._conversation_history.drop_last()
        self._pending_user_generation = None
        self.query_cancel.set()
        self._query_generation += 1
        self._stop_level_updates()
        self.conversation_button.set_label("Conversation")
        self.conversation_button.remove_css_class("destructive-action")
        if self.conversation_button.get_active():
            self.conversation_button.set_active(False)
        self.record_button.set_sensitive(True)
        self.level.set_value(0)
        self._set_status("Ready")

    def _conversation_capture_error(self, error: str) -> bool:
        self.assistant.go_offline()
        self._toast(f"Microphone error: {error}")
        return False

    def _on_conversation_woken(self) -> bool:
        if self.assistant.is_paused:
            # The wake word was heard: that is exactly what ends a pause. Resume, then carry on as for any
            # wake word, so a request spoken in the same breath is captured normally.
            self.assistant.resume()
            self._toast("I'm back.")
        self.assistant.wake(self.assistant.token())
        self._toast(f"Heard “{self.settings.wake_word}” — listening…")
        self._set_status("Listening for your request…", busy=True)
        return False

    def _on_conversation_prompt(self, text: str) -> bool:
        if not text:
            return False
        heard = self.conversation.last_heard if self.conversation is not None else text
        if self.assistant.is_paused:
            resumed, rest = resume_request(heard, self.settings.wake_word)
            if not resumed:
                # Stay paused: the sentence is neither a resume nor a request.
                return False
            self.assistant.resume()
            self._toast("I'm back.")
            if rest:
                self.ask_ai(rest)
            return False
        cmd = hearing.direct_command(heard)
        if cmd:
            if cmd == "pause":
                # "Pause" is for Voxa herself. The music pauses only when it is named ("pause the music").
                self._pause_now()
                return False
            call = intents.route(cmd)
            if call is not None:
                self._run_tool(call, cmd)
            return False
        if hearing.should_drop_noise(heard, self._follow_up_token is not None):
            self._set_status("Ready")
            return False
        if is_pause_request(heard, media_playing=active_player() is not None):
            self._pause_now()
            return False
        stripped = strip_wake_word(text, self.settings.wake_word)
        if stripped is not None and not stripped:
            return False
        self.ask_ai(stripped if stripped is not None else text)
        return False

    def _on_conversation_exit(self, kind: str) -> bool:
        # The user ended the exchange by voice: cancel any in-flight request,
        # make the mic available again, and either drop back to listening for
        # the wake word ("cancel") or leave conversation mode ("goodbye").
        self.query_cancel.set()
        self._query_generation += 1
        if self._pending_user_generation is not None:
            self._conversation_history.drop_last()
        self._pending_user_generation = None
        if self.conversation is not None:
            self.conversation.unmute()
        self.speech.stop()
        self._live_reset()
        self._speaking_since = 0.0
        self._barge_gate.reset()
        if kind == "goodbye":
            # Exactly what pressing the red OFFLINE button does.
            self.stop_current_work()
            self._set_status("Goodbye!")
        else:
            # A spoken "cancel": the request was abandoned and its late completion is
            # ignored on purpose, so the assistant must be returned to READY right here.
            self._end_query_task("cancelled")
            self.assistant.abandon(self.assistant.token())
            if kind == "stop":
                self._clear_display()
            self._set_status(self._conversation_idle_status())
        return False

    def _open_app(self, app_name: str, prompt: str) -> None:
        """"Open X": launch a menu application directly, without asking the AI."""
        self._tool_ran_for_request = True
        self.query_cancel.set()
        self._query_generation += 1
        self._end_query_task("cancelled")
        self.shell.exchange_panel.show_question(prompt)
        task = self.assistant.begin_task(f"Opening {app_name}")
        self._query_task_id = task.id if task is not None else None
        token = self.assistant.token()
        self.assistant.prompt_accepted(token)
        self._set_status(f"Opening {app_name}…", busy=True)

        # While OFFLINE there is no session to guard: the app still opens, with a toast.
        active = self.assistant.is_active

        def guard(callback):
            return self._for_session(token, callback) if active else callback

        def worker() -> None:
            try:
                if self._app_cache is None:
                    self._app_cache = apps.list_apps()
                app = apps.match_app(app_name, self._app_cache)
                if app is not None:
                    apps.launch(app)
                idle(guard(self._on_app_opened), app.name if app else None, app_name)
            except Exception as exc:  # noqa: BLE001 - host access may be missing
                idle(guard(self._on_app_open_failed), str(exc))

        threading.Thread(target=worker, name="open-app", daemon=True).start()

    def _on_app_opened(self, name: str | None, requested: str) -> bool:
        reply = f"Opening {name}." if name else f"I couldn't find an app called {requested}."
        self._end_query_task("done" if name else "cancelled")
        self.shell.exchange_panel.show_answer(reply)
        if not self.assistant.is_active:
            self._toast(reply)
        elif self.conversation_active:
            self._conversation_speak(reply)
        else:
            self.assistant.reply_finished(self.assistant.token())
            self._toast(reply)
        self._set_status(reply)
        return False

    def _on_app_open_failed(self, error: str) -> bool:
        self._end_query_task("cancelled")
        self._fail_assistant("Could not open the app")
        self._toast(f"Could not open the app: {error}")
        return False

    def _log_action(
        self,
        heard: str,
        route: str,
        tool: str = "",
        args: dict[str, str] | None = None,
        ok: bool | None = None,
        speech: str = "",
        detail: str = "",
        ms: int = 0,
    ) -> None:
        """Record one command in the action log; logging never breaks a command."""
        self.action_log.append(
            ActionRecord(
                time=datetime.now().isoformat(timespec="seconds"),
                heard=heard,
                route=route,
                tool=tool,
                args=args or {},
                ok=ok,
                speech=speech,
                detail=detail,
                ms=ms,
            )
        )

    def _run_tool(self, call: intents.ToolCall, prompt: str) -> None:
        """A recognised command goes straight to its tool, never to the model."""
        if repeat.repeatable(call.tool):
            self._last_tool_call = call
        self._tool_ran_for_request = True
        self.query_cancel.set()
        self._query_generation += 1
        self._end_query_task("cancelled")
        self.shell.exchange_panel.show_question(prompt)
        title = call.tool.replace("_", " ").capitalize()
        caption = working_caption(call.tool, call.args)
        task = self.assistant.begin_task(title)
        self._query_task_id = task.id if task is not None else None
        token = self.assistant.token()
        self.assistant.prompt_accepted(token, caption)
        self._set_status(caption, busy=True)

        # While OFFLINE there is no session to guard: the tool still runs, with a toast.
        active = self.assistant.is_active

        def guard(callback):
            return self._for_session(token, callback) if active else callback

        def worker() -> None:
            started = time.monotonic()
            try:
                result = self.tools.call(call.tool, call.args)
                self._log_action(
                    prompt,
                    "tool",
                    tool=call.tool,
                    args=dict(call.args),
                    ok=result.ok,
                    speech=result.speech,
                    detail=result.detail,
                    ms=int((time.monotonic() - started) * 1000),
                )
                idle(guard(self._on_tool_finished), result)
            except ToolError as exc:
                self._log_action(
                    prompt,
                    "tool",
                    tool=call.tool,
                    args=dict(call.args),
                    ok=False,
                    detail=str(exc),
                    ms=int((time.monotonic() - started) * 1000),
                )
                idle(guard(self._on_tool_failed), str(exc))
            except Exception as exc:  # noqa: BLE001 - tool handlers touch the host
                self._log_action(
                    prompt,
                    "tool",
                    tool=call.tool,
                    args=dict(call.args),
                    ok=False,
                    detail=str(exc),
                    ms=int((time.monotonic() - started) * 1000),
                )
                idle(guard(self._on_tool_failed), str(exc))

        threading.Thread(target=worker, name="run-tool", daemon=True).start()

    def _start_routine(self, steps: tuple[str, ...]) -> None:
        """Queue a routine's steps and run the first through the normal router."""
        self._routine_queue = list(steps)
        self._routine_active = True
        self._run_routine_step()

    def _run_routine_step(self) -> None:
        """Run the next queued step, or say Done when the queue is empty."""
        if not self._routine_queue:
            self._routine_active = False
            self._toast("Done.")
            return
        step = self._routine_queue.pop(0)
        call = intents.route(step)
        if call is not None:
            self._run_tool(call, step)
        else:
            # A step Voxa cannot route: skip it and continue with the next.
            self._run_routine_step()

    def _on_tool_finished(self, result) -> bool:
        if self._routine_active:
            # A routine step just finished: run the next one, or say Done when the
            # queue is empty. The per-step speech is suppressed; only "Done." speaks.
            self._run_routine_step()
            return False
        speech = result.speech
        self._end_query_task("done" if result.ok else "cancelled")
        if speech:
            self.shell.exchange_panel.show_answer(speech)
            if not self.assistant.is_active:
                self._toast(speech)
            elif self.conversation_active:
                self._conversation_speak(speech)
            else:
                self.assistant.reply_finished(self.assistant.token())
                self._toast(speech)
            self._set_status(speech)
        else:
            # A silent tool (typing into a window, for instance): no speech, no toast,
            # but the turn is over and the assistant is listening again.
            if self.assistant.is_active:
                self.assistant.reply_finished(self.assistant.token())
            if self.conversation is not None and not self.assistant.is_paused:
                self.conversation.open_followup(self.settings.followup_seconds)
            if self.conversation_active and self.settings.followup_seconds > 0:
                self._set_status("Listening for a follow-up…")
            else:
                self._set_status(self._conversation_idle_status())
        if result.ok:
            self._maybe_suggest()
        return False

    def _on_tool_failed(self, error: str) -> bool:
        self._end_query_task("cancelled")
        self._fail_assistant("That didn't work")
        self._toast(f"That didn't work: {error}")
        return False

    def _maybe_suggest(self) -> None:
        """Spawn a worker thread to check for a coaching suggestion."""
        def worker() -> None:
            try:
                records = self.action_log.read(limit=50)
                suggestion = suggest(records, self.suggestion_state, self.settings.suggestions_enabled)
            except Exception:  # noqa: BLE001 - coaching must never break a command
                return
            if suggestion is not None:
                heard = str(records[-1].get("heard", "")) if records else ""
                self.suggestion_state.offered(suggestion.key)
                idle(self._on_suggestion, suggestion, heard)

        threading.Thread(target=worker, name="suggest", daemon=True).start()

    def _on_suggestion(self, suggestion: Suggestion, heard: str) -> bool:
        """Show a coaching suggestion on the main thread."""
        self._toast(suggestion.text)
        if self.conversation_active:
            self._conversation_speak(suggestion.text)
        self._log_action(
            heard,
            "suggestion",
            detail=suggestion.key,
            speech=suggestion.text,
        )
        return False

    def _clear_display(self) -> None:
        """A blank interface: no transcript, answer, exchange card or remembered turns."""
        self._conversation_history.clear()
        self._pending_user_generation = None
        self._set_text(self.transcript_view, "")
        self._set_text(self.response_view, "")
        self.shell.exchange_panel.clear()

    def _own_language(self) -> str:
        """The language the current character speaks by default."""
        avatar = get_avatar(self.settings.character_id)
        locale = avatar.locale if avatar is not None else "en-US"
        for name, (tag, _, _) in LANGUAGES.items():
            if tag == locale:
                return name
        return "English"

    def _reply_voice(self) -> str:
        """The voice to speak with: the reply language's, matching the
        character's gender, while a foreign reply language is active."""
        if self._reply_language and self._reply_language != self._own_language():
            _, female, male = LANGUAGES[self._reply_language]
            avatar = get_avatar(self.settings.character_id)
            gender = avatar.gender if avatar is not None else "Other"
            return female if gender == "Female" else male
        return self.settings.tts_voice

    def _conversation_speak(self, text: str) -> None:
        typing.LAST_REPLY = text
        self._recent_spoken = self._speaking_text
        self._speaking_text = text
        if self.conversation is not None:
            self.conversation.mute()
        self._speaking_since = time.monotonic()
        self._barge_gate.reset()
        token = self.assistant.token()
        self.assistant.reply_started(token)
        # Tie the completion notification to this reply: after a barge-in the old speech's
        # "done" must not touch a newer request.
        reply_id = self.assistant.current_reply()
        self._set_status("Speaking…", busy=True)
        self.shell.set_speech_clock(self._lip_clock)
        self.speech.speak(
            text,
            self.settings.tts_rate,
            self._reply_voice(),
            on_started=lambda: idle(self._for_session(token, self._set_status), "Speaking…", True),
            on_done=lambda: idle(self._for_session(token, self._on_conversation_speech_done), reply_id),
            on_error=lambda error: idle(self._for_session(token, self._on_conversation_speech_error), error),
            on_words=lambda words: idle(self.shell.set_word_timeline, words),
            **(
                {"before_play": self._live_before_play}
                if self._live_client is not None and self.settings.face_mode == "live"
                else {}
            ),
        )

    def _conversation_idle_status(self) -> str:
        return (
            f"Listening — say “{self.settings.wake_word}” to ask something"
            if self.conversation_active
            else "Ready"
        )

    def _on_conversation_speech_done(self, reply_id: int | None = None) -> bool:
        if self.conversation is not None and self.settings.followup_seconds > 0 and not self.assistant.is_paused:
            # Listen for a follow-up straight away, without the wake word.
            self.conversation.arm_prompt()
            self.conversation.open_followup(self.settings.followup_seconds)
            self._follow_up_token = self.assistant.token()
            GLib.timeout_add_seconds(self.FOLLOW_UP_SECONDS, self._follow_up_expired, self._follow_up_token)
        waiting = self.conversation is not None and self.conversation.waiting_for_prompt
        if self.conversation is not None:
            self.conversation.unmute()
        self._speaking_since = 0.0
        self._barge_gate.reset()
        self.assistant.reply_finished(self.assistant.token(), waiting_for_prompt=waiting, reply_id=reply_id)
        self._set_status("Listening for a follow-up…" if waiting else self._conversation_idle_status())
        return False

    FOLLOW_UP_SECONDS = 10

    def _follow_up_expired(self, token: int) -> bool:
        if token != self._follow_up_token or self.conversation is None:
            return False
        self._follow_up_token = None
        if self.conversation.waiting_for_prompt and self.assistant.model.state is AssistantState.LISTENING:
            self.conversation.waiting_for_prompt = False
            self.assistant.abandon(token)
            self._set_status(self._conversation_idle_status())
        return False

    def _on_conversation_speech_error(self, error: str) -> bool:
        if self.conversation is not None:
            self.conversation.unmute()
        self._speaking_since = 0.0
        self._barge_gate.reset()
        self._fail_assistant("Speech playback failed")
        self._toast(error)
        self._set_status(self._conversation_idle_status())
        return False

    def _on_dictation_status(self, text: str) -> bool:
        # Keep the label stable while recording. Rapid label replacement caused
        # stale glyph fragments with GTK 4 on some NVIDIA/X11/Compiz desktops.
        if self.listening and text == "Transcribing…":
            return False
        self._set_status(text, busy=self.listening)
        return False

    def _queue_level(self, value: float) -> None:
        # GStreamer's callback runs outside the GTK main loop. Store only the
        # newest level instead of adding an unbounded series of idle callbacks.
        self._latest_level = max(0.0, min(4000.0, value))

    def _start_level_updates(self) -> None:
        if not self._level_source:
            self._level_source = GLib.timeout_add(50, self._flush_level)

    def _flush_level(self) -> bool:
        # Both dictation and conversation mode run the live level meter; in
        # conversation mode this also samples for barge-in.
        if not self.listening and not self.conversation_active:
            self._level_source = 0
            return False
        self.level.set_value(self._latest_level)
        self.shell.set_audio_level(self._latest_level / 4000.0)
        self.status_box.queue_draw()
        self._maybe_barge_in()
        return True

    def _maybe_barge_in(self) -> None:
        """Interrupt a spoken reply when the user talks over it.

        The mic keeps reporting levels while muted, so sustained loud speech
        here means the user interrupted: stop the reply, unmute, and let the
        very next utterance become a prompt without the wake word. A level
        alone is not enough — the captured window is transcribed first and
        the reply is stopped only when the words are not her own.
        """
        if self.assistant.is_paused:
            self._barge_gate.reset()
            return
        if not self.conversation_active or self.conversation is None or not self.conversation.muted:
            self._barge_gate.reset()
            return
        if self._speaking_since <= 0.0:
            return
        if time.monotonic() - self._speaking_since < BARGE_IN_GRACE_SECONDS:
            self._barge_gate.reset()
            return
        level01 = self._latest_level / 4000.0
        if not self._barge_gate.update(level01, self._barge_gate.baseline):
            # Still her voice (or noise): teach the gate what "normal" is.
            self._barge_gate.note_speaking_level(level01)
            return
        heard = self.conversation.candidate_text()
        if is_own_voice(heard, self._speaking_text, self._recent_spoken):
            # The loud sound was Voxa's own voice: carry on speaking.
            self._barge_gate.reset()
            return
        self._barge_gate.reset()
        self._speaking_since = 0.0
        self.speech.stop()
        self._live_reset()
        self.conversation.unmute()
        self.conversation.arm_prompt()
        self.assistant.barge_in(self.assistant.token())
        self._toast("Interrupted — go ahead.")
        self._set_status("Listening for your request…", busy=True)

    def _stop_level_updates(self) -> None:
        if self._level_source:
            GLib.source_remove(self._level_source)
            self._level_source = 0
        self._latest_level = 0.0
        self.level.set_value(0)
        self.status_box.queue_draw()

    def _append_transcript(self, text: str) -> bool:
        buffer = self.transcript_view.get_buffer()
        end = buffer.get_end_iter()
        prefix = "" if buffer.get_char_count() == 0 else " "
        buffer.insert(end, prefix + text.strip())
        self._scroll_to_end(self.transcript_view)
        self._set_status("Listening…" if self.listening else "Ready", busy=self.listening)
        return False

    def _get_text(self, view: Gtk.TextView) -> str:
        buffer = view.get_buffer()
        return buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True).strip()

    def _set_text(self, view: Gtk.TextView, text: str) -> None:
        view.get_buffer().set_text(text)

    def _stop_dictation_for_action(self) -> None:
        """Stop microphone capture before actions that consume or replace text.

        Skipped while conversation mode owns the shared ``self.audio``
        pipeline (dictation and conversation mode are mutually exclusive, so
        there is never a dictation session to stop in that case) — otherwise
        this would tear down the mic capture conversation mode itself needs,
        since ``ask_ai`` runs this on every automatic conversation turn.
        Otherwise it must run unconditionally: a dictation session can
        already be producing results while ``self.listening`` is not yet
        set. ``stop_recording`` is idempotent, so an idle call is harmless.
        """
        if not self.conversation_active:
            self.stop_recording()

    def _copy_view_text(self, view: Gtk.TextView, *, empty_message: str, done_message: str) -> None:
        text = self._get_text(view)
        if not text:
            self._toast(empty_message)
            return
        display = Gdk.Display.get_default()
        if display is None:
            self._toast("The clipboard is unavailable.")
            return
        display.get_clipboard().set(text)
        self._toast(done_message)

    def copy_transcript(self) -> None:
        self._stop_dictation_for_action()
        self._copy_view_text(
            self.transcript_view,
            empty_message="There is no transcript to copy.",
            done_message="Transcript copied.",
        )

    def copy_response(self) -> None:
        self._copy_view_text(
            self.response_view,
            empty_message="There is no AI response to copy.",
            done_message="Response copied.",
        )

    def clear_all(self) -> None:
        self._stop_dictation_for_action()
        self.stop_current_work()
        # A fresh start includes forgetting the conversation thread: the model
        # shouldn't carry context from a cleared transcript into the next turn.
        self._conversation_history.clear()
        self._pending_user_generation = None
        self._set_text(self.transcript_view, "")
        self._set_text(self.response_view, "")
        self.shell.exchange_panel.clear()
        self._set_status("Ready")

    def _start_external_dictation(self, prompt: str) -> None:
        self._external_dictation = True
        self._last_dictated = ""
        self.assistant_model.set_state(AssistantState.DICTATING, "")
        if self.conversation is not None:
            self.conversation.hold_prompt()
        self._on_tool_finished(ToolResult.success("Dictating. Say stop dictating when you're done."))

    def _stop_external_dictation(self) -> None:
        self._external_dictation = False
        if self.assistant_model.state is AssistantState.DICTATING:
            self.assistant_model.set_state(AssistantState.READY, "")
        if self.conversation is not None:
            self.conversation.release_prompt()

    def _start_issue_flow(self, project: str, heard: str) -> None:
        """Find the project off the GTK thread, then ask for the title."""
        self._log_action(heard, "issue", detail=f"start {project}")

        def worker() -> None:
            owner = self.settings.github_owner or issueflow.detect_owner()
            repo = issueflow.find_repo(project, owner=owner)
            idle(self._begin_issue_flow, project, repo)

        threading.Thread(target=worker, name="issue-repo", daemon=True).start()

    def _begin_issue_flow(self, project: str, repo: str | None) -> bool:
        if repo is None:
            self._on_tool_finished(ToolResult.failure(f"I could not find a GitHub project called {project}. If I do not know your GitHub name yet, say: my GitHub name is, and then the name."))
            return False
        self._issue_flow = issueflow.IssueFlow(repo)
        self.assistant_model.set_state(AssistantState.DICTATING, self._issue_flow.caption)
        if self.conversation is not None:
            self.conversation.hold_prompt()
        self._on_tool_finished(ToolResult.success(f"This will go to {repo}. {issueflow.ASK_TITLE}"))
        return False

    def _feed_issue_flow(self, text: str) -> None:
        flow = self._issue_flow
        speech, url = flow.feed(text, parse_dictation_control(text))
        if url is not None:
            host.spawn(["xdg-open", url])
        if not flow.active:
            self._issue_flow = None
            if self.assistant_model.state is AssistantState.DICTATING:
                self.assistant_model.set_state(AssistantState.READY, "")
            if self.conversation is not None:
                self.conversation.release_prompt()
        else:
            self.assistant_model.set_state(AssistantState.DICTATING, flow.caption)
        if speech:
            self._on_tool_finished(ToolResult.success(speech))

    def _start_mail_flow(self, to_spoken: str, heard: str) -> None:
        self._log_action(heard, "email", detail=f"dictated to {to_spoken}")
        self._mail_flow = mailflow.MailFlow(to_spoken, dict(self.settings.contacts))
        self.assistant_model.set_state(AssistantState.DICTATING, self._mail_flow.caption)
        if self.conversation is not None:
            self.conversation.hold_prompt()
        self._on_tool_finished(ToolResult.success(self._mail_flow.first_question()))

    def _feed_mail_flow(self, text: str) -> None:
        flow = self._mail_flow
        speech, ready = flow.feed(text, parse_dictation_control(text))
        if flow.learned is not None:
            name, address = flow.learned
            flow.learned = None
            self.settings.contacts[name] = address
            self.config_store.save(self.settings)
        if ready:
            to, subject, body = flow.to, flow.subject, flow.body
            threading.Thread(target=mail.compose, args=(to, subject, body), name="mail-compose", daemon=True).start()
        if not flow.active:
            self._mail_flow = None
            if self.assistant_model.state is AssistantState.DICTATING:
                self.assistant_model.set_state(AssistantState.READY, "")
            if self.conversation is not None:
                self.conversation.release_prompt()
        else:
            self.assistant_model.set_state(AssistantState.DICTATING, flow.caption)
        if speech:
            self._on_tool_finished(ToolResult.success(speech))

    def _dictate_external(self, text: str) -> None:
        """Handle one utterance while dictating: type it, undo it, send it, or stop."""
        action = parse_dictation_control(text)
        if action == "stop":
            self._stop_external_dictation()
            self._on_tool_finished(ToolResult.success("Done dictating."))
            return
        if action == "undo":
            erased = len(self._last_dictated)
            self._last_dictated = ""

            def worker() -> None:
                try:
                    typing.erase(erased)
                except Exception as exc:  # noqa: BLE001 - worker boundary
                    idle(self._toast, f"I can't type here. {exc}")

            threading.Thread(target=worker, name="external-dictation", daemon=True).start()
            return
        if action == "send":
            self._stop_external_dictation()
            self._run_tool(intents.ToolCall("send_gmail", {}), text)
            return

        typed = format_dictation(text)
        self._last_dictated = typed
        self._log_action(text, "dictation", ok=True)

        def worker() -> None:
            try:
                typing.type_text(typed)
            except Exception:  # noqa: BLE001 - worker boundary
                idle(self._toast, "I can't type here.")

        threading.Thread(target=worker, name="external-dictation", daemon=True).start()

    def ask_ai(self, prompt: str | None = None) -> None:
        self._follow_up_token = None
        self._stop_dictation_for_action()
        if prompt is None:
            prompt = self._get_text(self.transcript_view)
        else:
            if self.conversation_active:
                # Keep the exchange visible while talking hands-free: each new
                # question adds a line instead of wiping the conversation.
                existing = self._get_text(self.transcript_view)
                self._set_text(self.transcript_view, f"{existing}\n{prompt}" if existing else prompt)
            else:
                self._set_text(self.transcript_view, prompt)
        if not prompt:
            self._toast("Speak or type something first.")
            return
        original = prompt
        prompt = hearing.normalize(prompt) or prompt
        if self._mail_flow is not None:
            self._feed_mail_flow(original)
            return
        if self._issue_flow is not None:
            self._feed_issue_flow(original)
            return
        # Dictating into a window wins over starting anything new: the words belong to the document.
        if self._external_dictation:
            self._dictate_external(original)
            return
        project = issueflow.parse_post_issue(prompt)
        if project is not None:
            self._start_issue_flow(project, original)
            return
        email_request = mail.parse_email_command(prompt)
        if email_request is not None and mailflow.wants_dictated_email(email_request.to, email_request.topic):
            self._start_mail_flow(email_request.to, original)
            return
        if repeat.is_repeat_action(prompt):
            if self._last_tool_call is None:
                self._on_tool_finished(ToolResult.failure("There is nothing to do again yet."))
            else:
                self._run_tool(self._last_tool_call, original)
            return
        if intents.is_start_dictation(prompt):
            self._start_external_dictation(prompt)
            return
        created = parse_create(prompt)
        if created is not None:
            name, steps = created
            self.routine_store.add(Routine(name=name, phrases=(name,), steps=tuple(steps)))
            self._log_action(prompt, "routine", detail=f"created {name}")
            joined = ", then ".join(steps)
            self._toast(f"Okay. When you say “{name}” I'll {joined}.")
            return
        if parse_list(prompt):
            self._log_action(prompt, "routine", detail="list")
            names = [r.name for r in self.routine_store.all()]
            self._toast("No routines yet." if not names else "Routines: " + ", ".join(names))
            return
        deleted = parse_delete(prompt)
        if deleted is not None:
            ok = self.routine_store.remove(deleted)
            self._log_action(prompt, "routine", ok=ok, detail=f"delete {deleted}")
            self._toast(f"Deleted the {deleted} routine." if ok else f"There is no {deleted} routine.")
            return
        matched = self.routine_store.match(prompt)
        if matched is not None:
            self._log_action(prompt, "routine", detail=f"run {matched.name}")
            self._start_routine(matched.steps)
            return
        call = intents.route(prompt)
        if call is not None:
            self._run_tool(call, prompt)
            return
        app_name = apps.parse_open_command(prompt)
        if app_name and len(app_name.split()) <= 5:
            self._log_action(prompt, "legacy")
            self._open_app(app_name, prompt)
            return
        if not self.settings.ollama_model:
            self._refresh_ollama_models()
            self._toast("No model is selected yet.")
            return

        email = mail.parse_email_command(prompt)
        if email is not None:
            self._draft_and_act(
                prompt,
                mail.DRAFT_SYSTEM_PROMPT,
                mail.build_prompt(email),
                "Drafting an email",
                lambda answer: self._finish_email_draft(email, answer),
            )
            self._log_action(prompt, "legacy")
            return
        document = documents.parse_document_command(prompt)
        if document is not None:
            self._draft_and_act(
                prompt,
                documents.DRAFT_SYSTEM_PROMPT,
                documents.build_prompt(document),
                "Writing a document",
                lambda answer: self._finish_document_draft(document, answer),
            )
            self._log_action(prompt, "legacy")
            return

        if planner.looks_like_browser_task(prompt):
            self._browse_and_report(prompt)
            return

        if planner.looks_like_command(prompt):
            self._plan_and_run(prompt)
            return

        self._search_record = None
        self._search_source = ""
        self._tool_ran_for_request = False
        heard_prompt = prompt
        self.query_cancel.set()
        cancel_event = threading.Event()
        self.query_cancel = cancel_event
        self._query_generation += 1
        generation = self._query_generation

        # In conversation mode the question joins the running thread of turns;
        # a plain dictation/typed question stays a single-shot prompt.
        if self.conversation_active:
            wanted = requested_language(prompt)
            if wanted is not None:
                self._reply_language = (
                    None if wanted == self._own_language() else wanted
                )
                if is_only_a_language_request(prompt) and self._last_user_prompt:
                    prompt = self._last_user_prompt
            self._conversation_history.add_user(prompt)
            messages: list[dict[str, str]] | None = self._conversation_history.messages()
            self._pending_user_generation = generation
            self._last_user_prompt = prompt
            if self._reply_language and self._reply_language != self._own_language():
                messages[0]["content"] += f"\nAnswer only in {self._reply_language}."
        else:
            messages = None
            self._pending_user_generation = None

        model = self.settings.ollama_model
        # A visible task for the request, and THINKING while hands-free. When OFFLINE
        # (for example an explicit Ask AI from the transcript window) no task is created.
        self._end_query_task("cancelled")
        self.shell.exchange_panel.show_question(prompt)
        task = self.assistant.begin_task(f"Answering: {prompt[:40]}")
        self._query_task_id = task.id if task is not None else None
        self.assistant.prompt_accepted(self.assistant.token())
        self._set_text(self.response_view, "")
        self.ask_button.set_sensitive(False)
        self._set_status(f"Asking {model}…", busy=True)
        self._start_gpu_monitor()

        # Batch streamed chunks so the main loop schedules at most one idle
        # callback per batch instead of one per token chunk.
        pending_chunks: list[str] = []

        def flush_chunks() -> None:
            if pending_chunks:
                batch = "".join(pending_chunks)
                pending_chunks.clear()
                idle(self._append_response, batch, generation, cancel_event)

        def on_chunk(chunk: str) -> None:
            pending_chunks.append(chunk)
            if len(pending_chunks) >= 8:
                flush_chunks()

        def worker() -> None:
            try:
                client = self._ai_client()
                nonlocal messages, prompt
                heard = prompt
                real_model = isinstance(client, (OllamaClient, LlamaCppClient, StrataClient))
                if self.settings.web_search != "never" and real_model:
                    context = self._web_context(prompt, generation, cancel_event)
                    if context:
                        if messages:
                            messages = [*messages[:-1], {**messages[-1], "content": f"{context}\n\n{messages[-1]['content']}"}]
                        else:
                            prompt = f"{context}\n\nQuestion: {prompt}"
                rec = self._search_record
                if rec is not None:
                    self._log_action(heard, rec["route"], args=rec["args"], detail=rec["detail"])
                else:
                    self._log_action(heard, "model")
                if isinstance(client, LlamaCppClient) and client.is_busy():
                    idle(self._on_server_busy, generation, cancel_event)
                if hasattr(client, "is_model_loaded") and client.is_model_loaded(model) is False:
                    idle(self._on_model_loading, model, generation, cancel_event)
                answer = client.generate_stream(
                    model=model,
                    prompt=prompt,
                    cancel_event=cancel_event,
                    on_chunk=on_chunk,
                    messages=messages,
                )
                flush_chunks()
                idle(self._on_query_finished, answer, generation, cancel_event, heard_prompt)
            except OllamaError as exc:
                flush_chunks()
                idle(self._on_query_error, str(exc), generation, cancel_event)

        threading.Thread(target=worker, name=f"ollama-query-{generation}", daemon=True).start()

    def _draft_and_act(
        self,
        prompt: str,
        system_prompt: str,
        user_prompt: str,
        label: str,
        finish: Callable[[str], str],
    ) -> None:
        """One-shot model request whose result is an action, not a chat reply.

        The turn is deliberately kept out of the conversation history: the model
        wrote a file or a mail draft rather than saying anything aloud, so
        remembering it as a spoken turn would make later answers refer to text
        the user never heard.
        """
        self.query_cancel.set()
        cancel_event = threading.Event()
        self.query_cancel = cancel_event
        self._query_generation += 1
        generation = self._query_generation

        self._end_query_task("cancelled")
        self.shell.exchange_panel.show_question(prompt)
        task = self.assistant.begin_task(f"{label}: {prompt[:40]}")
        self._query_task_id = task.id if task is not None else None
        self.assistant.prompt_accepted(self.assistant.token())
        self._set_text(self.response_view, "")
        self.ask_button.set_sensitive(False)
        self._set_status(f"{label}…", busy=True)
        self._start_gpu_monitor()

        def worker() -> None:
            try:
                client = self._ai_client()
                if hasattr(client, "is_model_loaded") and client.is_model_loaded(self.settings.ollama_model) is False:
                    idle(self._on_model_loading, self.settings.ollama_model, generation, cancel_event)
                answer = client.generate_stream(
                    model=self.settings.ollama_model,
                    prompt=user_prompt,
                    cancel_event=cancel_event,
                    on_chunk=lambda chunk: None,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                idle(self._on_draft_finished, answer, generation, cancel_event, finish)
            except OllamaError as exc:
                idle(self._on_draft_error, str(exc), generation, cancel_event)

        threading.Thread(target=worker, name=f"draft-{generation}", daemon=True).start()

    def _on_draft_finished(
        self, answer: str, generation: int, cancel_event: threading.Event, finish: Callable[[str], str]
    ) -> bool:
        if not self._query_is_current(generation, cancel_event):
            return False
        self.ask_button.set_sensitive(True)
        if cancel_event.is_set():
            self._end_query_task("cancelled")
            self._set_status("AI request stopped.")
            return False
        try:
            reply = finish(strip_reasoning(answer))
        except Exception as exc:  # noqa: BLE001 - the mail client or LibreOffice may not be installed
            self._end_query_task("failed", str(exc))
            logging.warning("AI draft failed: %s", exc)
            self._toast(f"Could not complete that: {exc}")
            self._set_status("Draft failed.")
            return False
        self._end_query_task("done")
        display = f"{reply}\n{self._search_source}" if self._search_source else reply
        self.shell.exchange_panel.show_answer(display)
        if not self.assistant.is_active:
            self._toast(reply)
        elif self.conversation_active:
            self._conversation_speak(reply)
        else:
            self.assistant.reply_finished(self.assistant.token())
            self._toast(reply)
        self._set_status(reply)
        return False

    def _on_draft_error(
        self, error: str, generation: int, cancel_event: threading.Event
    ) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        self.ask_button.set_sensitive(True)
        self._end_query_task("failed", error)
        logging.warning("AI request failed: %s", error)
        if self.conversation_active:
            self._fail_assistant("The AI request failed")
        self._toast(error)
        self._set_status(self._conversation_idle_status() if self.conversation_active else "AI request failed.")
        return False

    def _finish_email_draft(self, request: mail.EmailRequest, answer: str) -> str:
        subject, body = mail.parse_draft(answer)
        mail.compose(request.to, subject, body)
        recipient = f" to {request.to}" if request.to else ""
        subject_note = f" — subject “{subject}”" if subject else ""
        return f"I've drafted an email{recipient} in your mail app{subject_note}. Review it and hit send."

    def _finish_document_draft(self, request: documents.DocumentRequest, answer: str) -> str:
        title, body = documents.parse_draft(answer, request.title)
        path = documents.unique_path(title or request.title or request.kind)
        documents.write_odt(path, title, body)
        documents.open_in_libreoffice(path)
        return f"I've written “{title}” and opened it in LibreOffice — it's saved in Voxa Drafts."

    def _plan_and_run(self, prompt: str) -> None:
        self._draft_and_act(
            prompt,
            planner.system_prompt(self.tools),
            prompt,
            "Working on it",
            lambda answer: self._finish_plan(prompt, answer),
        )

    def _finish_plan(self, prompt: str, answer: str) -> str:
        try:
            plan = planner.parse_plan(answer, self.tools)
        except planner.PlanError as exc:
            self._log_action(prompt, "plan", ok=False, detail=str(exc))
            return "I couldn't work out how to do that."

        if not planner.plan_supported(plan, prompt):
            self._log_action(prompt, "plan", ok=False, detail="plan not supported by request")
            return f"I'm not sure what you meant by \u201c{prompt}\u201d."

        def on_step(call, result, ms: int) -> None:
            self._log_action(
                prompt,
                "plan",
                tool=call.tool,
                args=call.args,
                ok=result.ok,
                speech=result.speech,
                detail=result.detail,
                ms=ms,
            )

        result = planner.run_plan(plan, self.tools, on_step=on_step)
        self._maybe_suggest()
        return result

    def _browse_and_report(self, prompt: str) -> None:
        """Run the page-aware planner in one worker thread and speak the result."""
        self.query_cancel.set()
        cancel_event = threading.Event()
        self.query_cancel = cancel_event
        self._query_generation += 1
        generation = self._query_generation

        self._end_query_task("cancelled")
        self.shell.exchange_panel.show_question(prompt)
        task = self.assistant.begin_task(f"Browsing: {prompt[:40]}")
        self._query_task_id = task.id if task is not None else None
        self.assistant.prompt_accepted(self.assistant.token())
        self._set_text(self.response_view, "")
        self.ask_button.set_sensitive(False)
        self._set_status("Browsing…", busy=True)
        self._start_gpu_monitor()

        def on_step(call: intents.ToolCall, result: ToolResult, ms: int) -> None:
            self._log_action(
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
            idle(self._show_browser_step, detail, generation, cancel_event)

        def worker() -> None:
            try:
                def ask_model(messages: list[dict]) -> str:
                    return self._ai_client().generate_stream(
                        model=self.settings.ollama_model,
                        prompt=messages[-1]["content"],
                        messages=messages,
                        cancel_event=cancel_event,
                        on_chunk=lambda chunk: None,
                    )

                answer = planner.run_browser_task(
                    prompt,
                    ask_model,
                    get_session(),
                    self.tools,
                    on_step=on_step,
                    should_stop=cancel_event.is_set,
                    on_caption=on_caption,
                )
                idle(self._on_draft_finished, answer, generation, cancel_event, lambda text: text)
            except OllamaError as exc:
                idle(self._on_draft_error, str(exc), generation, cancel_event)
            except Exception:
                idle(
                    self._on_draft_error,
                    "I lost contact with my browser. Say that again and I'll reopen it.",
                    generation,
                    cancel_event,
                )

        threading.Thread(target=worker, name=f"browse-{generation}", daemon=True).start()

    def _show_browser_step(self, detail: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        self._set_status(detail, busy=True)
        self.assistant_model.set_state(AssistantState.WORKING, detail)
        return False

    def _web_context(self, prompt: str, generation: int, cancel_event: threading.Event) -> str:
        self._search_record = None
        self._search_source = ""
        previous = self._last_search_query
        if previous is not None and time.time() - self._last_search_at > 180:
            previous = None
        query = websearch.search_query_for(prompt, self.settings.web_search, previous)
        if not query:
            return ""
        self._last_search_query = query
        self._last_search_at = time.time()
        idle(self._on_web_search, query, generation, cancel_event)
        try:
            results = websearch.search(query)
        except Exception:  # noqa: BLE001 - offline or blocked: answer without the web
            idle(self._on_web_unavailable, generation, cancel_event)
            self._search_record = websearch.search_record_fields(query, [], True)
            return ""
        self._search_record = websearch.search_record_fields(query, results, False)
        if results:
            self._search_source = websearch.source_line(results)
            return websearch.format_for_prompt(query, results)
        return ""

    def _on_web_search(self, query: str, generation: int, cancel_event: threading.Event) -> bool:
        if self._query_is_current(generation, cancel_event) and not cancel_event.is_set():
            self._set_status(f"Searching the web for “{query}”…", busy=True)
            self.assistant_model.set_state(AssistantState.THINKING, "Searching the web…")
        return False

    def _on_web_unavailable(self, generation: int, cancel_event: threading.Event) -> bool:
        if self._query_is_current(generation, cancel_event) and not cancel_event.is_set():
            self._toast("Couldn't reach the web, so Voxa is answering from the model's own knowledge.")
        return False

    def _on_server_busy(self, generation: int, cancel_event: threading.Event) -> bool:
        if self._query_is_current(generation, cancel_event) and not cancel_event.is_set():
            note = "The AI server is busy with another request. Voxa will answer as soon as it is free."
            self._set_status(note, busy=True)
            self._toast(note)
            self.assistant_model.set_state(AssistantState.THINKING, "AI server busy — waiting in line")
        return False

    def _on_model_loading(self, model: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        self._loading_model_since = time.monotonic()
        self._loading_model_name = short_model_name(model)
        self._loading_generation = generation
        self._loading_cancel = cancel_event
        self._show_model_loading()
        self._loading_source = GLib.timeout_add(1000, self._model_loading_tick)
        return False

    def _show_model_loading(self) -> None:
        seconds = int(time.monotonic() - self._loading_model_since) if self._loading_model_since is not None else 0
        detail = f"Loading {self._loading_model_name}… {seconds}s"
        self._set_status(detail, busy=True)
        self.assistant_model.set_state(AssistantState.WORKING, detail)

    def _model_loading_tick(self) -> bool:
        if self._loading_model_since is None:
            return False
        if self._loading_cancel is not None and self._loading_cancel.is_set():
            self._stop_model_loading()
            return False
        if self._loading_generation != self._query_generation:
            self._stop_model_loading()
            return False
        self._show_model_loading()
        return True

    def _stop_model_loading(self) -> None:
        if self._loading_source:
            GLib.source_remove(self._loading_source)
            self._loading_source = 0
        self._loading_model_since = None
        self._loading_model_name = ""
        self._loading_generation = -1
        self._loading_cancel = None

    def _warm_up_model(self, model: str) -> None:
        # Loading a model into memory does not need the microphone, so this runs
        # whether the assistant is ACTIVE or OFFLINE. A 1-token request makes the
        # cold-start wait happen the moment the user picks the model.
        if not model:
            return
        client = self._ai_client()
        if not hasattr(client, "is_model_loaded") or client.is_model_loaded(model) is not False:
            return
        cancel_event = threading.Event()
        self.query_cancel = cancel_event
        self._query_generation += 1
        generation = self._query_generation
        idle(self._on_model_loading, model, generation, cancel_event)

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
                    idle(self._warm_up_finished, generation, cancel_event)
            except OllamaError as exc:
                if not cancel_event.is_set():
                    idle(self._warm_up_failed, str(exc), generation, cancel_event)

        threading.Thread(target=worker, name=f"warm-up-{generation}", daemon=True).start()

    def _warm_up_finished(self, generation: int, cancel_event: threading.Event) -> bool:
        if not self._query_is_current(generation, cancel_event):
            return False
        self._stop_model_loading()
        self._set_status("Ready")
        return False

    def _warm_up_failed(self, error: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._query_is_current(generation, cancel_event):
            return False
        self._stop_model_loading()
        self._toast(error)
        self._set_status("Ready")
        return False

    def _query_is_current(self, generation: int, cancel_event: threading.Event) -> bool:
        return generation == self._query_generation and cancel_event is self.query_cancel

    def _append_response(self, batch: str, generation: int, cancel_event: threading.Event) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        if self._loading_model_since is not None:
            self._stop_model_loading()
        buffer = self.response_view.get_buffer()
        buffer.insert(buffer.get_end_iter(), batch)
        self._scroll_to_end(self.response_view)
        self.shell.exchange_panel.show_answer(strip_reasoning(self._get_text(self.response_view)))
        self._early_speech_feed(batch, generation)
        return False

    # ---- speak while the model is still writing -------------------------------------------------

    def _early_speech_feed(self, batch: str, generation: int) -> None:
        """Hand streamed text to the sentence feeder and start speaking the first complete sentence."""
        if not self.conversation_active:
            return
        state = getattr(self, "_early", None)
        if state is None or state["generation"] != generation:
            if state is not None and state.get("next") is not None:
                state["next"].discard()  # a piece prepared for an answer that was abandoned
            state = {
                "next": None,
                "generation": generation,
                "feeder": SentenceFeeder(),
                "raw": "",
                "queue": [],
                "playing": False,
                "finished": False,
                "started": False,
                "disabled": is_thinking_model(self.settings.ollama_model),
                "token": None,
                "reply_id": None,
            }
            self._early = state
        if state["disabled"]:
            return
        state["raw"] += batch
        if looks_like_reasoning(state["raw"]):
            # A scratchpad is streaming: nothing has been spoken yet (or it would not be here), so fall back
            # to speaking the cleaned answer at the end.
            if not state["started"]:
                state["disabled"] = True
            return
        state["queue"].extend(state["feeder"].feed(batch))
        self._early_speech_pump()

    def _early_speech_pump(self) -> None:
        """Play the next piece, or — while one is playing — fetch the one after it so there is no gap."""
        state = getattr(self, "_early", None)
        if state is None:
            return
        live = self._live_client is not None and self.settings.face_mode == "live"
        rate, voice = self.settings.tts_rate, self._reply_voice()
        if state["playing"]:
            if state.get("next") is None and state["queue"]:
                text = " ".join(state["queue"])
                state["queue"].clear()
                state["next"] = self.speech.prepare(text, rate, voice, on_ready=self._live_prepare if live else None)
                state["next_text"] = text
            return
        item = state.get("next")
        item_text = state.get("next_text") or ""
        state["next"] = None
        state["next_text"] = ""
        text = ""
        if item is None:
            if not state["queue"]:
                return
            text = " ".join(state["queue"])
            state["queue"].clear()
            if live or state["started"]:
                # High needs the whole piece before it plays; later pieces are fetched as files as well.
                item = self.speech.prepare(text, rate, voice, on_ready=self._live_prepare if live else None)
        state["playing"] = True
        piece = item_text or text
        if piece:
            self._recent_spoken = self._speaking_text
            self._speaking_text = piece
        # Grace period per piece: speaker onset is the loudest moment.
        self._speaking_since = time.monotonic()
        self._barge_gate.reset()
        if not state["started"]:
            state["started"] = True
            if self.conversation is not None:
                self.conversation.mute()
            state["token"] = self.assistant.token()
            self.assistant.reply_started(state["token"])
            state["reply_id"] = self.assistant.current_reply()
            self._set_status("Speaking…", busy=True)
        token, generation = state["token"], state["generation"]
        self._reset_word_timeline()
        self.shell.set_speech_clock(self._lip_clock)
        callbacks = {
            "on_started": lambda: idle(self._for_session(token, self._set_status), "Speaking…", True),
            "on_done": lambda: idle(self._for_session(token, self._early_speech_part_done), generation),
            "on_error": lambda error: idle(self._for_session(token, self._early_speech_part_done), generation),
            "on_words": lambda words: idle(self.shell.set_word_timeline, words),
        }
        if item is not None:
            self.speech.speak_prepared(item, before_play=self._live_activate if live else None, **callbacks)
        else:
            # Medium/Low, first piece: stream it, which starts soonest.
            self.speech.speak(text, rate, voice, **callbacks)
        if state["queue"]:
            # More text is already waiting: start fetching it now, while this piece plays.
            self._early_speech_pump()

    def _early_speech_part_done(self, generation: int) -> bool:
        state = getattr(self, "_early", None)
        if state is None or state["generation"] != generation:
            return False
        state["playing"] = False
        if state["queue"] or state.get("next") is not None:
            self._early_speech_pump()
        elif state["finished"]:
            self._early = None
            self._on_conversation_speech_done(state["reply_id"])
        # otherwise the model is still writing: the next complete sentence restarts playback
        return False

    def _early_speech_finish(self, spoken: str, generation: int) -> bool:
        """The answer is complete. True when early speech is handling it (the caller must not speak it again)."""
        state = getattr(self, "_early", None)
        if state is None or state["generation"] != generation or not state["started"]:
            self._early = None
            return False
        state["finished"] = True
        state["queue"].extend(state["feeder"].finish(spoken))
        if state["playing"] or state["queue"] or state.get("next") is not None:
            self._early_speech_pump()
        else:
            self._early = None
            self._on_conversation_speech_done(state["reply_id"])
        return True

    def _lip_clock(self) -> float:
        """The audio position the face should show: a little AHEAD of what the player reports, because the
        sound card and the screen each add delay and the lips otherwise trail the voice."""
        played = self.speech.position()
        return played + self.settings.lip_sync_lead_ms / 1000.0 if played > 0.0 else 0.0

    def _reset_word_timeline(self) -> None:
        """Each spoken piece has its own clock starting at zero: forget the previous piece's word times."""
        view = getattr(self.shell, "assistant_view", None)
        renderer = getattr(view, "_photo_renderer", None) if view is not None else None
        reset = getattr(renderer, "reset_word_timeline", None)
        if reset is not None:
            reset()
        self.shell.reset_word_timeline()

    def _on_query_finished(
        self, answer: str, generation: int, cancel_event: threading.Event, heard: str = ""
    ) -> bool:
        if not self._query_is_current(generation, cancel_event):
            return False
        self._stop_model_loading()
        self.ask_button.set_sensitive(True)
        if (
            self.conversation_active
            and self._conversation_history
            and self._pending_user_generation == generation
        ):
            # This turn was abandoned (a spoken "cancel" or the Stop button),
            # or its answer was lost: undo the user turn pushed when it started
            # so the model's history stays coherent.
            self._conversation_history.drop_last()
        self._pending_user_generation = None
        if cancel_event.is_set():
            self._end_query_task("cancelled")
            self._set_status("AI request stopped.")
            return False
        self._end_query_task("done")
        self._set_status("AI response complete.")
        # A thinking model streams its scratchpad in with the reply. Replace
        # what was streamed with just the answer, so the chain of thought is
        # neither left on screen nor read aloud in conversation mode.
        spoken = strip_reasoning(answer)
        if claims_action(spoken) and not self._tool_ran_for_request:
            # The model talked as if it had acted, but no tool ran: answer honestly.
            self._log_action(heard, "false_claim", detail=first_sentences(answer, 1))
            spoken = FALSE_CLAIM_REPLY
        if spoken != answer:
            buffer = self.response_view.get_buffer()
            buffer.set_text(spoken)
            self._scroll_to_end(self.response_view)
        self.shell.exchange_panel.show_answer(spoken)
        replaced = spoken == FALSE_CLAIM_REPLY
        if spoken and self.conversation_active:
            self._conversation_history.add_assistant(spoken)
            if replaced:
                self._early = None  # what was said early is being corrected: say the honest answer in full
                self._conversation_speak(spoken)
            elif not self._early_speech_finish(spoken, generation):
                self._conversation_speak(spoken)
        else:
            if self.conversation_active:  # nothing to say: back to waiting
                waiting = self.conversation is not None and self.conversation.waiting_for_prompt
                self.assistant.reply_finished(self.assistant.token(), waiting_for_prompt=waiting)
            if answer and self.settings.auto_speak:
                self.speak_response()
        return False

    def _on_query_error(
        self, error: str, generation: int, cancel_event: threading.Event
    ) -> bool:
        if not self._query_is_current(generation, cancel_event) or cancel_event.is_set():
            return False
        self._stop_model_loading()
        self.ask_button.set_sensitive(True)
        if (
            self.conversation_active
            and self._conversation_history
            and self._pending_user_generation == generation
        ):
            # The turn produced no usable answer, so it must not sit in the
            # conversation history as an unanswered question with no reply.
            self._conversation_history.drop_last()
        self._pending_user_generation = None
        self._end_query_task("failed", error)
        logging.warning("AI request failed: %s", error)
        if self.conversation_active:
            self._fail_assistant("The AI request failed")
        self._toast(error)
        self._set_status(self._conversation_idle_status() if self.conversation_active else "AI request failed.")
        return False

    def speak_response(self) -> None:
        text = self._get_text(self.response_view) or self._get_text(self.transcript_view)
        if not text:
            self._toast("There is no text to speak.")
            return
        self._set_status("Starting speech…", busy=True)
        self.shell.set_speech_clock(self._lip_clock)
        self.speech.speak(
            text,
            self.settings.tts_rate,
            self.settings.tts_voice,
            on_started=lambda: idle(self._set_status, "Speaking…", True),
            on_done=lambda: idle(self._set_status, "Ready"),
            on_error=lambda error: idle(self._speech_error, error),
            on_words=lambda words: idle(self.shell.set_word_timeline, words),
            **(
                {"before_play": self._live_before_play}
                if self._live_client is not None and self.settings.face_mode == "live"
                else {}
            ),
        )

    def _speech_error(self, error: str) -> bool:
        self._set_status("Speech playback failed.")
        self._toast(error)
        return False

    def stop_current_work(self) -> None:
        # The OFFLINE kill switch: stops the microphone, wake-word monitoring, dictation,
        # speech and the running request, cancels tasks, and makes late callbacks stale.
        self.assistant.go_offline()
        self._stop_external_dictation()
        self._issue_flow = None
        self._mail_flow = None
        self._stop_ai_server_async()
        if self._installing:
            self._install_cancel.set()
        self.ask_button.set_sensitive(True)
        self._stop_model_loading()
        self._live_reset()
        self._set_status("Stopped.")

    @staticmethod
    def _gtk_theme_prefers_dark() -> bool:
        if host_theme_is_dark():
            return True
        gtk_settings = Gtk.Settings.get_default()
        if gtk_settings is None:
            return False
        if gtk_settings.find_property("gtk-application-prefer-dark-theme") is not None:
            if bool(gtk_settings.get_property("gtk-application-prefer-dark-theme")):
                return True
        if gtk_settings.find_property("gtk-theme-name") is not None:
            theme_name = str(gtk_settings.get_property("gtk-theme-name") or "")
            return "dark" in theme_name.casefold()
        return False

    def _apply_appearance(self) -> None:
        appearance = self.settings.appearance
        if appearance == "dark":
            scheme = Adw.ColorScheme.FORCE_DARK
        elif appearance == "light":
            scheme = Adw.ColorScheme.FORCE_LIGHT
        elif self.style_manager.get_system_supports_color_schemes():
            scheme = Adw.ColorScheme.PREFER_LIGHT
        elif self._gtk_theme_prefers_dark():
            scheme = Adw.ColorScheme.PREFER_DARK
        else:
            scheme = Adw.ColorScheme.PREFER_LIGHT
        self.style_manager.set_color_scheme(scheme)

    def show_preferences(self) -> None:
        dialog = Adw.PreferencesDialog()
        dialog.set_title("Preferences")
        dialog.add_css_class("voxa-preferences")
        # Tall and moderately wide: every page fits without scrolling, and the
        # rows are not stretched into long, hard-to-scan lines.
        dialog.set_content_width(680)
        dialog.set_content_height(640)

        # Three pages with short names (long names are cut off in the page
        # switcher), each split into titled groups.
        general_page = Adw.PreferencesPage(title="General", icon_name="preferences-system-symbolic")

        appearance_group = Adw.PreferencesGroup(title="Appearance")
        general_page.add(appearance_group)
        appearance_row = Adw.ComboRow(
            title="Color scheme",
            subtitle="Follow the desktop, or always use light or dark",
        )
        appearance_row.set_model(Gtk.StringList.new(APPEARANCE_LABELS))
        appearance_row.set_selected(APPEARANCE_VALUES.index(self.settings.appearance))
        appearance_group.add(appearance_row)

        choices = character_choices()
        character_ids = [character_id for character_id, _label in choices]
        self._character_ids = character_ids
        character_row = Adw.ComboRow(
            title="Assistant character",
            subtitle="An avatar with its matching voice, or the Classic badge",
        )
        character_row.set_model(Gtk.StringList.new([label for _character_id, label in choices]))
        character_row.set_selected(
            character_ids.index(self.settings.character_id) if self.settings.character_id in character_ids else 0
        )
        appearance_group.add(character_row)

        conversation_group = Adw.PreferencesGroup(
            title="Conversation",
            description="Say the wake word, then your request. Voxa acts on it or answers aloud.",
        )
        general_page.add(conversation_group)
        wake_word_row = Adw.EntryRow(title="Wake word")
        wake_word_row.set_text(self.settings.wake_word)
        conversation_group.add(wake_word_row)
        auto_speak_row = Adw.SwitchRow(
            title="Speak answers aloud",
            subtitle="Read each reply out as soon as it is ready",
        )
        auto_speak_row.set_active(self.settings.auto_speak)
        conversation_group.add(auto_speak_row)
        dialog.add(general_page)

        speech_page = Adw.PreferencesPage(title="Speech", icon_name="audio-input-microphone-symbolic")

        listening_group = Adw.PreferencesGroup(title="Listening")
        speech_page.add(listening_group)
        mic_names = [device.name for device in self.devices] or ["Default microphone"]
        selected_mic = next(
            (index for index, device in enumerate(self.devices) if device.identifier == self.settings.microphone_id),
            0,
        )
        # The selected name is already shown in the row; the subtitle says what the
        # setting is for, and the tooltip carries the full name when it is cut off.
        mic_row = Adw.ComboRow(title="Microphone", subtitle="Used for dictation and conversation")
        mic_row.set_model(Gtk.StringList.new(mic_names))
        mic_row.set_factory(string_item_factory(wrap=False, width_chars=30))
        mic_row.set_list_factory(string_item_factory(wrap=True, width_chars=56))
        mic_row.set_selected(selected_mic)
        mic_row.set_tooltip_text(mic_names[selected_mic])

        def update_mic_tooltip(row, _property) -> None:
            row.set_tooltip_text(mic_names[min(row.get_selected(), len(mic_names) - 1)])

        mic_row.connect("notify::selected", update_mic_tooltip)
        listening_group.add(mic_row)

        whisper_row = Adw.ComboRow(
            title="Recognition model",
            subtitle="Whisper size: smaller starts faster, larger hears more accurately",
        )
        whisper_model = Gtk.StringList.new(WHISPER_MODELS)
        whisper_row.set_model(whisper_model)
        try:
            whisper_row.set_selected(WHISPER_MODELS.index(self.settings.whisper_model))
        except ValueError:
            whisper_row.set_selected(1)
        listening_group.add(whisper_row)

        voice_group = Adw.PreferencesGroup(title="Voice")
        speech_page.add(voice_group)
        voice_row = Adw.ComboRow(
            title="Voice",
            subtitle="Natural online voice, with an offline fallback",
        )
        voice_row.set_model(Gtk.StringList.new([label for label, _voice in TTS_VOICES]))
        voice_ids = [voice_id for _label, voice_id in TTS_VOICES]
        voice_row.set_selected(
            voice_ids.index(self.settings.tts_voice) if self.settings.tts_voice in voice_ids else 0
        )
        voice_group.add(voice_row)

        def _sync_voice_from_character(row, _property) -> None:
            selected_index = row.get_selected()
            character_id = character_ids[min(selected_index, len(character_ids) - 1)]
            avatar = get_avatar(character_id)
            if avatar is not None and avatar.voice in voice_ids:
                voice_row.set_selected(voice_ids.index(avatar.voice))

        character_row.connect("notify::selected", _sync_voice_from_character)
        rate_row = Adw.SpinRow.new_with_range(80, 350, 5)
        rate_row.set_title("Speaking rate")
        rate_row.set_subtitle("Words per minute")
        rate_row.set_value(self.settings.tts_rate)
        voice_group.add(rate_row)
        dialog.add(speech_page)

        ai_page = Adw.PreferencesPage(title="Local AI", icon_name="system-run-symbolic")

        ai_group = Adw.PreferencesGroup(
            title="Model server",
            description="The local server that answers questions and plans tasks.",
        )
        ai_page.add(ai_group)
        backend_row = Adw.ComboRow(
            title="Backend",
            subtitle="llama.cpp-compatible server, or Ollama",
        )
        backend_row.set_model(Gtk.StringList.new(["llama.cpp", "Ollama", "Strata"]))
        backend_row.set_selected({"llamacpp": 0, "ollama": 1, "strata": 2}.get(self.settings.ai_backend, 0))
        ai_group.add(backend_row)

        model_names = self.ollama_models or ["No models found"]
        ai_row = Adw.ComboRow(title="Model", subtitle=self._hardware_summary)
        ai_row.set_model(Gtk.StringList.new(model_names))
        ai_row.set_factory(string_item_factory(wrap=False, width_chars=30))
        ai_row.set_list_factory(string_item_factory(wrap=True, width_chars=56))
        if self.settings.ollama_model in model_names:
            ai_row.set_selected(model_names.index(self.settings.ollama_model))
        ai_group.add(ai_row)

        llamacpp_row = Adw.EntryRow(title="Server address")
        llamacpp_row.set_text(self.settings.llamacpp_url)
        llamacpp_row.set_visible(self.settings.ai_backend == "llamacpp")
        ai_group.add(llamacpp_row)

        endpoint_row = Adw.EntryRow(title="Ollama address")
        endpoint_row.set_text(self.settings.ollama_url)
        endpoint_row.set_visible(self.settings.ai_backend == "ollama")
        ai_group.add(endpoint_row)

        strata_row = Adw.EntryRow(title="Strata address")
        strata_row.set_text(self.settings.strata_url)
        strata_row.set_visible(self.settings.ai_backend == "strata")
        ai_group.add(strata_row)

        answers_group = Adw.PreferencesGroup(title="Answers")
        ai_page.add(answers_group)
        web_search_row = Adw.ComboRow(
            title="Web search",
            subtitle="Sends the question text to DuckDuckGo when the answer needs facts from outside the model",
        )
        web_search_row.set_model(Gtk.StringList.new(WEB_SEARCH_LABELS))
        web_search_row.set_selected(WEB_SEARCH_VALUES.index(self.settings.web_search))
        answers_group.add(web_search_row)

        # Installing and pulling models is an Ollama-only convenience; a
        # llama.cpp server serves whichever GGUF the user started it with.
        # The group always exists so switching the backend combo shows or hides
        # it immediately instead of only on the next dialog open.
        ollama_group = Adw.PreferencesGroup(title="Ollama")
        ollama_group.set_visible(self.settings.ai_backend == "ollama")
        ai_page.add(ollama_group)
        install_row = Adw.ActionRow(
            title="Install or update Ollama",
            subtitle="Downloads the installer from ollama.com and runs it with a password prompt",
        )
        install_button = Gtk.Button(label="Install", valign=Gtk.Align.CENTER)
        install_button.connect("clicked", lambda *_: self._start_ollama_install())
        install_row.add_suffix(install_button)
        ollama_group.add(install_row)

        manage_row = Adw.ActionRow(title="Pull or remove models")
        manage_button = Gtk.Button(label="Manage models…", valign=Gtk.Align.CENTER)
        manage_button.connect("clicked", lambda *_: self._show_model_manager())
        manage_row.add_suffix(manage_button)
        ollama_group.add(manage_row)

        def update_backend_rows(*_):
            selected = backend_row.get_selected()
            llamacpp_row.set_visible(selected == 0)
            endpoint_row.set_visible(selected == 1)
            strata_row.set_visible(selected == 2)
            ollama_group.set_visible(selected == 1)

        backend_row.connect("notify::selected", update_backend_rows)
        dialog.add(ai_page)

        dialog.connect(
            "closed",
            self._save_preferences,
            appearance_row,
            whisper_row,
            mic_row,
            backend_row,
            ai_row,
            llamacpp_row,
            endpoint_row,
            strata_row,
            auto_speak_row,
            web_search_row,
            wake_word_row,
            character_row,
            voice_row,
            rate_row,
        )
        dialog.present(self)

    def _save_preferences(
        self,
        _dialog,
        appearance_row,
        whisper_row,
        mic_row,
        backend_row,
        ai_row,
        llamacpp_row,
        endpoint_row,
        strata_row,
        auto_speak_row,
        web_search_row,
        wake_word_row,
        character_row,
        voice_row,
        rate_row,
    ) -> None:
        new_whisper = WHISPER_MODELS[whisper_row.get_selected()]
        if self.devices:
            device = self.devices[min(mic_row.get_selected(), len(self.devices) - 1)]
            self.settings.microphone_id = device.identifier
            self.settings.microphone_name = device.name
        if self.ollama_models:
            self.settings.ollama_model = self.ollama_models[min(ai_row.get_selected(), len(self.ollama_models) - 1)]
        self.settings.ai_backend = {0: "llamacpp", 1: "ollama", 2: "strata"}.get(
            backend_row.get_selected(), "llamacpp"
        )
        self.settings.llamacpp_url = llamacpp_row.get_text().strip()
        self.settings.ollama_url = endpoint_row.get_text().strip()
        self.settings.strata_url = strata_row.get_text().strip()
        self.settings.auto_speak = auto_speak_row.get_active()
        self.settings.web_search = WEB_SEARCH_VALUES[min(web_search_row.get_selected(), len(WEB_SEARCH_VALUES) - 1)]
        self.settings.wake_word = wake_word_row.get_text().strip()
        self.settings.appearance = APPEARANCE_VALUES[appearance_row.get_selected()]
        self.settings.character_id = self._character_ids[
            min(character_row.get_selected(), len(self._character_ids) - 1)
        ]
        self.settings.tts_voice = TTS_VOICES[voice_row.get_selected()][1]
        self.settings.tts_rate = int(rate_row.get_value())
        self.config_store.save(self.settings)
        self._apply_appearance()
        assistant_view = getattr(self.shell, "assistant_view", None)
        if assistant_view is not None:
            assistant_view.set_character(self.settings.character_id)
        if new_whisper != self.settings.whisper_model:
            self._load_whisper(new_whisper)
        self._refresh_ollama_models()
        # Keep the main-shell backend dropdown in step with what was just saved.
        self.shell.set_backend(self.settings.ai_backend)

    def show_shortcuts(self) -> None:
        builder = Gtk.Builder.new_from_string(
            """
            <interface>
              <object class="GtkShortcutsWindow" id="shortcuts">
                <property name="modal">true</property>
                <child>
                  <object class="GtkShortcutsSection">
                    <property name="section-name">general</property>
                    <property name="title">General</property>
                    <child>
                      <object class="GtkShortcutsGroup">
                        <property name="title">Actions</property>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Start or stop dictation</property><property name="accelerator">&lt;Control&gt;r</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Start or stop conversation mode</property><property name="accelerator">&lt;Control&gt;&lt;Shift&gt;r</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Ask AI</property><property name="accelerator">&lt;Control&gt;Return</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Copy transcript</property><property name="accelerator">&lt;Control&gt;&lt;Shift&gt;c</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Copy AI response</property><property name="accelerator">&lt;Control&gt;&lt;Shift&gt;v</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Clear</property><property name="accelerator">&lt;Control&gt;l</property></object></child>
                        <child><object class="GtkShortcutsShortcut"><property name="title">Preferences</property><property name="accelerator">&lt;Control&gt;comma</property></object></child>
                      </object>
                    </child>
                  </object>
                </child>
              </object>
            </interface>
            """,
            -1,
        )
        window = builder.get_object("shortcuts")
        window.set_transient_for(self)
        window.present()

    def do_close_request(self) -> bool:
        self._closing = True
        getattr(self, "_welcome_cancel", threading.Event()).set()
        self.shell.close_focus_window()
        self._live_generation = getattr(self, "_live_generation", 0) + 1
        self._release_face_server()
        self.stop_current_work()
        self._stop_gpu_monitor()
        self.audio.stop()
        self.config_store.save(self.settings)
        return False
