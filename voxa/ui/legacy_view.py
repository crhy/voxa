"""The legacy dictation/transcript view.

It is the original interface, kept fully working: it is built up front but only
attached to a secondary window opened from the menu or the keyboard shortcuts.
Every widget it creates is handed back to the window under the same attribute
name it used before, so the many methods that drive these widgets keep working.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gdk, Gtk  # noqa: E402


@dataclass
class LegacyCallbacks:
    """The window handlers the legacy widgets are connected to."""

    record_toggled: Callable[[Gtk.ToggleButton], None]
    conversation_toggled: Callable[[Gtk.ToggleButton], None]
    copy: Callable[[], None]
    copy_reply: Callable[[], None]
    clear: Callable[[], None]
    ask: Callable[[], None]
    speak: Callable[[], None]
    stop: Callable[[], None]
    model_selected: Callable[[Gtk.DropDown, str], None]


class LegacyView:
    """Header bar, status strip, two editors and an action bar."""

    WIDGET_NAMES: tuple[str, ...] = (
        "record_button",
        "conversation_button",
        "progress",
        "status_box",
        "status_spinner",
        "status_label",
        "level",
        "gpu_box",
        "gpu_level",
        "gpu_label",
        "model_combo",
        "transcript_view",
        "response_view",
        "copy_button",
        "copy_response_button",
        "clear_button",
        "ask_button",
        "speak_button",
        "stop_button",
    )

    def __init__(self, callbacks: LegacyCallbacks) -> None:
        self.callbacks = callbacks

        self.view = Adw.ToolbarView()

        header = Adw.HeaderBar()
        header.set_title_widget(Adw.WindowTitle(title="Voxa", subtitle="Your personal voice assistant"))
        self.view.add_top_bar(header)

        self.record_button = Gtk.ToggleButton(label="Dictate")
        self.record_button.set_tooltip_text("Start or stop dictation (Ctrl+R)")
        self.record_button.connect("toggled", callbacks.record_toggled)
        header.pack_start(self.record_button)

        self.conversation_button = Gtk.ToggleButton(label="Conversation")
        self.conversation_button.set_tooltip_text(
            "Actively listen for the wake word, then transcribe and ask AI automatically (Ctrl+Shift+R)"
        )
        self.conversation_button.connect("toggled", callbacks.conversation_toggled)
        header.pack_start(self.conversation_button)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.view.set_content(root)

        self.progress = Gtk.ProgressBar()
        self.progress.set_visible(False)
        root.append(self.progress)

        self.status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.status_box.add_css_class("status-strip")
        self.status_box.set_margin_top(10)
        self.status_box.set_margin_bottom(8)
        self.status_box.set_margin_start(18)
        self.status_box.set_margin_end(18)
        self.status_spinner = Adw.Spinner()
        self.status_spinner.set_visible(False)
        self.status_label = Gtk.Label(label="Starting…", xalign=0)
        self.status_label.set_hexpand(True)
        self.level = Gtk.LevelBar()
        self.level.set_min_value(0)
        self.level.set_max_value(4000)
        self.level.set_value(0)
        self.level.set_size_request(150, -1)
        self.level.set_tooltip_text("Microphone level")

        self.gpu_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.gpu_box.set_visible(False)
        gpu_caption = Gtk.Label(label="GPU")
        gpu_caption.add_css_class("dim-label")
        self.gpu_level = Gtk.LevelBar()
        self.gpu_level.set_min_value(0)
        self.gpu_level.set_max_value(100)
        self.gpu_level.set_size_request(80, -1)
        self.gpu_label = Gtk.Label(label="0%")
        self.gpu_label.set_width_chars(4)
        self.gpu_box.append(gpu_caption)
        self.gpu_box.append(self.gpu_level)
        self.gpu_box.append(self.gpu_label)

        self.model_combo = Gtk.DropDown()
        self.model_combo.set_visible(False)
        self.model_combo.add_css_class("model-select")
        self.model_combo.set_tooltip_text("Model used by Ask AI")
        self.model_combo.connect("notify::selected", callbacks.model_selected)

        self.status_box.append(self.status_spinner)
        self.status_box.append(self.status_label)
        self.status_box.append(self.gpu_box)
        self.status_box.append(self.model_combo)
        self.status_box.append(self.level)
        root.append(self.status_box)
        self.status_css_provider = self._install_status_css()

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_wide_handle(True)
        paned.set_position(525)
        paned.set_vexpand(True)
        paned.set_start_child(self._build_editor("Transcript", editable=True, transcript=True))
        paned.set_end_child(self._build_editor("AI response", editable=False, transcript=False))
        root.append(paned)

        action_bar = Gtk.ActionBar()
        action_bar.set_revealed(True)
        self.view.add_bottom_bar(action_bar)

        self.copy_button = Gtk.Button(label="Copy")
        self.copy_button.set_tooltip_text("Copy the transcript (Ctrl+Shift+C)")
        self.copy_button.connect("clicked", lambda *_: callbacks.copy())
        action_bar.pack_start(self.copy_button)

        self.copy_response_button = Gtk.Button(label="Copy Reply")
        self.copy_response_button.set_tooltip_text("Copy the AI response")
        self.copy_response_button.connect("clicked", lambda *_: callbacks.copy_reply())
        action_bar.pack_start(self.copy_response_button)

        self.clear_button = Gtk.Button(label="Clear")
        self.clear_button.connect("clicked", lambda *_: callbacks.clear())
        action_bar.pack_start(self.clear_button)

        self.ask_button = Gtk.Button(label="Ask AI")
        self.ask_button.add_css_class("suggested-action")
        self.ask_button.connect("clicked", lambda *_: callbacks.ask())
        action_bar.pack_end(self.ask_button)

        self.speak_button = Gtk.Button(label="Speak")
        self.speak_button.connect("clicked", lambda *_: callbacks.speak())
        action_bar.pack_end(self.speak_button)

        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.connect("clicked", lambda *_: callbacks.stop())
        action_bar.pack_end(self.stop_button)

    def _build_editor(self, title: str, *, editable: bool, transcript: bool) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_top(6)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)

        heading = Gtk.Label(label=title, xalign=0)
        heading.add_css_class("heading")
        box.append(heading)

        view = Gtk.TextView()
        view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        view.set_editable(editable)
        view.set_cursor_visible(editable)
        view.set_top_margin(12)
        view.set_bottom_margin(12)
        view.set_left_margin(12)
        view.set_right_margin(12)
        view.add_css_class("card")
        view.add_css_class("document")
        view.set_vexpand(True)
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(view)
        scroller.set_vexpand(True)
        box.append(scroller)

        if transcript:
            self.transcript_view = view
        else:
            self.response_view = view
        return box

    @staticmethod
    def _install_status_css() -> Gtk.CssProvider | None:
        display = Gdk.Display.get_default()
        if display is None:
            return None
        provider = Gtk.CssProvider()
        provider.load_from_data(
            b"""
            .status-strip { background-color: @window_bg_color; }
            .model-select { min-width: 170px; }
            """
        )
        Gtk.StyleContext.add_provider_for_display(
            display,
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )
        return provider
