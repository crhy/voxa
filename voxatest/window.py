from __future__ import annotations

import math
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path

import gi

gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Pango", "1.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

from .config import DEFAULT_DATA_PATH, default_failure_report_path, default_reports_dir, load_settings  # noqa: E402
from .report import parse_event_line, parse_progress_line  # noqa: E402

READ_CHUNK_BYTES = 4096

PASS_RGB = (0.25, 0.80, 0.52)
FAIL_RGB = (0.94, 0.38, 0.40)
WAIT_RGB = (0.36, 0.39, 0.47)
NOW_RGB = (0.98, 0.76, 0.30)

CATEGORY_NAMES = {
    "desktop_apps": "Desktop",
    "web_and_maps": "Web & maps",
    "media_controls": "Media",
    "system_status": "System",
    "files_and_productivity": "Files & notes",
    "software_help_and_knowledge": "Help & facts",
}

CSS = """
.vt-root { background: #14161c; color: #e9ecf3; }
.vt-card { background: #1d2029; border-radius: 16px; padding: 18px 20px; }
.vt-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 1.6px; opacity: 0.55; }
.vt-prompt { font-size: 23px; font-weight: 700; }
.vt-reply { font-size: 15px; opacity: 0.86; }
.vt-phase { font-size: 13px; font-weight: 600; color: #f9c24d; }
.vt-phase.pass { color: #40cc85; }
.vt-phase.fail { color: #f06166; }
.vt-stat-number { font-size: 26px; font-weight: 800; }
.vt-stat-number.pass { color: #40cc85; }
.vt-stat-number.fail { color: #f06166; }
.vt-stat-label { font-size: 11px; font-weight: 600; letter-spacing: 1.2px; opacity: 0.55; }
.vt-list { background: transparent; }
.vt-list row { background: #1d2029; border-radius: 12px; margin-bottom: 6px; padding: 10px 14px; }
.vt-badge { font-size: 15px; font-weight: 800; border-radius: 999px; min-width: 28px; min-height: 28px; }
.vt-badge.pass { background: alpha(#40cc85, 0.18); color: #40cc85; }
.vt-badge.fail { background: alpha(#f06166, 0.18); color: #f06166; }
.vt-row-prompt { font-size: 14px; font-weight: 600; }
.vt-row-reply { font-size: 12px; opacity: 0.62; }
.vt-row-time { font-size: 12px; opacity: 0.5; }
.vt-log { font-size: 11px; }
"""


@dataclass(slots=True)
class RunTally:
    """What the charts draw: every finished case in order, plus the totals."""

    total: int = 0
    outcomes: list[bool] = field(default_factory=list)
    by_category: dict[str, list[int]] = field(default_factory=dict)  # category -> [passed, failed]
    running: bool = False

    @property
    def passed(self) -> int:
        return sum(self.outcomes)

    @property
    def failed(self) -> int:
        return len(self.outcomes) - self.passed

    @property
    def remaining(self) -> int:
        return max(0, self.total - len(self.outcomes))

    def add(self, category: str, passed: bool) -> None:
        self.outcomes.append(passed)
        counts = self.by_category.setdefault(category or "general", [0, 0])
        counts[0 if passed else 1] += 1
        self.total = max(self.total, len(self.outcomes))

    def percent(self) -> int:
        done = len(self.outcomes)
        return round(100 * self.passed / done) if done else 0

    def slices(self) -> list[tuple[int, tuple[float, float, float]]]:
        """The donut's slices as (count, colour): passed, failed, still to go."""
        waiting = self.remaining if (self.outcomes or self.total) else 1
        return [(self.passed, PASS_RGB), (self.failed, FAIL_RGB), (waiting, WAIT_RGB)]


def category_label(category: str) -> str:
    return CATEGORY_NAMES.get(category, category.replace("_", " ").capitalize() or "General")


class VoxaTestWindow(Adw.ApplicationWindow):
    """The harness window: the prompt being tested, each result as it lands, and the score so far."""

    def __init__(self, application: Adw.Application, cases: str = "", autostart: bool = False) -> None:
        super().__init__(application=application)
        self.set_title("VoxaTest")
        self.set_default_size(940, 1000)
        self.set_size_request(620, 520)
        self._proc: Gio.Subprocess | None = None
        self._line_buffer = ""
        self._tally = RunTally()
        self._errors = 0
        self._install_css()
        self._build_ui()
        if cases:
            self.cases_entry.set_text(cases)
        if autostart:
            GLib.timeout_add(600, self._autostart)

    def _autostart(self) -> bool:
        self._on_start()
        return GLib.SOURCE_REMOVE

    def _install_css(self) -> None:
        provider = Gtk.CssProvider()
        provider.load_from_string(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        Adw.StyleManager.get_default().set_color_scheme(Adw.ColorScheme.FORCE_DARK)

    # ------------------------------------------------------------------ layout

    def _build_ui(self) -> None:
        self.toast_overlay = Adw.ToastOverlay()
        toolbar = Adw.ToolbarView()
        self.toast_overlay.set_child(toolbar)
        self.set_content(self.toast_overlay)

        menu = Gio.Menu()
        menu.append("Failure report", "win.failure-report")
        menu.append("Open reports folder", "win.reports")
        menu.append("About VoxaTest", "app.about")
        for name, callback in (("failure-report", self._open_failure_report), ("reports", self._open_reports_dir)):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", callback)
            self.add_action(action)

        header = Adw.HeaderBar()
        header.set_title_widget(Adw.WindowTitle.new("VoxaTest", "Speaking to Voxa and checking what she does"))
        self.start_button = Gtk.Button(label="Start")
        self.start_button.add_css_class("suggested-action")
        self.start_button.connect("clicked", self._on_start)
        header.pack_start(self.start_button)
        self.stop_button = Gtk.Button(label="Stop", sensitive=False)
        self.stop_button.connect("clicked", self._on_stop)
        header.pack_start(self.stop_button)
        header.pack_end(Gtk.MenuButton(menu_model=menu, icon_name="open-menu-symbolic"))
        toolbar.add_top_bar(header)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.add_css_class("vt-root")
        toolbar.set_content(root)
        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        for side in ("top", "bottom", "start", "end"):
            getattr(inner, f"set_margin_{side}")(16)
        inner.set_vexpand(True)
        root.append(inner)

        inner.append(self._build_score_card())
        inner.append(self._build_now_card())

        self.result_list = Gtk.ListBox()
        self.result_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self.result_list.add_css_class("vt-list")
        self.result_scroll = Gtk.ScrolledWindow(vexpand=True)
        self.result_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.result_scroll.set_child(self.result_list)
        inner.append(self.result_scroll)

        inner.append(self._build_setup_row())

    def _build_score_card(self) -> Gtk.Widget:
        card = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=22)
        card.add_css_class("vt-card")

        self.donut = Gtk.DrawingArea()
        self.donut.set_content_width(170)
        self.donut.set_content_height(170)
        self.donut.set_draw_func(self._draw_donut)
        card.append(self.donut)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, hexpand=True, valign=Gtk.Align.CENTER)
        stats = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=28)
        self.passed_label = self._stat(stats, "PASSED", "pass")
        self.failed_label = self._stat(stats, "FAILED", "fail")
        self.remaining_label = self._stat(stats, "TO GO", "")
        right.append(stats)

        self.strip = Gtk.DrawingArea()
        self.strip.set_content_height(26)
        self.strip.set_hexpand(True)
        self.strip.set_draw_func(self._draw_strip)
        right.append(self.strip)

        self.bars = Gtk.DrawingArea()
        self.bars.set_content_height(6 * 20)
        self.bars.set_hexpand(True)
        self.bars.set_draw_func(self._draw_bars)
        right.append(self.bars)
        card.append(right)
        return card

    def _stat(self, parent: Gtk.Box, caption: str, css: str) -> Gtk.Label:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        number = Gtk.Label(label="0", xalign=0)
        number.add_css_class("vt-stat-number")
        if css:
            number.add_css_class(css)
        label = Gtk.Label(label=caption, xalign=0)
        label.add_css_class("vt-stat-label")
        box.append(number)
        box.append(label)
        parent.append(box)
        return number

    def _build_now_card(self) -> Gtk.Widget:
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        card.add_css_class("vt-card")
        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        self.now_eyebrow = Gtk.Label(label="READY", xalign=0, hexpand=True)
        self.now_eyebrow.add_css_class("vt-eyebrow")
        self.spinner = Gtk.Spinner()
        self.phase_label = Gtk.Label(label="", xalign=1)
        self.phase_label.add_css_class("vt-phase")
        top.append(self.now_eyebrow)
        top.append(self.phase_label)
        top.append(self.spinner)
        self.prompt_label = Gtk.Label(label="Press Start to speak the first prompt to Voxa.", xalign=0, wrap=True)
        self.prompt_label.add_css_class("vt-prompt")
        self.reply_label = Gtk.Label(
            label="Voxa must be open and listening for her wake word.", xalign=0, wrap=True
        )
        self.reply_label.add_css_class("vt-reply")
        card.append(top)
        card.append(self.prompt_label)
        card.append(self.reply_label)
        return card

    def _build_setup_row(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.append(Gtk.Label(label="Cases", valign=Gtk.Align.CENTER, opacity=0.7))
        self.cases_entry = Gtk.Entry(hexpand=True, text=str(DEFAULT_DATA_PATH))
        self.cases_entry.set_placeholder_text("Path to a test cases file")
        row.append(self.cases_entry)
        browse = Gtk.Button(label="Browse…")
        browse.connect("clicked", self._on_browse)
        row.append(browse)
        row.append(Gtk.Label(label="Limit", valign=Gtk.Align.CENTER, opacity=0.7))
        self.limit_spin = Gtk.SpinButton(
            adjustment=Gtk.Adjustment(value=0, lower=0, upper=10_000, step_increment=1, page_increment=10),
            numeric=True,
            value=0,
        )
        self.limit_spin.set_tooltip_text("0 runs every case in the file")
        row.append(self.limit_spin)
        box.append(row)

        self.status_label = Gtk.Label(label="Idle", xalign=0, opacity=0.7)
        box.append(self.status_label)

        expander = Gtk.Expander(label="Run log")
        log_scroll = Gtk.ScrolledWindow()
        log_scroll.set_min_content_height(150)
        self.log_view = Gtk.TextView(
            editable=False, cursor_visible=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR
        )
        self.log_view.add_css_class("vt-log")
        log_scroll.set_child(self.log_view)
        expander.set_child(log_scroll)
        box.append(expander)
        return box

    # ------------------------------------------------------------------ charts

    def _draw_donut(self, _area, cr, width: int, height: int) -> None:
        tally = self._tally
        radius = min(width, height) / 2 - 10
        cx, cy = width / 2, height / 2
        cr.set_line_width(18)
        slices = tally.slices()
        whole = max(1, sum(count for count, _rgb in slices))
        start = -math.pi / 2
        for count, rgb in slices:
            if count <= 0:
                continue
            sweep = 2 * math.pi * count / whole
            cr.set_source_rgb(*rgb)
            cr.arc(cx, cy, radius, start, start + sweep)
            cr.stroke()
            start += sweep
        cr.set_source_rgb(0.92, 0.93, 0.96)
        cr.select_font_face("Sans", 0, 1)
        big = f"{tally.percent()}%" if tally.outcomes else "–"
        cr.set_font_size(34)
        extents = cr.text_extents(big)
        cr.move_to(cx - extents.width / 2 - extents.x_bearing, cy + 6)
        cr.show_text(big)
        cr.select_font_face("Sans", 0, 0)
        cr.set_font_size(12)
        cr.set_source_rgba(0.92, 0.93, 0.96, 0.6)
        small = f"{tally.passed} of {len(tally.outcomes)} passed" if tally.outcomes else "no results yet"
        extents = cr.text_extents(small)
        cr.move_to(cx - extents.width / 2 - extents.x_bearing, cy + 28)
        cr.show_text(small)

    def _draw_strip(self, _area, cr, width: int, height: int) -> None:
        tally = self._tally
        total = max(tally.total, len(tally.outcomes), 1)
        gap = 4
        cell = max(4.0, min(22.0, (width - gap * (total - 1)) / total))
        for index in range(total):
            if index < len(tally.outcomes):
                rgb = PASS_RGB if tally.outcomes[index] else FAIL_RGB
            elif index == len(tally.outcomes) and tally.running:
                rgb = NOW_RGB
            else:
                rgb = WAIT_RGB
            cr.set_source_rgb(*rgb)
            x = index * (cell + gap)
            self._rounded(cr, x, (height - cell) / 2, cell, cell, min(5.0, cell / 3))
            cr.fill()

    def _draw_bars(self, _area, cr, width: int, _height: int) -> None:
        names = list(self._tally.by_category.items())[:6]
        label_width = 110
        cr.select_font_face("Sans", 0, 0)
        cr.set_font_size(11)
        for row, (category, (passed, failed)) in enumerate(names):
            y = row * 20
            cr.set_source_rgba(0.92, 0.93, 0.96, 0.7)
            cr.move_to(0, y + 12)
            cr.show_text(category_label(category))
            done = passed + failed
            bar = max(10, width - label_width - 44)
            x = label_width
            for count, rgb in ((passed, PASS_RGB), (failed, FAIL_RGB)):
                if count:
                    span = bar * count / done
                    cr.set_source_rgb(*rgb)
                    self._rounded(cr, x, y + 4, max(span - 2, 2), 9, 4)
                    cr.fill()
                    x += span
            cr.set_source_rgba(0.92, 0.93, 0.96, 0.7)
            cr.move_to(label_width + bar + 8, y + 12)
            cr.show_text(f"{passed}/{done}")

    @staticmethod
    def _rounded(cr, x: float, y: float, w: float, h: float, r: float) -> None:
        cr.new_sub_path()
        cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()

    def _redraw(self) -> None:
        tally = self._tally
        self.passed_label.set_text(str(tally.passed))
        self.failed_label.set_text(str(tally.failed))
        self.remaining_label.set_text(str(tally.remaining))
        for area in (self.donut, self.strip, self.bars):
            area.queue_draw()

    # ------------------------------------------------------------------ events

    def _note_event(self, event: dict) -> None:
        kind = event.get("event")
        if kind == "case_start":
            self._tally.total = int(event.get("total", 0) or 0)
            category = category_label(str(event.get("category", ""))).upper()
            self.now_eyebrow.set_text(f"TEST {event.get('index', '?')} OF {event.get('total', '?')}  ·  {category}")
            self.prompt_label.set_text(f"“{event.get('prompt', '')}”")
            self.reply_label.set_text("Voxa is being asked…")
            self._set_phase("Speaking and listening", "")
            self._redraw()
        elif kind == "case_result":
            passed = event.get("status") == "PASS"
            if event.get("status") == "ERROR":
                self._errors += 1
            self._tally.add(str(event.get("category", "")), passed)
            reply = str(event.get("reply") or "").strip()
            notes = str(event.get("notes") or "").strip()
            if reply:
                shown = f"Voxa: {reply}"
            else:
                shown = "Voxa did it without speaking." if passed else "Voxa said nothing."
            self.reply_label.set_text(shown if passed else f"{shown}\n{notes}")
            self._set_phase("Passed" if passed else "Failed", "pass" if passed else "fail")
            self._add_row(str(event.get("prompt", "")), shown, notes, passed, event.get("seconds"))
            self._redraw()

    def _set_phase(self, text: str, css: str) -> None:
        self.phase_label.set_text(text)
        for name in ("pass", "fail"):
            self.phase_label.remove_css_class(name)
        if css:
            self.phase_label.add_css_class(css)

    def _add_row(self, prompt: str, reply: str, notes: str, passed: bool, seconds) -> None:
        line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        badge = Gtk.Label(label="✓" if passed else "✕", valign=Gtk.Align.CENTER)
        badge.add_css_class("vt-badge")
        badge.add_css_class("pass" if passed else "fail")
        text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        title = Gtk.Label(label=prompt, xalign=0)
        title.add_css_class("vt-row-prompt")
        title.set_ellipsize(Pango.EllipsizeMode.END)
        detail = Gtk.Label(label=reply if passed else f"{reply}  —  {notes}", xalign=0)
        detail.add_css_class("vt-row-reply")
        detail.set_ellipsize(Pango.EllipsizeMode.END)
        detail.set_tooltip_text(f"{reply}\n{notes}")
        text.append(title)
        text.append(detail)
        took = Gtk.Label(label=f"{seconds}s" if seconds is not None else "", valign=Gtk.Align.CENTER)
        took.add_css_class("vt-row-time")
        line.append(badge)
        line.append(text)
        line.append(took)
        self.result_list.append(line)
        GLib.idle_add(self._scroll_results_to_end)

    def _scroll_results_to_end(self) -> bool:
        adjustment = self.result_scroll.get_vadjustment()
        adjustment.set_value(adjustment.get_upper())
        return GLib.SOURCE_REMOVE

    # ------------------------------------------------------------------ running

    def _on_browse(self, *_args) -> None:
        dialog = Gtk.FileChooserNative.new("Choose test cases", self, Gtk.FileChooserAction.OPEN, "_Open", "_Cancel")
        current = self.cases_entry.get_text().strip()
        if current:
            dialog.set_file(Gio.File.new_for_path(current))

        def on_response(dlg: Gtk.FileChooserNative, response: int) -> None:
            if response == Gtk.ResponseType.ACCEPT:
                file = dlg.get_file()
                if file is not None and file.get_path():
                    self.cases_entry.set_text(file.get_path())
            dlg.destroy()

        dialog.connect("response", on_response)
        dialog.show()

    def _on_start(self, *_args) -> None:
        if self._proc is not None:
            return

        settings = load_settings()
        if settings.voxa_echo_mode and settings.voxa_echo_mode != "off":
            message = (
                "Turn off Voxa echo cancellation in Preferences › Speech before running acoustic tests; "
                "otherwise Voxa removes VoxaTest's spoken prompts."
            )
            self.status_label.set_text(message)
            self._append_log(f"[preflight] {message}\n")
            self._toast(message)
            return

        cases = Path(self.cases_entry.get_text().strip() or DEFAULT_DATA_PATH).expanduser()
        if not cases.is_file():
            self._toast(f"Case file not found: {cases}")
            return

        cmd = [sys.executable, "-u", "-m", "voxatest", "run", "--cases", str(cases)]
        limit = int(self.limit_spin.get_value())
        if limit > 0:
            cmd.extend(["--limit", str(limit)])

        self._append_log(f"$ {' '.join(cmd)}\n")
        self._reset_progress()
        self._set_running(True)
        try:
            self._proc = Gio.Subprocess.new(
                cmd,
                Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_MERGE,
            )
        except GLib.Error as exc:
            self._proc = None
            self._set_running(False)
            self._append_log(f"Failed to start run: {exc}\n")
            return

        stdout = self._proc.get_stdout_pipe()
        if stdout is not None:
            self._read_stdout(stdout)
        self._proc.wait_check_async(None, self._on_process_exited)

    def _on_stop(self, *_args) -> None:
        if self._proc is None:
            return
        self.stop_button.set_sensitive(False)
        self.status_label.set_text("Stopping — writing partial report…")
        self._append_log("\n[stop] terminating run…\n")
        proc = self._proc
        try:
            proc.send_signal(signal.SIGTERM)
        except GLib.Error:
            proc.force_exit()
        GLib.timeout_add_seconds(20, self._force_stop, proc)

    def _force_stop(self, proc: Gio.Subprocess) -> bool:
        if proc is self._proc:
            proc.force_exit()
        return GLib.SOURCE_REMOVE

    def _read_stdout(self, stream: Gio.InputStream) -> None:
        stream.read_bytes_async(READ_CHUNK_BYTES, GLib.PRIORITY_DEFAULT, None, self._on_stdout_chunk)

    def _on_stdout_chunk(self, stream: Gio.InputStream, result: Gio.AsyncResult) -> None:
        try:
            chunk = stream.read_bytes_finish(result)
        except GLib.Error:
            return
        size = chunk.get_size() if chunk is not None else 0
        if size <= 0:
            return
        self._line_buffer += chunk.get_data().decode("utf-8", errors="replace")
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            event = parse_event_line(line)
            if event is not None:
                self._note_event(event)
                continue
            self._append_log(line + "\n")
            self._note_progress_line(line)
        self._read_stdout(stream)

    def _on_process_exited(self, proc: Gio.Subprocess, result: Gio.AsyncResult, *_args) -> None:
        tally = self._tally
        try:
            proc.wait_check_finish(result)
            detail = "Run finished: every test passed."
        except GLib.Error:
            if tally.outcomes:
                detail = f"Run finished: {tally.passed} passed, {tally.failed} failed."
            else:
                detail = "The run could not start — is Voxa open and is the local model running?"
        if proc is self._proc:
            self._proc = None
        self._set_running(False)
        self.status_label.set_text(detail)
        if tally.outcomes:
            self.now_eyebrow.set_text("FINISHED")
            self.prompt_label.set_text(f"{tally.passed} of {len(tally.outcomes)} tests passed")
            self.reply_label.set_text(detail)
            self._set_phase("", "")
        self._redraw()
        self._append_log(f"\n[{detail}]\n")

    def _set_running(self, running: bool) -> None:
        self._tally.running = running
        self.start_button.set_sensitive(not running)
        self.stop_button.set_sensitive(running)
        self.cases_entry.set_sensitive(not running)
        self.limit_spin.set_sensitive(not running)
        if running:
            self.status_label.set_text("Running…")
            self.now_eyebrow.set_text("STARTING")
            self.prompt_label.set_text("Loading the speech model…")
            self.reply_label.set_text("The first prompt is spoken in a few seconds.")
            self.spinner.start()
        else:
            self.spinner.stop()

    def _reset_progress(self) -> None:
        self._line_buffer = ""
        self._tally = RunTally()
        self._errors = 0
        child = self.result_list.get_first_child()
        while child is not None:
            following = child.get_next_sibling()
            self.result_list.remove(child)
            child = following
        self._redraw()

    def _note_progress_line(self, line: str) -> None:
        parsed = parse_progress_line(line)
        if parsed is None:
            return
        done, total, _status = parsed
        self.status_label.set_text(f"Running… {done} of {total} done")

    def _append_log(self, text: str) -> None:
        buffer = self.log_view.get_buffer()
        buffer.insert(buffer.get_end_iter(), text)
        end = buffer.get_end_iter()
        mark = buffer.create_mark(None, end, False)
        self.log_view.scroll_to_mark(mark, 0.0, False, 0.0, 0.0)
        buffer.delete_mark(mark)

    def _open_reports_dir(self, *_args) -> None:
        path = default_reports_dir()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._toast(f"Could not create {path}: {exc}")
            return

        def on_launch_finish(launcher: Gtk.FileLauncher, result: Gio.AsyncResult) -> None:
            try:
                launcher.launch_finish(result)
            except GLib.Error as exc:
                self._toast(f"Could not open reports folder: {exc.message}")

        launcher = Gtk.FileLauncher.new(Gio.File.new_for_path(str(path)))
        launcher.launch(self, None, on_launch_finish)

    def _open_failure_report(self, *_args) -> None:
        path = default_failure_report_path()
        if not path.exists():
            self._toast(f"No failure report yet at {path}")
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._toast(f"Could not read {path}: {exc}")
            return

        dialog = Adw.Dialog(title="Failure report", content_width=680, content_height=560)
        toolbar = Adw.ToolbarView()
        dialog.set_child(toolbar)
        toolbar.add_top_bar(Adw.HeaderBar(show_end_title_buttons=False))
        scroll = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
        view = Gtk.TextView(editable=False, cursor_visible=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.get_buffer().set_text(text)
        view.set_left_margin(8)
        view.set_right_margin(8)
        scroll.set_child(view)
        toolbar.set_content(scroll)
        dialog.present(self)

    def _toast(self, message: str) -> None:
        self.toast_overlay.add_toast(Adw.Toast(title=message))
