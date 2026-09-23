from __future__ import annotations

import signal
import sys
from pathlib import Path

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

from .config import DEFAULT_DATA_PATH, default_failure_report_path, default_reports_dir  # noqa: E402
from .report import parse_progress_line  # noqa: E402

READ_CHUNK_BYTES = 4096


class VoxaTestWindow(Adw.ApplicationWindow):
    """Responsive harness window: pick cases, start/stop a run, watch the log."""

    def __init__(self, application: Adw.Application) -> None:
        super().__init__(application=application)
        self.set_title("VoxaTest")
        self.set_default_size(920, 640)
        self.set_size_request(640, 420)
        self._proc: Gio.Subprocess | None = None
        self._line_buffer = ""
        self._done = self._total = self._passed = self._failed = self._errors = 0
        self._build_ui()

    def _build_ui(self) -> None:
        self.toast_overlay = Adw.ToastOverlay()
        toolbar = Adw.ToolbarView()
        self.toast_overlay.set_child(toolbar)
        self.set_content(self.toast_overlay)

        menu = Gio.Menu()
        menu.append("About VoxaTest", "app.about")
        header = Adw.HeaderBar()
        title = Adw.WindowTitle.new("VoxaTest", "Speak-test Voxa end to end")
        header.set_title_widget(title)
        menu_button = Gtk.MenuButton(menu_model=menu, icon_name="open-menu-symbolic")
        header.pack_end(menu_button)
        toolbar.add_top_bar(header)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        toolbar.set_content(outer)

        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        controls.set_margin_top(12)
        controls.set_margin_bottom(6)
        controls.set_margin_start(12)
        controls.set_margin_end(12)
        outer.append(controls)

        hint = Gtk.Label(
            label=(
                "Start Voxa first and switch on Conversation mode. "
                "VoxaTest speaks each prompt with the wake word and listens for Voxa's reply."
            ),
            xalign=0,
            wrap=True,
            opacity=0.7,
        )
        controls.append(hint)

        file_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        file_row.append(Gtk.Label(label="Cases file", xalign=0, valign=Gtk.Align.CENTER))
        self.cases_entry = Gtk.Entry(hexpand=True, text=str(DEFAULT_DATA_PATH))
        self.cases_entry.set_placeholder_text("Path to test_cases.json")
        file_row.append(self.cases_entry)
        browse = Gtk.Button(label="Browse…", valign=Gtk.Align.CENTER)
        browse.connect("clicked", self._on_browse)
        file_row.append(browse)
        controls.append(file_row)

        options = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        options.append(Gtk.Label(label="Case limit", valign=Gtk.Align.CENTER))
        self.limit_spin = Gtk.SpinButton(
            adjustment=Gtk.Adjustment(value=0, lower=0, upper=10_000, step_increment=1, page_increment=10),
            numeric=True,
            value=0,
        )
        self.limit_spin.set_tooltip_text("0 runs every case in the file")
        options.append(self.limit_spin)
        options.append(Gtk.Label(label="(0 = all)", valign=Gtk.Align.CENTER, opacity=0.7))

        self.start_button = Gtk.Button(label="Start", valign=Gtk.Align.CENTER)
        self.start_button.add_css_class("suggested-action")
        self.start_button.connect("clicked", self._on_start)
        options.append(self.start_button)

        self.stop_button = Gtk.Button(label="Stop", valign=Gtk.Align.CENTER, sensitive=False)
        self.stop_button.add_css_class("destructive-action")
        self.stop_button.connect("clicked", self._on_stop)
        options.append(self.stop_button)

        self.report_button = Gtk.Button(label="Failure report", valign=Gtk.Align.CENTER)
        self.report_button.connect("clicked", self._open_failure_report)
        options.append(self.report_button)
        self.reports_button = Gtk.Button(label="Open reports", valign=Gtk.Align.CENTER)
        self.reports_button.connect("clicked", self._open_reports_dir)
        options.append(self.reports_button)
        controls.append(options)

        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.spinner = Gtk.Spinner()
        status_row.append(self.spinner)
        self.status_label = Gtk.Label(label="Idle", xalign=0, hexpand=True)
        status_row.append(self.status_label)
        controls.append(status_row)

        self.progress_bar = Gtk.ProgressBar(show_text=True)
        self.progress_bar.set_valign(Gtk.Align.CENTER)
        controls.append(self.progress_bar)

        log_scroll = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
        log_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        log_scroll.set_margin_start(12)
        log_scroll.set_margin_end(12)
        log_scroll.set_margin_bottom(12)
        self.log_view = Gtk.TextView(editable=False, cursor_visible=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.log_view.set_left_margin(8)
        self.log_view.set_right_margin(8)
        self.log_view.set_top_margin(6)
        self.log_view.set_bottom_margin(6)
        log_scroll.set_child(self.log_view)
        outer.append(log_scroll)

    def _on_browse(self, *_args) -> None:
        dialog = Gtk.FileChooserNative.new(
            "Choose test cases",
            self,
            Gtk.FileChooserAction.OPEN,
            "_Open",
            "_Cancel",
        )
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

        cases = Path(self.cases_entry.get_text().strip() or DEFAULT_DATA_PATH).expanduser()
        if not cases.is_file():
            self._toast(f"Case file not found: {cases}")
            return

        cmd = [sys.executable, "-u", "-m", "voxatest", "run", "--cases", str(cases)]
        limit = int(self.limit_spin.get_value())
        if limit > 0:
            cmd.extend(["--limit", str(limit)])

        self._append_log(f"$ {' '.join(cmd)}\n")
        self._set_running(True)
        self._reset_progress()
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

        stdout = self._proc.get_stdout()
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
        text = chunk.get_data().decode("utf-8", errors="replace")
        self._append_log(text)
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._note_progress_line(line)
        self._read_stdout(stream)

    def _on_process_exited(self, proc: Gio.Subprocess, result: Gio.AsyncResult, *_args) -> None:
        try:
            ok = proc.wait_check_finish(result)
            detail = "Run finished." if ok else "Run finished with failures (see the failure report)."
        except GLib.Error as exc:
            detail = f"Run stopped: {exc.message}"
        if proc is self._proc:
            self._proc = None
        self._set_running(False)
        self.status_label.set_text(detail)
        self._append_log(f"\n[{detail}]\n")

    def _set_running(self, running: bool) -> None:
        self.start_button.set_sensitive(not running)
        self.stop_button.set_sensitive(running)
        self.cases_entry.set_sensitive(not running)
        self.limit_spin.set_sensitive(not running)
        if running:
            self.status_label.set_text("Running…")
            self.spinner.start()
        else:
            self.spinner.stop()

    def _reset_progress(self) -> None:
        self._line_buffer = ""
        self._done = self._total = self._passed = self._failed = self._errors = 0
        self.progress_bar.set_fraction(0.0)
        self.progress_bar.set_text("0/0")

    def _note_progress_line(self, line: str) -> None:
        parsed = parse_progress_line(line)
        if parsed is None:
            return
        done, total, status = parsed
        self._done = done
        self._total = total
        if status == "PASS":
            self._passed += 1
        elif status == "FAIL":
            self._failed += 1
        else:
            self._errors += 1
        fraction = done / total if total > 0 else 0.0
        self.progress_bar.set_fraction(min(fraction, 1.0))
        self.progress_bar.set_text(
            f"{done}/{total} · {self._passed} passed · {self._failed} failed · {self._errors} errors"
        )

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
