#!/usr/bin/env python3
"""Replay one text request through Voxa's REAL request pipeline, headless.

Task vf01 (issue #72): every heard request ended with empty speech, so the
model answer must be reproducible without a microphone. This tool builds the
production ``MainWindow`` exactly like the window tests do (throwaway config
dirs, heavy startup work stubbed) but keeps the request pipeline itself real:
``ask_ai`` -> routing -> the query worker -> the real ``OllamaClient`` against
the real Ollama server -> ``_on_query_finished`` -> speech.

It never touches the microphone, never plays sound and never opens a window on
the owner's display: it re-runs itself inside ``xvfb-run`` when needed.  Audio
capture, Whisper and TTS are recorders; everything between the text and the
spoken reply is Voxa's own code, config and Ollama.

Usage:
    .venv/bin/python3 tools/replay_request.py \
        [--config PATH] [--timeout SECONDS] "What is the capital of France?"

Prints the spoken reply, the final status line, and the actions.jsonl-style
records the pipeline wrote during the run.  Exit code 0 only when a non-empty
reply was spoken.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Environment must be set before voxa.window is imported: GTK, PulseAudio and
# the welcome flow all read it at import/startup time (same tricks as tests).
os.environ.setdefault("PULSE_SERVER", "unix:/nonexistent-voxa-replay")
os.environ["VOXA_NO_WELCOME"] = "1"
os.environ["GSK_RENDERER"] = "cairo"


def _own_virtual_display() -> None:
    """Run the whole replay under Xvfb so no window ever lands on the owner's desktop."""
    if os.environ.get("VOXA_REPLAY_XVFB") == "1" or shutil.which("xvfb-run") is None:
        return
    os.environ["VOXA_REPLAY_XVFB"] = "1"
    cmd = ["xvfb-run", "-a", sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    sys.exit(subprocess.call(cmd, env=os.environ))


def _import_gi():
    """Import PyGObject, reaching the distro copy when a plain venv lacks it."""
    try:
        import gi
    except ImportError:
        import site

        for extra in ("/usr/lib/python3/dist-packages", "/usr/lib/python3.13/dist-packages"):
            if (Path(extra) / "gi").is_dir():
                site.addsitedir(extra)
                break
        import gi
    return gi


def _find_config(explicit: str | None) -> Path:
    """The real config: the Flatpak app's if present, else the user's."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    home = Path.home()
    candidates.append(home / ".var/app/io.github.crhy.voxa/config/voxa/config.json")
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    candidates.append(config_home / "voxa/config.json")
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit("no Voxa config.json found (tried --config, the Flatpak app, and XDG_CONFIG_HOME)")


class FakeSpeech:
    """Records what the pipeline would have spoken; no network, no sound."""

    def __init__(self, window) -> None:
        self.window = window
        self.spoken: list[str] = []
        self.done_count = 0

    def stop(self) -> None:
        pass

    def speak(self, text, rate, voice, *, on_started, on_done, on_error, on_words=None, before_play=None) -> None:
        self.spoken.append(text)
        if on_started is not None:
            on_started()
        if on_done is not None:
            on_done()

    def prepare(self, text, rate, voice, on_ready=None):
        class _Item:
            ready = type("_R", (), {"wait": lambda self, timeout=None: True})()
            error = ""
            path = ""
            words = ()
            cancelled = type("_C", (), {"is_set": lambda self: False})()

        item = _Item()
        self.spoken.append(text)
        return item

    def speak_prepared(self, item, *, on_started, on_done, on_error, on_words=None, before_play=None) -> None:
        if on_started is not None:
            on_started()
        if on_done is not None:
            on_done()


class FakeAudio:
    """A microphone that never captures: the heard text is handed to the window by hand."""

    def __init__(self) -> None:
        self.is_active = False

    def start(self, device_id, on_audio, on_level, on_error) -> None:
        self.is_active = True

    def stop(self) -> None:
        self.is_active = False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request", nargs="+", help="the request text, as if heard")
    parser.add_argument("--config", help="config.json to use (default: the installed one)")
    parser.add_argument("--typed", action="store_true", help="ask as if typed (no conversation mode)")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for the reply")
    args = parser.parse_args(argv)
    request = " ".join(args.request)
    _own_virtual_display()

    gi = _import_gi()
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw, GLib, Gtk

    if not Gtk.init_check():
        print("replay_request: no display (run under xvfb-run)", file=sys.stderr)
        return 2
    Adw.init()

    real_config = _find_config(args.config)
    print(f"replay_request: config {real_config}")

    # If the request has to restart a stopped backend, the restarted ``ollama
    # serve`` must still find the models: the sandboxed HOME below would hide
    # the real ``~/.ollama``.
    real_ollama_models = Path.home() / ".ollama" / "models"
    if real_ollama_models.is_dir():
        os.environ.setdefault("OLLAMA_MODELS", str(real_ollama_models))

    tmp = Path(tempfile.mkdtemp(prefix="voxa-replay-"))
    (tmp / "config/voxa").mkdir(parents=True)
    (tmp / "config/voxa/config.json").write_bytes(real_config.read_bytes())
    os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
    os.environ["XDG_STATE_HOME"] = str(tmp / "state")
    os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")
    os.environ["HOME"] = str(tmp)
    action_log = tmp / "state/voxa/actions.jsonl"
    os.environ["VOXA_ACTION_LOG"] = str(action_log)

    from voxa import window as window_module  # noqa: E402  (after env)
    from voxa.window import MainWindow  # noqa: E402

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
        setattr(MainWindow, name, lambda *a, **k: None)

    app = Adw.Application(application_id="io.github.crhy.voxa.replay")
    app.register(None)
    window = MainWindow(app)
    window.present()

    speech = FakeSpeech(window)
    window.speech = speech
    notices: list[str] = []
    window._toast = lambda text: notices.append(str(text))

    loop = GLib.MainLoop()
    started = time.monotonic()

    def finished() -> None:
        loop.quit()

    def tick() -> bool:
        if speech.spoken or time.monotonic() - started > args.timeout:
            GLib.idle_add(finished)
            return False
        return True

    GLib.timeout_add(250, tick)

    if args.typed:
        window.ask_ai(request)
    else:
        # The production hands-free entry point: ACTIVE, then the conversation
        # callback that Whisper would fire with the heard text.
        from voxa.audio import AudioDevice

        window.audio = FakeAudio()
        window.whisper._model = object()  # WhisperService.ready
        window.devices = [AudioDevice(identifier="mic", name="Microphone", device=None)]
        if not window.assistant.activate():
            print(f"replay_request: could not go ACTIVE: {getattr(window, '_start_failure', '')}",
                  file=sys.stderr)
            window._closing = True
            window.destroy()
            return 2
        window.conversation.last_heard = request
        window._on_conversation_prompt(request)
    loop.run()

    spoken_text = "\n".join(speech.spoken)
    print(f"replay_request: status line: {window.status_label.get_text()}")
    for notice in notices:
        print(f"replay_request: notice: {notice}")
    print(f"replay_request: response view: {window._get_text(window.response_view)!r}")
    print(f"replay_request: spoken reply: {spoken_text!r}")
    print("replay_request: action log:")
    if action_log.is_file():
        for line in action_log.read_text().splitlines():
            print(f"  {line}")
    else:
        print("  (none written)")

    window._closing = True
    window.destroy()
    for _ in range(20):
        GLib.MainContext.default().iteration(False)

    return 0 if spoken_text.strip() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
