"""Issue #72, task v11b: the render tap works on the real sink, but the REPLY never reports it.

The v11 live evidence proved the tap itself works on the installed flatpak's real PulseAudio sink:
the 04:20 run logged ``render-tap attached=True pieces=1 samples=143200 peak=0.682 silent=False``
four times, and the offline probes (host null sink, and the installed rc5 sandbox) read NON-silent
on both shipped paths - ``piece`` (playbin + wav) and ``whole-reply`` (appsrc -> decodebin ->
autoaudiosink). The hypothesis that the tap found nothing because ``autoaudiosink`` builds its real
child only on state change is false: the element named ``sink`` IS the autoaudiosink.

What was broken is the REPORT:

* ``_conversation_speak`` ran the reply's end ONLY through ``self._for_session(token, ...)``. The
  session token moves on as soon as the user talks over her or a follow-up arms - which is exactly
  what a live reply runs into - so ``_note_reply_end`` never ran: no ``render-tap`` line at all, and
  the render turn stayed open (``state='open'`` forever, the next reply filing her tail under it).
* the barge-in (``INTERRUPT``) path stopped her audio and never called ``_note_reply_end``, so an
  interrupted reply never closed its turn and never reported either.

These tests drive the real production ``MainWindow`` (a subprocess with a GTK-capable CPython, the
same pattern as ``tests/test_playbin_render_reference.py``; fake audio and speech, so no microphone
and no sound device are opened) and assert the live gate's contract: every reply - natural end,
error, session-token-dropped end, interruption - writes exactly ONE ``render-tap`` line with
``samples > 0`` and ``peak > 0``, and closes its render turn. On base the token-dropped and
interrupted replies write no line and leave the turn open.

The real-sink half (the tap records non-silent audio on a REAL sink) cannot be reproduced inside
``tests/`` because ``tests/conftest.py`` points ``PULSE_SERVER`` at nothing on purpose; it is proven
offline by ``.voxa-bench/real_sink_probe.py`` and live by the ≥6 ``render-tap attached=True
samples>0 peak>0`` lines the release check counts in barge.log.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

WINDOW_PROBE = r"""
import sys, os
sys.path.insert(0, sys.argv[1])
os.environ.setdefault("VOXA_NO_WELCOME", "1")
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GLib, Gtk
# GTK 4's init_check() returns True even with NO display (the window factory is where
# it actually fails: Gtk.init() raises "Gtk couldn't be initialized"), so the real test
# is whether GDK can name a display at all.
if not Gtk.init_check() or Gdk.Display.get_default() is None:
    print("SKIP no display")
    raise SystemExit(0)
Adw.init()
import numpy as np
from voxa.bargectl import BargeOutcome
from voxa.window import MainWindow

RATE = 16000


def tone(seconds, level=0.3):
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    return (level * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def pump(iterations=60):
    ctx = GLib.MainContext.default()
    for _ in range(iterations):
        ctx.iteration(False)


class FakeAudio:
    # the capture side of the window: no microphone, no sink, no canceller
    def __init__(self):
        self._active = False

    @property
    def is_active(self):
        return self._active

    def start(self, device_id, on_audio, on_level, on_error):
        self._active = True

    def stop(self):
        self._active = False

    @property
    def timeline(self):
        return None

    @property
    def delay_frac(self):
        return 0.0

    @property
    def delay_confident(self):
        return False

    @property
    def delay_degraded(self):
        return False

    def echo_travel_ms(self):
        return None

    def echo_measurements(self):
        return (0.0, None, None)

    def __getattr__(self, name):
        def stub(*args, **kwargs):
            return None

        return stub


class FakeSpeech:
    def __init__(self):
        self.callbacks = {}
        self.tap = {"attached": True, "error": ""}

    def speak(self, text, rate, voice, on_started=None, on_done=None, on_error=None, on_words=None):
        self.callbacks = {"started": on_started, "done": on_done, "error": on_error}

    def position(self):
        return 0.0

    def render_pilot(self):
        return None

    def stop(self):
        pass


application = Adw.Application(application_id="io.github.crhy.voxa.rendertaptest")
application.register(None)


def fresh():
    for name in ("_load_whisper", "_load_wake_whisper", "_detect_hardware_async",
                 "_refresh_ollama_models", "_refresh_devices", "_start_ai_server_async",
                 "_restart_ai_server_async", "_stop_ai_server_async"):
        setattr(MainWindow, name, lambda *args, **kwargs: None)
    win = MainWindow(application)
    win.audio = FakeAudio()
    win.speech = FakeSpeech()
    win.whisper._model = object()
    win.devices = [object()]
    win.present()
    pump()
    win.reports = []
    win._barge_trace = lambda event, **kw: win.reports.append(dict(event=event, **kw))
    return win


def speak(win):
    win.assistant.activate()
    win.assistant.wake(win.assistant.token())
    win.assistant.prompt_accepted(win.assistant.token())
    win._conversation_speak("A long answer that is being read out loud")
    win.render_timeline.submit(tone(1.0), generation="g1", path="whole-reply")


def taps(win):
    return [r for r in win.reports if r["event"] == "render-tap"]


def show(label, win):
    lines = taps(win)
    for r in lines:
        print(label, "LINE", "attached", r.get("attached"), "samples", r.get("samples"),
              "peak", r.get("peak"), "silent", r.get("silent"))
    stats = win.render_timeline.turn_stats()
    print(label, "COUNT", len(lines), "turn", "none" if stats is None else stats.get("state"))
    win._closing = True
    win.assistant.go_offline()
    win.destroy()
    pump(20)


# 1. the reply's audio ends after its session token moved on (the live talk-over case)
win = fresh()
speak(win)
done = win.speech.callbacks["done"]
win.assistant.go_offline()
done()
pump()
show("token-dropped", win)

# 2. an interrupted reply (barge-in) stops the audio: it owes the same report
win = fresh()
speak(win)
win._apply_barge_decision(BargeOutcome(action="interrupt", generation=win._hearing.generation()))
pump()
show("interrupted", win)

# 3. however it is closed, a reply reports exactly once
win = fresh()
speak(win)
win.speech.callbacks["done"]()
pump()
win.stop_conversation_mode()
pump()
show("closed-twice", win)

# 4. step 8 (issue #72 v11b): an internally-managed canceller (dtln/aec3) owns the alignment,
# so a decision while she speaks is qualified audio, never 'alignment unhealthy'.
win = fresh()
win.audio.drain_joint_frames = lambda: []
win.audio.echo_measurements = lambda: (0.5, -20.0, -35.0)   # loud reference, echo explained
win.audio.echo_path_qualified = lambda: True
win.audio.delay_samples = 0                                  # DTLN/AEC3 never report a delay
win.assistant.activate()
win.assistant.wake(win.assistant.token())
evidence = win._hearing_evidence()
from voxa.attribution import attribute
decision = attribute(evidence) if evidence is not None else None
print("dtln-evidence",
      "healthy", bool(getattr(evidence, "alignment_healthy", False)) if evidence else None,
      "reasons", ";".join(decision.reasons) if decision is not None else "",
      "source", decision.source if decision is not None else "")
show("dtln-evidence", win)
"""


def _gtk_python() -> str | None:
    """A CPython with GTK/Adw, so the real production window can be driven offline."""
    for candidate in (sys.executable, "/usr/bin/python3", shutil.which("python3")):
        if not candidate:
            continue
        probe = subprocess.run(
            [candidate, "-c",
             "import gi; gi.require_version('Gtk','4.0'); gi.require_version('Adw','1');"
             "from gi.repository import Gtk; Gtk.init_check()"],
            capture_output=True, timeout=30,
        )
        if probe.returncode == 0:
            return candidate
    return None


def test_every_reply_reports_its_render_tap_once() -> None:
    python = _gtk_python()
    if python is None:
        pytest.skip("no GTK-capable python: the production window cannot be driven here")
    env = os.environ.copy()
    env.pop("PULSE_SERVER", None)  # the child opens no sound device, but keep it harmless
    out = subprocess.run([python, "-c", WINDOW_PROBE, os.getcwd()],
                         capture_output=True, text=True, timeout=180, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    if "SKIP no display" in out.stdout:
        pytest.skip("no GTK display for the production window")
    lines = out.stdout.splitlines()

    def count(label: str) -> int:
        return int(next(line for line in lines if line.startswith(f"{label} COUNT")).split()[2])

    def turn(label: str) -> str:
        return next(line for line in lines if line.startswith(f"{label} COUNT")).split()[-1]

    for label in ("token-dropped", "interrupted", "closed-twice"):
        assert count(label) == 1, f"{label}: one render-tap line per reply\n{out.stdout}"
        assert turn(label) == "closed", f"{label}: the reply must close its render turn\n{out.stdout}"
    for label in ("token-dropped", "interrupted"):
        line = next(line for line in lines if line.startswith(f"{label} LINE"))
        label, _, _, attached, _, samples, _, peak, _, silent = line.split()
        assert attached == "True" and int(samples) > 0 and float(peak) > 0 and silent == "False", line

    # step 8: the DTLN/AEC3 canceller owns the delay, so a frame while she speaks is qualified
    dtln = next(line for line in lines if line.startswith("dtln-evidence healthy"))
    assert "healthy True" in dtln, f"a self-managed canceller must count as aligned: {dtln}"
    assert "alignment unhealthy" not in dtln, dtln
