"""Voxa's own way to operate another program's window: what is on the screen, not the accessibility bus.

A screenshot is read with OCR, the control is found by the text written on it, and the click or the typing
goes through the X server's XTest extension. Nothing here needs desktop accessibility, so this layer is the
fallback when the AT-SPI layer (see `voxa/agent/ui.py` and `voxa/agent/a11y_scope.py`) cannot reach the program -
which is the normal case for a GTK program that started with accessibility off.

Every part is injectable - `capture` (the screenshot), `ocr` (the words on it), `sink` (the input) and `region`
(the window to look in) - so tests exercise the whole path against fakes and never touch a real desktop.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from voxa import simulation
from voxa.agent.host import host_command

log = logging.getLogger("voxa.agent.visual_control")

# The same grabbers voxa/agent/tools/screenshot.py tries, in the same order.
GRABBERS = (
    ("scrot", ["scrot", "{path}"]),
    ("gnome-screenshot", ["gnome-screenshot", "-f", "{path}"]),
    ("import", ["import", "-window", "root", "{path}"]),
    ("grim", ["grim", "{path}"]),
)

TAKING_CONTROL = "Voxa is operating this program for you."


@dataclass(frozen=True, slots=True)
class Word:
    """One word read off the screen, with its box in screen pixels."""

    text: str
    x: int
    y: int
    width: int = 0
    height: int = 0


@dataclass(frozen=True, slots=True)
class Control:
    """A control found by its visible text; x and y are where a click lands on it."""

    text: str
    x: int
    y: int


class InputSink(Protocol):
    """How the click and the keystrokes reach the screen. `XTestSink` is the real one; tests inject a fake."""

    def click(self, x: int, y: int) -> bool: ...

    def type_text(self, text: str) -> bool: ...


class XTestSink(InputSink):
    """Synthesize the click or the typing through the X server's XTest extension.

    `xdotool` is the XTest client Spaced ships (it drives the XTEST protocol the same way the `xte`/`xtea`
    clients do), so one action is one command.
    """

    def __init__(self, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
        self.runner = runner

    def _run(self, argv: list[str]) -> bool:
        if simulation.actions_simulated():
            log.info("simulated action: would run %s", argv)
            return True
        try:
            proc = self.runner(host_command(argv), capture_output=True, text=True, timeout=15, check=False)
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def click(self, x: int, y: int) -> bool:
        return self._run(["xdotool", "mousemove", str(x), str(y), "click", "1"])

    def type_text(self, text: str) -> bool:
        return self._run(["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text])


def norm(text: str) -> str:
    """Fold a label the way voxa/agent/uihost_script.py does, so both layers match the same words."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").replace("_", "").casefold())).strip()


def score(wanted: str, have: str) -> int:
    """3 exact, 2 starts with, 1 contains, 0 no match."""
    wanted, have = norm(wanted), norm(have)
    if not wanted or not have:
        return 0
    if wanted == have:
        return 3
    if have.startswith(wanted):
        return 2
    return 1 if wanted in have else 0


def parse_words(tsv: str) -> list[Word]:
    """Read tesseract's TSV output into words with boxes; [] when there is nothing to read."""
    words: list[Word] = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            left, top = int(row["left"]), int(row["top"])
            width, height = int(row.get("width") or 0), int(row.get("height") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        words.append(Word(text, left, top, width, height))
    return words


def group_lines(words: list[Word]) -> list[tuple[str, int, int, int, int]]:
    """Join words that sit on one line of the screen: (text, left, top, right, bottom).

    A button's label is usually several words, so matching whole lines is what finds "Install Selected".
    """
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda item: (item.y, item.x)):
        height = word.height or 12
        for line in lines:
            last = line[-1]
            if abs((word.y + word.height / 2) - (last.y + last.height / 2)) <= max(4, min(height, last.height or 12) / 2):
                line.append(word)
                break
        else:
            lines.append([word])
    grouped: list[tuple[str, int, int, int, int]] = []
    for line in lines:
        text = " ".join(item.text for item in line)
        left = min(item.x for item in line)
        top = min(item.y for item in line)
        right = max(item.x + item.width for item in line)
        bottom = max(item.y + item.height for item in line)
        grouped.append((text, left, top, right, bottom))
    return grouped


def _in_region(word: Word, region: tuple[int, int, int, int] | None) -> bool:
    if region is None:
        return True
    left, top, right, bottom = region
    return left <= word.x <= right and top <= word.y <= bottom


def window_region(app: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> tuple[int, int, int, int] | None:
    """The box of the named window, so only that window is searched and clicked; None for the whole screen."""
    if not app or simulation.actions_simulated():
        return None
    try:
        found = runner(
            host_command(["xdotool", "search", "--name", "--maxlevels", "0", app]),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    ids = (found.stdout or "").split()
    if not ids:
        return None
    try:
        proc = runner(
            host_command(["xdotool", "getwindowgeometry", "--shell", ids[0]]),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values: dict[str, int] = {}
    for line in (proc.stdout or "").splitlines():
        key, _, value = line.partition("=")
        if key.strip() in {"X", "Y", "Width", "Height"}:
            try:
                values[key.strip()] = int(value.strip())
            except ValueError:
                return None
    if len(values) != 4:
        return None
    return (values["X"], values["Y"], values["X"] + values["Width"], values["Y"] + values["Height"])


def capture_screen() -> str:
    """Write the screen to a PNG and return its path; "" when no grabber works."""
    if simulation.actions_simulated():
        log.info("simulated action: would take a screenshot")
        return ""
    handle, path = tempfile.mkstemp(prefix="voxa-screen-", suffix=".png")
    os.close(handle)
    for program, template in GRABBERS:
        probe = subprocess.run(
            host_command(["sh", "-c", 'command -v "$1"', "sh", program]),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if probe.returncode != 0:
            continue
        command = [part.replace("{path}", path) for part in template]
        subprocess.run(host_command(command), capture_output=True, timeout=10, check=False)
        try:
            if os.path.getsize(path) > 0:
                return path
        except OSError:
            continue
    try:
        os.unlink(path)
    except OSError:
        pass
    return ""


def ocr_words(path: str) -> list[Word]:
    """Read the words off a screenshot with tesseract; [] when tesseract is missing or nothing is legible."""
    if not path or simulation.actions_simulated():
        return []
    try:
        proc = subprocess.run(
            host_command(["tesseract", path, "-", "tsv"]),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_words(proc.stdout or "")


def find_control(
    app: str,
    label: str,
    *,
    region: tuple[int, int, int, int] | None = None,
    capture: Callable[[], str] | None = None,
    ocr: Callable[[str], list[Word]] | None = None,
    region_of: Callable[[str], tuple[int, int, int, int] | None] | None = None,
) -> Control | None:
    """The control whose visible text matches `label`, best match wins (exact, then starts with, then contains)."""
    capture = capture or capture_screen
    ocr = ocr or ocr_words
    if region is None:
        region_of = region_of or (lambda name: window_region(name))
        region = region_of(app)
    path = capture()
    if not path:
        return None
    words = [word for word in (ocr(path) or []) if _in_region(word, region)]
    best_score, best = 0, None
    for text, left, top, right, bottom in group_lines(words):
        value = score(label, text)
        if value > best_score:
            best_score, best = value, (text, (left + right) // 2, (top + bottom) // 2)
            if value == 3:
                break
    if best is None:
        return None
    return Control(text=best[0], x=best[1], y=best[2])


def _announce(notice: Callable[[str], None] | None, app: str) -> None:
    """Say that Voxa has taken control of the program, and log it either way. Never raises."""
    message = f"{TAKING_CONTROL} {app}" if app else TAKING_CONTROL
    log.info("%s", message)
    if notice is None:
        return
    try:
        notice(message)
    except Exception:
        log.warning("the taking-control notice failed", exc_info=True)


def press_text(
    app: str,
    label: str,
    *,
    sink: InputSink | None = None,
    capture: Callable[[], str] | None = None,
    ocr: Callable[[str], list[Word]] | None = None,
    region: tuple[int, int, int, int] | None = None,
    notice: Callable[[str], None] | None = None,
) -> dict:
    """Click the control called `label` in `app`'s window, reading the screen instead of the accessibility bus.

    The answer has the same shape as `voxa/agent/ui.py`'s: {"ok": true, "clicked": {...}} or {"ok": false, "error": ...}.
    """
    if simulation.actions_simulated():
        log.info("simulated action: would click %r in %r", label, app)
        return {"ok": True, "simulated": True}
    control = find_control(app, label, region=region, capture=capture, ocr=ocr)
    if control is None:
        return {"ok": False, "error": "not found on screen", "app": app}
    sink = sink or XTestSink()
    _announce(notice, app)
    if not sink.click(control.x, control.y):
        return {"ok": False, "error": "input failed", "app": app, "clicked": {"text": control.text}}
    return {"ok": True, "app": app, "clicked": {"text": control.text, "x": control.x, "y": control.y}, "layer": "visual"}


def type_at(
    app: str,
    label: str,
    text: str,
    *,
    sink: InputSink | None = None,
    capture: Callable[[], str] | None = None,
    ocr: Callable[[str], list[Word]] | None = None,
    region: tuple[int, int, int, int] | None = None,
    notice: Callable[[str], None] | None = None,
) -> dict:
    """Click into the named control to focus it, then type `text` there."""
    if simulation.actions_simulated():
        log.info("simulated action: would type %r into %r", text, label)
        return {"ok": True, "simulated": True}
    control = find_control(app, label, region=region, capture=capture, ocr=ocr)
    if control is None:
        return {"ok": False, "error": "not found on screen", "app": app}
    sink = sink or XTestSink()
    _announce(notice, app)
    if not sink.click(control.x, control.y) or not sink.type_text(text):
        return {"ok": False, "error": "input failed", "app": app, "typed": text}
    return {"ok": True, "app": app, "typed": text, "at": {"text": control.text, "x": control.x, "y": control.y}}
