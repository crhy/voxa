from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from voxa import apps, simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools.applications import open_app

log = logging.getLogger("voxa.agent.tools.windows")

_THE_PREFIX = re.compile(r"^the\s+", re.IGNORECASE)

ON_LOCK: Callable | None = None

_LOCKERS = (
    ("mate-screensaver-command", "--lock"),
    ("xdg-screensaver", "lock"),
    ("dm-tool", "lock"),
    ("xscreensaver-command", "-lock"),
    ("gnome-screensaver-command", "--lock"),
)


@dataclass(frozen=True, slots=True)
class Window:
    id: str
    wm_class: str
    title: str


def parse_wmctrl(listing: str) -> list[Window]:
    """Turn `wmctrl -lx` output into Window records.

    Each line is "<id> <desktop> <class> <host> <title...>"; the title may be
    missing entirely or contain spaces.
    """
    windows: list[Window] = []
    for line in listing.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        windows.append(Window(parts[0], parts[2], parts[4] if len(parts) > 4 else ""))
    return windows


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def list_windows() -> list[Window]:
    result = _run(["wmctrl", "-lx"])
    return parse_wmctrl(result.stdout)


def match_windows(name: str, windows: list[Window]) -> list[Window]:
    """Windows whose title or wm_class contains the spoken name."""
    needle = _normalize(name)
    if not needle:
        return []
    matches: list[Window] = []
    for window in windows:
        if "voxa" in window.wm_class.casefold() and "voxa" not in needle:
            continue
        if needle in _normalize(window.title) or needle in _normalize(window.wm_class):
            matches.append(window)
    return matches


_DESKTOP_MARKERS = ("desktop_window", "mate-panel", "desktop", "panel")


def window_label(title: str, wm_class: str) -> str:
    """A short name to SAY for a window, from its title or wm_class."""
    parts = re.split(r"\s[-—]\s", title)
    if len(parts) >= 2 and parts[-1].strip():
        label = parts[-1].strip()
    else:
        label = wm_class.rsplit(".", 1)[-1]
        if label:
            label = label[0].upper() + label[1:]
    label = re.sub(r"\s+(?:Web Browser|File Manager)$", "", label, flags=re.IGNORECASE).strip()
    return label[:30]


def app_name_index(apps_list: list) -> dict[str, str]:
    """Map lower-cased app keys to the menu name a person would say."""
    index: dict[str, str] = {}
    for app in apps_list:
        menu = app.name.strip()
        trimmed = re.sub(r"\s+(?:Web Browser|File Manager|Text Editor)$", "", menu, flags=re.IGNORECASE).strip()
        if len(trimmed) >= 3:
            menu = trimmed
        menu = re.sub(r"^(?:MATE|GNOME)\s+", "", menu, flags=re.IGNORECASE).strip() or menu
        stem = app.path.rsplit("/", 1)[-1]
        if stem.endswith(".desktop"):
            stem = stem[: -len(".desktop")]
        parts = [p for p in stem.split(".") if p]
        exec_base = parts[-1] if parts else stem
        keys = {app.name.lower(), stem, exec_base}
        for key in list(keys):
            keys.add(key.replace("-", "").replace("_", ""))
        for key in keys:
            if key:
                index.setdefault(key, menu)
    return index


def friendly_label(title: str, wm_class: str, app_names: dict[str, str]) -> str:
    """The name a person would use for a window, preferring the app menu name."""
    parts = wm_class.rsplit(".", 1)
    for part in parts:
        key = part.casefold()
        if key in app_names:
            return app_names[key]
    for part in parts:
        key = part.casefold().replace("-", "").replace("_", "")
        if key in app_names:
            return app_names[key]
    base = window_label(title, wm_class)
    title_parts = re.split(r"\s[-—]\s", title)
    from_title = len(title_parts) >= 2 and bool(title_parts[-1].strip())
    if from_title and len(base) <= 4 and base.isalnum():
        cls = wm_class.rsplit(".", 1)[-1]
        base = cls[0].upper() + cls[1:] if cls else base
    if ("-" in base or "_" in base) and " " not in base:
        base = base.replace("-", " ").replace("_", " ")
        base = re.sub(r"^(?:mate|gnome|xfce4|org gnome) ", "", base, flags=re.IGNORECASE)
        base = base.title()
    return base


_NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def describe_windows(labels: list[str]) -> str:
    """Spoken summary of the open windows, grouping duplicates."""
    if not labels:
        return "Nothing is open."
    counts: dict[str, int] = {}
    order: list[str] = []
    for label in labels:
        if label not in counts:
            counts[label] = 0
            order.append(label)
        counts[label] += 1
    parts: list[str] = []
    for label in order:
        count = counts[label]
        if count == 1:
            parts.append(label)
        else:
            word = _NUMBER_WORDS.get(count, str(count))
            parts.append(f"{word} {label} windows")
    if len(parts) > 6:
        shown = parts[:6]
        more = len(parts) - 6
        shown.append(f"{more} more")
    else:
        shown = parts
    if len(shown) == 1:
        joined = shown[0]
    else:
        joined = ", ".join(shown[:-1]) + " and " + shown[-1]
    total = len(labels)
    if total == 1:
        head = "You have one window open: "
    else:
        head = f"You have {total} windows open: "
    return head + joined + "."


def flatpak_app_id(desktop_path: str) -> str | None:
    """The Flatpak app id behind a .desktop file, or None for a non-Flatpak app."""
    if "flatpak/exports/" not in desktop_path:
        return None
    return desktop_path.rsplit("/", 1)[-1].removesuffix(".desktop")


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        capture_output=True,
        text=True,
        timeout=5,
        check=check,
    )


_GENERIC_BROWSER = {"browser", "web browser", "internet browser", "your browser", "voxa's browser", "voxa browser"}


def _close_own_browser(name: str) -> ToolResult | None:
    """"Close the browser" means the browser Voxa opened, never the user's own with all their tabs.

    Returns None when the request is not about a generic "browser" (e.g. "close Brave": the user's choice).
    """
    if name.lower() not in _GENERIC_BROWSER:
        return None
    try:
        from ..browser_session import browser_socket
        from .web import get_session

        session = get_session()
        if browser_socket(session._port) is not None:
            session.close_browser()
            return ToolResult.success("Closing my browser.")
    except Exception as exc:  # noqa: BLE001 - fall through to the honest answer below
        log.debug("could not close the controlled browser: %s", exc)
    return ToolResult.failure("My browser isn't open. Say the browser's name, like “close Brave”, to close yours.")


def close_app(args: dict[str, str]) -> ToolResult:
    name = _THE_PREFIX.sub("", args["name"].strip())
    own = _close_own_browser(name)
    if own is not None:
        return own
    windows = match_windows(name, list_windows())
    app = apps.match_app(name, apps.list_apps())
    app_id = flatpak_app_id(app.path) if app is not None else None

    if windows:
        for window in windows:
            _run(["wmctrl", "-i", "-c", window.id])
        # A Flatpak app can linger behind a "save changes?" prompt, so kill it too.
        if app_id is not None:
            _run(["flatpak", "kill", app_id], check=False)
        return ToolResult.success(f"Closing {name}.")

    if app_id is not None:
        result = _run(["flatpak", "kill", app_id], check=False)
        if result.returncode == 0:
            return ToolResult.success(f"Closing {app.name}.")

    return ToolResult.failure(f"{name} doesn't seem to be open.")


def close_window(args: dict[str, str]) -> ToolResult:
    _run(["wmctrl", "-c", ":ACTIVE:"])
    return ToolResult.success("Closed.")


def switch_to(args: dict[str, str]) -> ToolResult:
    name = _THE_PREFIX.sub("", args["name"].strip())
    windows = match_windows(name, list_windows())
    if not windows:
        # "bring up the calculator" should open it when it is not already running.
        return open_app({"name": name})
    _run(["wmctrl", "-i", "-a", windows[0].id])
    return ToolResult.success(f"Switching to {name}.")


def _active_window_id() -> str:
    result = _run(["xdotool", "getactivewindow"], check=False)
    return result.stdout.strip()


def minimize_app(args: dict[str, str]) -> ToolResult:
    name = _THE_PREFIX.sub("", args["name"].strip())
    if not name:
        window_id = _active_window_id()
        if not window_id:
            return ToolResult.failure("Nothing seems to be open.")
        _run(["xdotool", "windowminimize", window_id], check=False)
        return ToolResult.success("Minimized.")
    windows = match_windows(name, list_windows())
    if not windows:
        return ToolResult.failure(f"{name} does not seem to be open.")
    for window in windows:
        _run(["xdotool", "windowminimize", window.id], check=False)
    return ToolResult.success(f"Minimized {name}.")


def maximize_app(args: dict[str, str]) -> ToolResult:
    name = _THE_PREFIX.sub("", args["name"].strip())
    if not name:
        window_id = _active_window_id()
        if not window_id:
            return ToolResult.failure("Nothing seems to be open.")
        _run(["wmctrl", "-i", "-r", window_id, "-b", "add,maximized_vert,maximized_horz"], check=False)
        return ToolResult.success("Maximized.")
    windows = match_windows(name, list_windows())
    if not windows:
        return ToolResult.failure(f"{name} does not seem to be open.")
    for window in windows:
        _run(["wmctrl", "-i", "-r", window.id, "-b", "add,maximized_vert,maximized_horz"], check=False)
    return ToolResult.success(f"Maximized {name}.")


def restore_app(args: dict[str, str]) -> ToolResult:
    name = _THE_PREFIX.sub("", args["name"].strip())
    if not name:
        window_id = _active_window_id()
        if not window_id:
            return ToolResult.failure("Nothing seems to be open.")
        _run(["wmctrl", "-i", "-r", window_id, "-b", "remove,maximized_vert,maximized_horz"], check=False)
        _run(["wmctrl", "-i", "-a", window_id], check=False)
        return ToolResult.success("Restored.")
    windows = match_windows(name, list_windows())
    if not windows:
        return ToolResult.failure(f"{name} does not seem to be open.")
    for window in windows:
        _run(["wmctrl", "-i", "-r", window.id, "-b", "remove,maximized_vert,maximized_horz"], check=False)
        _run(["wmctrl", "-i", "-a", window.id], check=False)
    return ToolResult.success(f"Restored {name}.")


def minimize_all(args: dict[str, str]) -> ToolResult:
    _run(["wmctrl", "-k", "on"], check=False)
    return ToolResult.success("Minimized everything.")


def lock_screen(args: dict[str, str]) -> ToolResult:
    if ON_LOCK is not None:
        ON_LOCK()
    for name, flag in _LOCKERS:
        probe = _run(["sh", "-c", f"command -v {name}"], check=False)
        if probe.returncode != 0:
            continue
        if name == "mate-screensaver-command":
            daemon = _run(["sh", "-c", "pgrep -x mate-screensaver"], check=False)
            if daemon.returncode != 0:
                _run(["sh", "-c", "mate-screensaver &"], check=False)
        result = _run([name, flag], check=False)
        if result.returncode == 0:
            return ToolResult.success("Locking the screen.")
    return ToolResult.failure("I could not find a screen locker on this computer.")


def _is_desktop(window: Window) -> bool:
    """True for the desktop's own windows and Voxa's own windows."""
    wm = window.wm_class.casefold()
    if any(marker in wm for marker in _DESKTOP_MARKERS):
        return True
    return window.title.strip() == "Voxa"


def list_open_windows(args: dict[str, str]) -> ToolResult:
    try:
        index = app_name_index(apps.list_apps())
    except Exception:
        index = {}
    labels = [
        friendly_label(w.title, w.wm_class, index)
        for w in list_windows()
        if not _is_desktop(w)
    ]
    return ToolResult.success(describe_windows(labels))


def active_window_name(args: dict[str, str]) -> ToolResult:
    result = _run(["xdotool", "getactivewindow", "getwindowname"], check=False)
    title = result.stdout.strip()
    if not title:
        return ToolResult.failure("I cannot tell which window is in front.")
    return ToolResult.success(f"This is {title[:80]}.")


def window_tools() -> list[Tool]:
    return [
        Tool(
            name="close_app",
            description="Close an open application by name.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=close_app,
            required=("name",),
        ),
        Tool(
            name="close_window",
            description="Close the window that currently has focus.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=close_window,
            required=(),
        ),
        Tool(
            name="switch_to",
            description="Focus an already open application by name.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=switch_to,
            required=("name",),
        ),
        Tool(
            name="minimize_app",
            description="Minimize an open application by name, or the focused window when no name is given.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=minimize_app,
            required=(),
        ),
        Tool(
            name="maximize_app",
            description="Maximize an open application by name, or the focused window when no name is given.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=maximize_app,
            required=(),
        ),
        Tool(
            name="restore_app",
            description="Restore an application to its normal size by name, or the focused window when no name is given.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=restore_app,
            required=(),
        ),
        Tool(
            name="minimize_all",
            description="Minimize every open window to show the desktop.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=minimize_all,
            required=(),
        ),
        Tool(
            name="list_open_windows",
            description="Say which windows are open.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=list_open_windows,
            required=(),
        ),
        Tool(
            name="active_window_name",
            description="Say which window is in front.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=active_window_name,
            required=(),
        ),
        Tool(
            name="lock_screen",
            description="Lock the screen so it turns off or blanks.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=lock_screen,
            required=(),
        ),
    ]
