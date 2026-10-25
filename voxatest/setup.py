from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path


def _default_host(cmd: list[str], *, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    if Path("/.flatpak-info").exists():
        cmd = ["flatpak-spawn", "--host", *cmd]
    return subprocess.run(cmd, input=stdin, timeout=20, check=False, capture_output=True)


def _host_path(path: str) -> str:
    if path.startswith("~"):
        return "$HOME" + path[1:]
    return path


def _default_clipboard(text: str) -> bool:
    """Put text on the desktop clipboard with GTK (xclip is not installed on these machines)."""
    try:
        import gi

        gi.require_version("Gdk", "4.0")
        from gi.repository import Gdk
    except Exception:
        return False
    try:
        display = Gdk.Display.get_default()
        if display is None:
            return False
        display.get_clipboard().set_content(Gdk.ContentProvider.new_for_value(text))
        return True
    except Exception:
        return False


def run_setup(
    steps: list[dict[str, object]],
    *,
    data_dir: Path,
    host: Callable[..., subprocess.CompletedProcess] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clipboard_write: Callable[[str], bool] | None = None,
) -> str:
    if host is None:
        host = _default_host
    for step in steps:
        if "copy" in step:
            src = data_dir / str(step["copy"])
            dest = _host_path(str(step["to"]))
            try:
                content = src.read_bytes()
            except OSError as exc:
                return f"Could not read sample {step['copy']}: {exc}."
            cmd = ["sh", "-c", f'mkdir -p "$(dirname "{dest}")"; cat > "{dest}"']
            result = host(cmd, stdin=content)
            if result.returncode != 0:
                return f"Copy to {step['to']} failed (exit {result.returncode})."
        elif "open" in step:
            path = _host_path(str(step["open"]))
            app = str(step.get("with", "xdg-open"))
            cmd = ["sh", "-c", f"setsid -f {app} \"{path}\""]
            result = host(cmd)
            if result.returncode != 0:
                return f"Could not open {step['open']} with {app}."
        elif "keys" in step:
            cmd = ["xdotool", "key", "--clearmodifiers", str(step["keys"])]
            result = host(cmd)
            if result.returncode != 0:
                return f"Keys {step['keys']} failed (exit {result.returncode})."
        elif "clipboard" in step:
            write = clipboard_write or _default_clipboard
            if not write(str(step["clipboard"])):
                return "Could not put text on the clipboard."
        elif "makefile" in step:
            path = _host_path(str(step["makefile"]))
            size = int(step.get("size", 0))
            cmd = ["sh", "-c", f'mkdir -p "$(dirname "{path}")"; truncate -s {size} "{path}"']
            result = host(cmd)
            if result.returncode != 0:
                return f"Create {step['makefile']} failed (exit {result.returncode})."
        elif "remove" in step:
            path = _host_path(str(step["remove"]))
            flag = "-f -r" if step.get("recursive") else "-f"
            cmd = ["sh", "-c", f"rm {flag} \"{path}\""]
            result = host(cmd)
            if result.returncode != 0:
                return f"Remove {step['remove']} failed (exit {result.returncode})."
        elif "gitrepo" in step:
            path = _host_path(str(step["gitrepo"]))
            script = (
                f"mkdir -p \"{path}\" && cd \"{path}\" && "
                "printf 'first\\n' > notes.txt && "
                "git init -q -b main 2>/dev/null || git init -q && "
                "git -c user.name=voxa -c user.email=voxa@local add -A && "
                'git -c user.name=voxa -c user.email=voxa@local commit -qm "First commit" && '
                "printf 'second\\n' >> notes.txt"
            )
            result = host(["sh", "-c", script])
            if result.returncode != 0:
                return f"Create git repo {step['gitrepo']} failed (exit {result.returncode})."
        elif "wait" in step:
            sleep(float(step["wait"]))
        else:
            return f"Unknown setup step: {step}."
    return ""
