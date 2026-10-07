from __future__ import annotations

import logging
import os
import subprocess

from voxa import simulation
from voxa.agent.filematch import best_match, folder_key, resolve_folder, size_bytes, spoken_list
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=20,
        check=check,
    )


def _folder(spoken: str) -> str | None:
    key = folder_key(spoken)
    if key is None:
        return None
    home = _run(["sh", "-c", "echo $HOME"]).stdout.strip()
    if key == "":
        return home
    reported = _run(["xdg-user-dir", key], check=False).stdout.strip()
    return resolve_folder(key, home, reported)


def _names(folder: str) -> list[str]:
    out = _run(["ls", "-1A", folder])
    return [line for line in out.stdout.splitlines() if not line.startswith(".")]


def _error_text(error: Exception) -> str:
    stderr = getattr(error, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    if isinstance(stderr, str) and stderr.strip():
        return stderr.strip().splitlines()[0]
    return str(error).strip().splitlines()[0] if str(error).strip() else "unknown error"


def copy_file(args: dict[str, str]) -> ToolResult:
    try:
        source = _folder(args["source"])
        destination = _folder(args["destination"])
        if source is None:
            return ToolResult.failure(f"I do not know the folder {args['source']}.")
        if destination is None:
            return ToolResult.failure(f"I do not know the folder {args['destination']}.")
        match = best_match(args["name"], _names(source))
        if match is None:
            return ToolResult.failure(f"I could not find {args['name']} in {args['source']}.")
        _run(["gio", "copy", os.path.join(source, match), destination + "/"])
        return ToolResult.success(f"Copied {match} to {args['destination']}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def move_file(args: dict[str, str]) -> ToolResult:
    try:
        source = _folder(args["source"])
        destination = _folder(args["destination"])
        if source is None:
            return ToolResult.failure(f"I do not know the folder {args['source']}.")
        if destination is None:
            return ToolResult.failure(f"I do not know the folder {args['destination']}.")
        match = best_match(args["name"], _names(source))
        if match is None:
            return ToolResult.failure(f"I could not find {args['name']} in {args['source']}.")
        _run(["gio", "move", os.path.join(source, match), destination + "/"])
        return ToolResult.success(f"Moved {match} to {args['destination']}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def trash_file(args: dict[str, str]) -> ToolResult:
    try:
        source = _folder(args["folder"])
        if source is None:
            return ToolResult.failure(f"I do not know the folder {args['folder']}.")
        match = best_match(args["name"], _names(source))
        if match is None:
            return ToolResult.failure(f"I could not find {args['name']} in {args['folder']}.")
        _run(["gio", "trash", os.path.join(source, match)])
        return ToolResult.success(f"Moved {match} to the trash.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def empty_trash(args: dict[str, str]) -> ToolResult:
    try:
        _run(["gio", "trash", "--empty"])
        return ToolResult.success("The trash is empty.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def find_file(args: dict[str, str]) -> ToolResult:
    try:
        home = _folder("home")
        if home is None:
            return ToolResult.failure("I cannot reach your home folder.")
        words = args["name"].split()
        pattern = "*" + "*".join(words) + "*"
        out = _run(
            ["timeout", "12", "find", home, "-xdev", "-not", "-path", "*/.*", "-iname", pattern],
            check=False,
        )
        lines = out.stdout.splitlines()[:20]
        if not lines:
            if out.returncode == 124:
                return ToolResult.failure("I ran out of time looking. Try naming a folder.")
            return ToolResult.failure(f"I could not find a file called {args['name']}.")
        entries = [
            f"{os.path.basename(path)} in {os.path.basename(os.path.dirname(path))}" for path in lines
        ]
        if len(lines) == 1:
            parent = os.path.dirname(lines[0])
            _run(["gio", "open", parent], check=False)
        return ToolResult.success(f"I found {len(lines)}: {spoken_list(entries)}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def open_folder(args: dict[str, str]) -> ToolResult:
    try:
        path = _folder(args["folder"])
        if path is None:
            return ToolResult.failure(f"I do not know the folder {args['folder']}.")
        _run(["gio", "open", path])
        return ToolResult.success(f"Opening {args['folder']}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def list_folder(args: dict[str, str]) -> ToolResult:
    try:
        path = _folder(args["folder"])
        if path is None:
            return ToolResult.failure(f"I do not know the folder {args['folder']}.")
        names = _names(path)
        if not names:
            return ToolResult.success(f"{os.path.basename(path)} is empty.")
        return ToolResult.success(
            f"{len(names)} items in {args['folder']}: {spoken_list(names, 5)}."
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def rename_file(args: dict[str, str]) -> ToolResult:
    try:
        path = _folder(args["folder"])
        if path is None:
            return ToolResult.failure(f"I do not know the folder {args['folder']}.")
        match = best_match(args["name"], _names(path))
        if match is None:
            return ToolResult.failure(f"I could not find {args['name']} in {args['folder']}.")
        old_path = os.path.join(path, match)
        new_file_name = args["new_name"].strip()
        if not os.path.splitext(new_file_name)[1]:
            new_file_name += os.path.splitext(match)[1]
        _run(["gio", "rename", old_path, new_file_name])
        return ToolResult.success(f"Renamed {match} to {new_file_name}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def make_folder(args: dict[str, str]) -> ToolResult:
    try:
        path = _folder(args["folder"])
        if path is None:
            return ToolResult.failure(f"I do not know the folder {args['folder']}.")
        new_name = args["name"].strip().title()
        _run(["mkdir", "--", os.path.join(path, new_name)])
        return ToolResult.success(f"Created the folder {new_name} in {args['folder']}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def find_large_files(args: dict[str, str]) -> ToolResult:
    try:
        home = _folder("home")
        if home is None:
            return ToolResult.failure("I cannot reach your home folder.")
        limit = size_bytes(args["amount"], args["unit"])
        out = _run(
            ["timeout", "12", "find", home, "-xdev", "-not", "-path", "*/.*", "-type", "f", "-size", f"+{limit}c"],
            check=False,
        )
        lines = out.stdout.splitlines()[:20]
        if not lines:
            if out.returncode == 124:
                return ToolResult.failure("I ran out of time looking. Try naming a folder.")
            return ToolResult.failure("No files are bigger than that.")
        names = [os.path.basename(path) for path in lines]
        return ToolResult.success(
            f"I found {len(lines)} files bigger than {args['amount']} {args['unit']}: {spoken_list(names)}."
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def file_manage_tools() -> list[Tool]:
    return [
        Tool(
            name="copy_file",
            description="Copy a file from one folder to another.",
            parameters={
                "name": "the file to copy",
                "source": "the folder to copy from",
                "destination": "the folder to copy to",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=copy_file,
            required=("name", "source", "destination"),
        ),
        Tool(
            name="move_file",
            description="Move a file from one folder to another.",
            parameters={
                "name": "the file to move",
                "source": "the folder to move from",
                "destination": "the folder to move to",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=move_file,
            required=("name", "source", "destination"),
        ),
        Tool(
            name="trash_file",
            description="Move a file to the trash.",
            parameters={"name": "the file to trash", "folder": "the folder it is in"},
            risk=RiskLevel.REVERSIBLE,
            handler=trash_file,
            required=("name", "folder"),
        ),
        Tool(
            name="empty_trash",
            description="Empty the trash.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=empty_trash,
            required=(),
        ),
        Tool(
            name="find_file",
            description="Find a file by name anywhere in the home folder.",
            parameters={"name": "the file name to look for"},
            risk=RiskLevel.REVERSIBLE,
            handler=find_file,
            required=("name",),
        ),
        Tool(
            name="find_large_files",
            description="Find files bigger than a given size in the home folder.",
            parameters={"amount": "the size amount", "unit": "the size unit, megabytes or gigabytes"},
            risk=RiskLevel.REVERSIBLE,
            handler=find_large_files,
            required=("amount", "unit"),
        ),
        Tool(
            name="open_folder",
            description="Open a folder.",
            parameters={"folder": "the folder to open"},
            risk=RiskLevel.REVERSIBLE,
            handler=open_folder,
            required=("folder",),
        ),
        Tool(
            name="list_folder",
            description="List the items in a folder.",
            parameters={"folder": "the folder to list"},
            risk=RiskLevel.REVERSIBLE,
            handler=list_folder,
            required=("folder",),
        ),
        Tool(
            name="rename_file",
            description="Rename a file in a folder.",
            parameters={
                "name": "the file to rename",
                "new_name": "the new name for the file",
                "folder": "the folder the file is in",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=rename_file,
            required=("name", "new_name", "folder"),
        ),
        Tool(
            name="make_folder",
            description="Create a new folder inside a folder.",
            parameters={"name": "the name of the new folder", "folder": "the folder to create it in"},
            risk=RiskLevel.REVERSIBLE,
            handler=make_folder,
            required=("name", "folder"),
        ),
    ]
