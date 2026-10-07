from __future__ import annotations

import logging
import os
import subprocess

from voxa import simulation
from voxa.agent.filematch import (
    STANDARD_NAMES,
    best_match,
    count_noun,
    folder_key,
    resolve_folder,
    size_bytes,
    speakable,
    spoken_list,
)
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger(__name__)

LAST_UNDO: tuple[str, list[list[str]]] | None = None


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
    global LAST_UNDO
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
        LAST_UNDO = (f"Removed the copy of {match}.", [["gio", "trash", os.path.join(destination, match)]])
        return ToolResult.success(f"Copied {match} to {args['destination']}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def move_file(args: dict[str, str]) -> ToolResult:
    global LAST_UNDO
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
        LAST_UNDO = (
            f"Moved {match} back to {args['source']}.",
            [["gio", "move", os.path.join(destination, match), source + "/"]],
        )
        return ToolResult.success(f"Moved {match} to {args['destination']}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def trash_file(args: dict[str, str]) -> ToolResult:
    global LAST_UNDO
    try:
        source = _folder(args["folder"])
        if source is None:
            return ToolResult.failure(f"I do not know the folder {args['folder']}.")
        match = best_match(args["name"], _names(source))
        if match is None:
            return ToolResult.failure(f"I could not find {args['name']} in {args['folder']}.")
        _run(["gio", "trash", os.path.join(source, match)])
        LAST_UNDO = (f"Put {match} back.", [["gio", "trash", "--restore", "trash:///" + match]])
        return ToolResult.success(f"Moved {match} to the trash.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def empty_trash(args: dict[str, str]) -> ToolResult:
    global LAST_UNDO
    try:
        _run(["gio", "trash", "--empty"])
        LAST_UNDO = None
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
            [
                "timeout",
                "15",
                "find",
                home,
                "-xdev",
                "(",
                "-name",
                ".*",
                "-o",
                "-name",
                "node_modules",
                "-o",
                "-name",
                "__pycache__",
                ")",
                "-prune",
                "-o",
                "-iname",
                pattern,
                "-print",
            ],
            check=False,
        )
        all_lines = out.stdout.splitlines()
        if not all_lines:
            if out.returncode == 124:
                return ToolResult.failure("I ran out of time looking. Try naming a folder.")
            return ToolResult.failure(f"I could not find a file called {args['name']}.")
        top = set(STANDARD_NAMES.values())

        def rank(path: str) -> tuple[int, int]:
            parent = os.path.dirname(path)
            is_top = os.path.basename(parent) in top or parent.rstrip("/") == home.rstrip("/")
            return (0 if is_top else 1, path.count("/"))

        lines = sorted(all_lines, key=rank)[:20]
        if len(lines) == 1:
            _run(["gio", "open", os.path.dirname(lines[0])], check=False)
        total = len(all_lines)
        if total > 3:
            entries = [f"{os.path.basename(p)} in {os.path.basename(os.path.dirname(p))}" for p in lines[:3]]
            return ToolResult.success(f"I found {total}. The closest are {spoken_list(entries, 3)}.")
        entries = [f"{os.path.basename(p)} in {os.path.basename(os.path.dirname(p))}" for p in lines]
        return ToolResult.success(f"I found {total}: {spoken_list(entries)}.")
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
        spoken = speakable(names, 5)
        if not spoken:
            return ToolResult.success(f"{count_noun(len(names), 'item')} in {args['folder']}.")
        return ToolResult.success(
            f"{count_noun(len(names), 'item')} in {args['folder']}, including {spoken_list(spoken, 5)}."
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def rename_file(args: dict[str, str]) -> ToolResult:
    global LAST_UNDO
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
        LAST_UNDO = (f"Renamed it back to {match}.", [["gio", "rename", os.path.join(path, new_file_name), match]])
        return ToolResult.success(f"Renamed {match} to {new_file_name}.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def make_folder(args: dict[str, str]) -> ToolResult:
    global LAST_UNDO
    try:
        path = _folder(args["folder"])
        if path is None:
            return ToolResult.failure(f"I do not know the folder {args['folder']}.")
        new_name = args["name"].strip().title()
        _run(["mkdir", "--", os.path.join(path, new_name)])
        LAST_UNDO = (f"Removed the folder {new_name}.", [["rmdir", "--", os.path.join(path, new_name)]])
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
            [
                "timeout",
                "15",
                "find",
                home,
                "-xdev",
                "(",
                "-name",
                ".*",
                "-o",
                "-name",
                "node_modules",
                "-o",
                "-name",
                "__pycache__",
                ")",
                "-prune",
                "-o",
                "-type",
                "f",
                "-size",
                f"+{limit}c",
                "-print",
            ],
            check=False,
        )
        lines = out.stdout.splitlines()[:20]
        if not lines:
            if out.returncode == 124:
                return ToolResult.failure("I ran out of time looking. Try naming a folder.")
            return ToolResult.failure("No files are bigger than that.")
        names = [os.path.basename(path) for path in lines]
        return ToolResult.success(
            f"I found {count_noun(len(lines), 'file')} bigger than {args['amount']} {args['unit']}: {spoken_list(names)}."
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {_error_text(error)}")


def undo_file_action(args: dict[str, str]) -> ToolResult:
    global LAST_UNDO
    if LAST_UNDO is None:
        if args.get("otherwise") == "key":
            # A bare "undo that" with no file action behind it means the ordinary undo of the focused program.
            from voxa.agent.tools.typing import _press_key_handler

            return _press_key_handler({"key": "undo"})
        return ToolResult.failure("There is no file action to undo.")
    sentence, commands = LAST_UNDO
    LAST_UNDO = None
    try:
        for command in commands:
            _run(command)
        return ToolResult.success(sentence)
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
        Tool(
            name="undo_file_action",
            description="Undo the last file action that was done.",
            parameters={"otherwise": "what to do when there is no file action: key = the ordinary undo"},
            risk=RiskLevel.REVERSIBLE,
            handler=undo_file_action,
            required=(),
        ),
    ]
