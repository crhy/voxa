from __future__ import annotations

import logging
import os
import subprocess

from voxa import simulation
from voxa.agent.convert import command, output_path, spoken_done, target_extension
from voxa.agent.filematch import best_match
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)


def _run(command: list[str], stdin: str | None = None, check: bool = True, timeout: float = 6) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _locate(name: str, folder: str) -> str | None:
    if folder:
        path = filemanage._folder(folder)
        if path is None:
            return None
        match = best_match(name, filemanage._names(path))
        return os.path.join(path, match) if match is not None else None
    for spoken in ("desktop", "documents", "downloads"):
        path = filemanage._folder(spoken)
        if path is None:
            continue
        match = best_match(name, filemanage._names(path))
        if match is not None:
            return os.path.join(path, match)
    return None


def _convert_handler(args: dict[str, str]) -> ToolResult:
    name = args["name"]
    ext = target_extension(args["format"])
    if ext is None:
        return ToolResult.failure("I can't convert to that format.")
    try:
        source = _locate(name, args.get("folder", ""))
        if source is None:
            return ToolResult.failure(f"I can't find a file called {name}.")
        out = output_path(source, ext)
        _run(command(source, ext), check=False, timeout=60)
        if simulation.actions_simulated():
            return ToolResult.success(spoken_done(source, ext))
        found = _run(["test", "-f", out], check=False)
        if found.returncode != 0:
            return ToolResult.failure("The conversion did not work.")
        return ToolResult.success(spoken_done(source, ext))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(f"That did not work: {filemanage._error_text(error)}")


def convert_tools() -> list[Tool]:
    return [
        Tool(
            name="convert_file",
            description="Convert a document or image to another file format, keeping the result next to the original.",
            parameters={"name": "the file to convert", "format": "the format to convert to", "folder": "which folder it is in"},
            risk=RiskLevel.REVERSIBLE,
            handler=_convert_handler,
            required=("name", "format"),
        ),
    ]
