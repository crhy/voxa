from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable

from voxa import simulation
from voxa.agent.filematch import best_match
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage
from voxa.agent.transcribe_file import AUDIO_EXTENSIONS, output_path, pieces, spoken_done, tidy

log = logging.getLogger(__name__)

transcriber: Callable[[bytes], str] | None = None  # set by the window to the loaded Whisper model; None means "not ready"


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


def _run_bytes(command: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, b"", b"")
    return subprocess.run(host_command(command), capture_output=True, timeout=timeout, check=False)


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


def _transcribe_handler(args: dict[str, str]) -> ToolResult:
    name = args["name"]
    try:
        source = _locate(name, args.get("folder", ""))
        if source is None:
            return ToolResult.failure(f"I can't find a recording called {name}.")
        if os.path.splitext(source)[1].lower() not in AUDIO_EXTENSIONS:
            return ToolResult.failure("That is not a recording I can transcribe.")
        model = transcriber
        if model is None:
            return ToolResult.failure("My hearing model is still loading.")
        pcm = _run_bytes(["ffmpeg", "-v", "error", "-i", source, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]).stdout
        if not pcm:
            if simulation.actions_simulated():
                return ToolResult.success(spoken_done(source, 0))
            return ToolResult.failure("I could not read any sound from that file.")
        text = tidy([model(piece) for piece in pieces(pcm)])
        _run(["sh", "-c", 'cat > "$1"', "sh", output_path(source)], stdin=text)
    except (OSError, subprocess.SubprocessError) as error:
        return ToolResult.failure(f"That did not work: {filemanage._error_text(error)}")
    return ToolResult.success(spoken_done(source, len(text.split())))


def transcribe_file_tools() -> list[Tool]:
    return [
        Tool(
            name="transcribe_file",
            description="Transcribe an audio or video recording into a text file next to it, using the Whisper model.",
            parameters={"name": "the recording to transcribe", "folder": "which folder it is in"},
            risk=RiskLevel.REVERSIBLE,
            handler=_transcribe_handler,
            required=("name",),
        ),
    ]
