"""Meeting and call notes as voice tools (crhy/voxa#65).

A recording already on the machine is transcribed by the same local Whisper model
the transcribe_file tool uses, turned into one dated block in the Voxa Meeting Notes
file in Documents, and read back later. Nothing is uploaded: the recording is read
locally, the model runs locally, and the block is written locally.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from datetime import date

from voxa import simulation
from voxa.agent import meeting_notes
from voxa.agent.filematch import best_match
from voxa.agent.host import host_command
from voxa.agent.meeting_notes import FILE_NAME, _count_word, _day_phrase
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import calendar_local as calendar_tools
from voxa.agent.tools import filemanage
from voxa.agent.tools import transcribe_file as recording_tools
from voxa.agent.transcribe_file import AUDIO_EXTENSIONS, pieces

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your meeting notes."
NO_RECORDINGS = "I have no meeting notes yet."

transcriber: Callable[[bytes], str] | None = None  # a test's fake; None means "use the loaded model"


def _transcript(texts: list[str]) -> str:
    """Join the transcribed pieces, keeping the lines the speaker labels live on."""
    lines: list[str] = []
    for text in texts:
        lines += [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _model() -> Callable[[bytes], str] | None:
    """The Whisper model: the injected one when a test set it, otherwise the loaded one."""
    if transcriber is not None:
        return transcriber
    return recording_tools.transcriber


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=6,
        check=check,
    )


def _run_bytes(command: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        return subprocess.CompletedProcess(command, 0, b"", b"")
    return subprocess.run(host_command(command), capture_output=True, timeout=timeout, check=False)


def _path() -> str | None:
    folder = filemanage._folder("documents")
    if folder is None:
        return None
    return os.path.join(folder, FILE_NAME)


def _read() -> str:
    path = _path()
    if path is None:
        raise OSError("no documents folder")
    done = _run(["cat", path], check=False)
    if done.returncode != 0:
        return ""
    return done.stdout


def _write(content: str) -> None:
    path = _path()
    if path is None:
        raise OSError("no documents folder")
    _run(["sh", "-c", 'cat > "$1"', "sh", path], stdin=content)


def _locate(name: str, folder: str) -> str | None:
    """The recording the user means, by the same route the transcribe_file tool takes."""
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


def _archive() -> str | None:
    """The meeting notes file, or None when it cannot be read."""
    try:
        return _read()
    except (OSError, subprocess.SubprocessError):
        return None


def _day_window(spoken: str, today: date) -> tuple[date, date] | None:
    """The days a spoken day phrase covers; None when it names no day."""
    if not spoken:
        return None
    span = meeting_notes.relative_span(spoken, today)
    if span is not None:
        return span
    day = meeting_notes.relative_day(spoken, today)
    return (day, day) if day is not None else None


def _found(content: str, args: dict[str, str], today: date) -> list[meeting_notes.Record]:
    items = meeting_notes.search(content, keyword=args.get("keyword", "").strip() or None)
    window = _day_window(args.get("day", "").strip(), today)
    if window is not None:
        items = [item for item in items if window[0] <= item.day <= window[1]]
    return items


def _meeting_note_from_recording_handler(args: dict[str, str]) -> ToolResult:
    name = args["name"].strip()
    if not name:
        return ToolResult.failure("Which recording do you mean?")
    try:
        source = _locate(name, args.get("folder", ""))
        if source is None:
            return ToolResult.failure(f"I can't find a recording called {name}.")
        if os.path.splitext(source)[1].lower() not in AUDIO_EXTENSIONS:
            return ToolResult.failure("That is not a recording I can transcribe.")
        model = _model()
        if model is None:
            return ToolResult.failure("My hearing model is still loading.")
        pcm = _run_bytes(
            ["ffmpeg", "-v", "error", "-i", source, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
        ).stdout
        if not pcm:
            if not simulation.actions_simulated():
                return ToolResult.failure("I could not read any sound from that file.")
            # A simulated run reads no sound at all, so the model is asked once with
            # nothing: that keeps the record, its decisions and its action items on the
            # tested path without a microphone or a real model.
            transcript = _transcript([model(b"")])
        else:
            transcript = _transcript([model(piece) for piece in pieces(pcm)])
        title = args.get("title", "").strip() or os.path.splitext(os.path.basename(source))[0]
        existing = _archive()
        if existing is None:
            return ToolResult.failure(REACH_ERROR)
        block = meeting_notes.record(title, date.today(), transcript)
        _write(meeting_notes.append_record(existing, block))
    except (OSError, subprocess.SubprocessError) as error:
        return ToolResult.failure(f"That did not work: {filemanage._error_text(error)}")
    saved = meeting_notes.parse(block)[0]
    speech = (
        f"I saved {saved.title} in your meeting notes: "
        f"{_count_word(len(saved.decisions), 'decision')} and "
        f"{_count_word(len(saved.actions), 'action item')}. "
        "The transcript stays on this machine."
    )
    if saved.speakers:
        speech += " " + " and ".join(saved.speakers[:3]) + " spoke."
    return ToolResult.success(speech)


def _meeting_note_recall_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    content = _archive()
    if content is None:
        return ToolResult.failure(REACH_ERROR)
    if not content.strip():
        return ToolResult.success(NO_RECORDINGS)
    items = _found(content, args, today)
    if not items:
        return ToolResult.success("I have no meeting notes for that.")
    return ToolResult.success(meeting_notes.spoken_recall(items, today))


def _meeting_note_actions_handler(args: dict[str, str]) -> ToolResult:
    today = date.today()
    content = _archive()
    if content is None:
        return ToolResult.failure(REACH_ERROR)
    if not content.strip():
        return ToolResult.success(NO_RECORDINGS)
    items = _found(content, args, today)
    if not items:
        return ToolResult.success("I have no meeting notes for that.")
    actions = [action for item in items for action in item.actions]
    if not actions:
        return ToolResult.success(meeting_notes.spoken_actions(items, today))
    parsed = calendar_tools.split_when(args.get("when", "").strip(), today)
    day, time = parsed if parsed else (today, None)
    try:
        calendar = calendar_tools._read()
        for action in actions:
            calendar = calendar_tools.add(calendar, day, time, action)
        calendar_tools._write(calendar)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure("I could not reach your calendar.")
    return ToolResult.success(f"I put {_count_word(len(actions), 'action item')} on your calendar for {_day_phrase(day, today)}.")


def meeting_note_tools() -> list[Tool]:
    return [
        Tool(
            name="meeting_note_from_recording",
            description="Transcribe a recording on this machine into a dated meeting note with its decisions and action items.",
            parameters={"name": "the recording", "folder": "which folder it is in", "title": "what to call the meeting"},
            risk=RiskLevel.REVERSIBLE,
            handler=_meeting_note_from_recording_handler,
            required=("name",),
        ),
        Tool(
            name="meeting_note_recall",
            description="Read back the meeting notes of a day or a word, with their decisions and action items.",
            parameters={"day": "which day, e.g. yesterday or last week", "keyword": "what the meeting was about"},
            risk=RiskLevel.READ_ONLY,
            handler=_meeting_note_recall_handler,
        ),
        Tool(
            name="meeting_note_actions",
            description="Put the action items of a meeting note on the calendar.",
            parameters={
                "name": "which meeting",
                "day": "which day the meeting was on",
                "when": "when to put the items on the calendar",
            },
            risk=RiskLevel.REVERSIBLE,
            handler=_meeting_note_actions_handler,
        ),
    ]
