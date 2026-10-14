from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime

from voxa import simulation
from voxa.agent.databackup import (
    BACKUP_FOLDER,
    backup_dir_name,
    parse_listing,
    spoken_done,
    spoken_last,
)
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

REACH_ERROR = "I could not reach your backups."


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


def _backup_now_handler(args: dict[str, str]) -> ToolResult:
    try:
        docs = filemanage._folder("documents")
        if docs is None:
            return ToolResult.failure("I cannot reach your documents folder.")
        name = backup_dir_name(datetime.now())
        dest = os.path.join(docs, BACKUP_FOLDER, name)
        done = _run(
            [
                "sh",
                "-c",
                'mkdir -p "$1" && cp -p "$2"/Voxa\\ *.md "$1"/ 2>/dev/null; ls "$1" | wc -l',
                "sh",
                dest,
                docs,
            ],
            check=False,
        )
        count = int(done.stdout.strip()) if done.stdout.strip().isdigit() else 0
        if count == 0:
            _run(["rmdir", "--", dest], check=False)
            return ToolResult.failure("There is nothing to back up yet.")
        return ToolResult.success(spoken_done(count, name))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError) as error:
        return ToolResult.failure(REACH_ERROR, detail=str(error))


def _backup_last_handler(args: dict[str, str]) -> ToolResult:
    try:
        docs = filemanage._folder("documents")
        if docs is None:
            return ToolResult.failure("I cannot reach your documents folder.")
        backups = os.path.join(docs, BACKUP_FOLDER)
        done = _run(["ls", "-1", backups], check=False)
        names = parse_listing(done.stdout)
        return ToolResult.success(spoken_last(names, datetime.now()))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        return ToolResult.failure(REACH_ERROR, detail=str(error))


def databackup_tools() -> list[Tool]:
    return [
        Tool(
            name="backup_now",
            description="Back up your Voxa data files into a dated backup folder.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=_backup_now_handler,
            required=(),
        ),
        Tool(
            name="backup_last",
            description="Report when your last data backup ran.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=_backup_last_handler,
            required=(),
        ),
    ]
