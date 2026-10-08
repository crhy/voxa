from __future__ import annotations

import logging
import os
import subprocess

from voxa import simulation
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.tools import filemanage

log = logging.getLogger(__name__)

# A few frames are thrown away first: a webcam's first frames are dark while it sets its exposure.
GST_SCRIPT = (
    "import sys, gi\n"
    "gi.require_version('Gst', '1.0')\n"
    "from gi.repository import Gst\n"
    "Gst.init(None)\n"
    "p = Gst.parse_launch(sys.argv[1] + ' num-buffers=20 ! videoconvert ! jpegenc ! multifilesink location=\"' + sys.argv[2] + '\"')\n"
    "p.set_state(Gst.State.PLAYING)\n"
    "m = p.get_bus().timed_pop_filtered(8 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)\n"
    "p.set_state(Gst.State.NULL)\n"
    "sys.exit(0 if m is not None and m.type == Gst.MessageType.EOS else 1)\n"
)
SOURCE = "v4l2src"

# Tried in order on the host; "{path}" is replaced. The last one needs only Python with GStreamer.
GRABBERS = (
    ("ffmpeg", ["ffmpeg", "-loglevel", "error", "-y", "-f", "v4l2", "-i", "/dev/video0", "-frames:v", "1", "{path}"]),
    ("fswebcam", ["fswebcam", "-q", "-r", "1280x720", "--no-banner", "-S", "15", "{path}"]),
    ("python3", ["python3", "-c", GST_SCRIPT, "{source}", "{path}"]),
)

NO_CAMERA = "I could not take a picture: I found no camera I can use."
last_picture = ""


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command[:2])
        return subprocess.CompletedProcess(command, 0, "", "")
    return subprocess.run(
        host_command(command), input=stdin, capture_output=True, text=True, timeout=20, check=check
    )


def picture_name(stamp: str) -> str:
    return f"Photo {stamp}.jpg"


def take_picture(args: dict[str, str] | None = None) -> ToolResult:
    global last_picture
    try:
        folder = filemanage._folder("pictures")
        if folder is None:
            return ToolResult.failure(NO_CAMERA)
        stamp = _run(["date", "+%Y-%m-%d %H-%M-%S"]).stdout.strip()
        path = os.path.join(folder, picture_name(stamp))
        for program, template in GRABBERS:
            if _run(["sh", "-c", 'command -v "$1"', "sh", program], check=False).returncode != 0:
                continue
            command = [part.replace("{path}", path).replace("{source}", SOURCE) for part in template]
            if _run(command, check=False).returncode != 0:
                continue
            if _run(["test", "-s", path], check=False).returncode == 0:
                last_picture = path
                return ToolResult.success("Picture saved to Pictures. Say show me the picture to see it.", detail=path)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure(NO_CAMERA)
    return ToolResult.failure(NO_CAMERA)


def show_picture(args: dict[str, str] | None = None) -> ToolResult:
    if not last_picture:
        return ToolResult.failure("I have not taken a picture yet.")
    try:
        _run(["gio", "open", last_picture], check=False)
    except (OSError, subprocess.SubprocessError):
        return ToolResult.failure("I could not open the picture.")
    return ToolResult.success("Here it is.", detail=last_picture)


def camera_tools() -> list[Tool]:
    return [
        Tool(
            name="take_picture",
            description="Take a picture with the webcam and save it to the Pictures folder.",
            parameters={},
            risk=RiskLevel.REVERSIBLE,
            handler=take_picture,
        ),
        Tool(
            name="show_picture",
            description="Open the picture that was just taken.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=show_picture,
        ),
    ]
