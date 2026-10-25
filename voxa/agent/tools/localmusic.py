from __future__ import annotations

from voxa import apps
from voxa.agent.localmusic import pick_tracks
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult


def _music_files() -> tuple[str, list[str]]:
    from voxa.agent.tools.windows import _run

    home = _run(["sh", "-c", 'printf %s "$HOME"'], check=False).stdout.strip()
    root = _run(["xdg-user-dir", "MUSIC"], check=False).stdout.strip()
    if not root or root.rstrip("/") == home.rstrip("/"):
        root = f"{home}/Music"
    lines = _run(["find", root, "-maxdepth", "4", "-type", "f"], check=False).stdout.splitlines()
    return (root, lines)


def play_on_app(args: dict[str, str]) -> ToolResult:
    app = apps.match_app(apps.clean_spoken_app_name(args["app"]), apps.list_apps())
    if app is None:
        return ToolResult.failure(f"I couldn't find an app called {args['app']}.")
    root, files = _music_files()
    tracks = pick_tracks(args["query"], root, files)
    if not tracks:
        return ToolResult.failure(f"I couldn't find {args['query']} in your Music folder.")
    if apps.launch(app, tracks) is False:
        return ToolResult.failure(f"I couldn't open {app.name}.")
    return ToolResult.success(f"Playing {args['query']} in {app.name}.", detail=f"{len(tracks)} files")


def local_music_tools() -> list[Tool]:
    return [
        Tool(
            name="play_on_app",
            description="Play music from the user's Music folder in a named desktop app.",
            parameters={"query": "what to play", "app": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=play_on_app,
            required=("query", "app"),
        )
    ]
