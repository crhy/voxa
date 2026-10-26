from __future__ import annotations

from voxa import apps
from voxa.agent.localmusic import pick_tracks
from voxa.agent.player import AudaciousPlayer, VlcPlayer, active_player, music_player
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


def play_local_music(query: str) -> ToolResult | None:
    """Play a matching Music-folder track, or let the caller try online media."""
    root, files = _music_files()
    tracks = pick_tracks(query, root, files)
    if not tracks:
        return None
    player = active_player() or music_player()
    player.play_files(tracks)
    return ToolResult.success(
        f"Playing {query} from your Music folder.", detail=f"{len(tracks)} files"
    )


def resume_app(args: dict[str, str]) -> ToolResult:
    """Resume the playlist already loaded in a named media player."""
    spoken = apps.clean_spoken_app_name(args["app"])
    key = spoken.casefold()
    player = None
    display_name = spoken
    if "audacious" in key:
        player = AudaciousPlayer()
        display_name = "Audacious"
    elif key == "vlc" or "vlc media player" in key:
        player = VlcPlayer()
        display_name = "VLC"
    if player is not None and player.status() in ("Playing", "Paused", "Stopped"):
        player.resume()
        return ToolResult.success(f"Playing the loaded playlist in {display_name}.")

    app = apps.match_app(spoken, apps.list_apps())
    if app is None:
        return ToolResult.failure(f"I couldn't find an app called {args['app']}.")
    if apps.launch(app) is False:
        return ToolResult.failure(f"I couldn't open {app.name}.")
    return ToolResult.success(f"Opened {app.name}. Its loaded playlist is ready.")


def local_music_tools() -> list[Tool]:
    return [
        Tool(
            name="play_on_app",
            description="Play music from the user's Music folder in a named desktop app.",
            parameters={"query": "what to play", "app": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=play_on_app,
            required=("query", "app"),
        ),
        Tool(
            name="resume_app",
            description="Resume the playlist already loaded in a named music player.",
            parameters={"app": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=resume_app,
            required=("app",),
        ),
    ]
