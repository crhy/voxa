from __future__ import annotations

import logging

from voxa import apps
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult

log = logging.getLogger("voxa.agent.tools.applications")


def open_app(args: dict[str, str]) -> ToolResult:
    # The menu is read on every call (a few milliseconds) rather than cached for the
    # life of the process, so a newly installed app is found straight away.
    name = args["name"]
    app = apps.match_app(name, apps.list_apps())
    if app is None:
        return ToolResult.failure(f"I couldn't find an app called {name}.")
    apps.launch(app)
    return ToolResult.success(f"Opening {app.name}.", detail=app.path)


def application_tools() -> list[Tool]:
    return [
        Tool(
            name="open_app",
            description="Launch a desktop application by name.",
            parameters={"name": "the application name"},
            risk=RiskLevel.REVERSIBLE,
            handler=open_app,
            required=("name",),
        )
    ]
