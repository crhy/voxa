from __future__ import annotations

import os

HOST_THEME_ENV = "VOXA_HOST_GTK_THEME"


def release_forced_theme(environ=os.environ) -> str:
    """Move a forced GTK_THEME out of the way before GTK is initialised.

    A set GTK_THEME makes libadwaita drop its own stylesheet. If GTK_THEME is
    set and non-empty, its value is copied to HOST_THEME_ENV, GTK_THEME is
    deleted, and the value is returned. Otherwise nothing changes and "" is
    returned.
    """
    theme = environ.get("GTK_THEME", "")
    if not theme:
        return ""
    environ[HOST_THEME_ENV] = theme
    del environ["GTK_THEME"]
    return theme


def host_theme_is_dark(environ=os.environ) -> bool:
    """True when the released host theme looks dark."""
    theme = environ.get(HOST_THEME_ENV, "")
    if "dark" in theme.casefold():
        return True
    return theme.casefold().endswith(":dark")


BUBBLE_BG_DARK = "#242426"
BUBBLE_BG_LIGHT = "#f9f9fa"


def window_bg_color(dark: bool) -> str:
    """Return the application window background for the current theme."""
    return "#1d1d1e" if dark else "#eaebeb"


def initial_size(saved: tuple[int, int] | None) -> tuple[int, int]:
    """Return the saved window size, or the smallest usable start size."""
    if saved is None:
        return (1, 1)
    return saved
