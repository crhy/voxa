"""Follow the Spaced desktop theme, live.

Spaced's theme switcher (``/usr/local/bin/spaced-switch-theme``) writes two
files under ``~/.config/spaced``:

* ``libadwaita.css``  -- libadwaita ``@define-color`` overrides for the active
  desktop theme (navy for the Win11/Win7 dark themes, grey for Spaced Dark/Mint,
  light values for the light themes).
* ``theme.json``      -- a small record with a ``"dark"`` flag.

Voxa loads both so its palette matches the desktop the way Spaced RGB and
Spaced About do.  When the files are absent (a machine that is not Spaced
Linux) Voxa leaves libadwaita's normal behaviour alone: the colour scheme
stays ``DEFAULT`` and follows the portal.

The provider is installed at ``GTK_STYLE_PROVIDER_PRIORITY_APPLICATION`` on the
default display, so it sits above the desktop theme's own base stylesheet but
below Voxa's interface CSS (installed at the same priority later, which wins
by load order) and below user CSS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gio", "2.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gio, GLib, Gdk, Gtk  # noqa: E402

# Debounce window for the live file watcher. The theme switcher rewrites the
# css and json in quick succession; one reload per burst is enough.
DEBOUNCE_MS = 300

CSS_FILENAME = "libadwaita.css"
THEME_FILENAME = "theme.json"


def spaced_config_dir() -> Path:
    """Directory holding Spaced's libadwaita.css and theme.json.

    Honours ``XDG_CONFIG_HOME``; inside the Flatpak the host's file lands at
    the same path once the manifest grants ``xdg-config/spaced``.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "spaced"


def read_dark(theme_json: Path) -> bool | None:
    """Return the theme.json ``"dark"`` flag, or None when unknown/missing."""
    try:
        data = json.loads(theme_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("dark")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "dark"}
    if isinstance(value, int):
        return bool(value)
    return None


def read_css(css_path: Path) -> str | None:
    """Return the libadwaita.css text, or None when missing/unreadable."""
    try:
        return css_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


class SpacedPalette:
    """Loads the Spaced palette into GTK and keeps it in sync with the files."""

    def __init__(
        self,
        config_dir: Path | None = None,
        style_manager: "Adw.StyleManager | None" = None,
        display: "Gdk.Display | None" = None,
    ) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else spaced_config_dir()
        self.css_path = self.config_dir / CSS_FILENAME
        self.theme_path = self.config_dir / THEME_FILENAME
        self.style_manager = style_manager or Adw.StyleManager.get_default()
        self._display = display
        self.provider: Gtk.CssProvider | None = None
        self.monitor = None
        # Last "dark" flag read from theme.json (None = unknown/missing).
        self.theme_dark: bool | None = None
        self.reloads = 0
        self._changed = False
        self._timer = None

    @property
    def display(self) -> "Gdk.Display | None":
        if self._display is None:
            self._display = Gdk.Display.get_default()
        return self._display

    def install(self) -> bool:
        """Apply the palette once and start watching the files.

        Returns True when at least one Spaced file was found and applied, False
        when neither exists (non-Spaced machine: leave libadwaita alone).
        """
        css = read_css(self.css_path)
        dark = read_dark(self.theme_path)
        if css is None and dark is None:
            return False
        self.theme_dark = dark
        self._apply(css, dark)
        self._watch()
        return True

    def _apply(self, css: str | None, dark: bool | None) -> None:
        display = self.display
        if self.provider is None:
            self.provider = Gtk.CssProvider()
            if display is not None:
                Gtk.StyleContext.add_provider_for_display(
                    display,
                    self.provider,
                    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
                )
        if css is not None:
            self.provider.load_from_string(css)
        if dark is True:
            self.style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)
        elif dark is False:
            self.style_manager.set_color_scheme(Adw.ColorScheme.FORCE_LIGHT)
        # dark is None: leave the scheme as it is (DEFAULT follows the portal).

    def _watch(self) -> None:
        """Watch libadwaita.css; reload the palette when the theme changes."""
        if self.monitor is not None:
            return
        try:
            # monitor_type 1 == CONTENTS (GFileMonitorType). The monitor stays
            # alive as long as we keep this reference, so Voxa reloads the
            # palette whenever the theme switcher rewrites the css.
            handle = Gio.File.new_for_path(str(self.css_path))
            self.monitor = handle.monitor(1)
            self.monitor.connect("changed", lambda *_: self._on_change())
        except Exception:
            self.monitor = None

    def _on_change(self) -> None:
        """File changed: debounce, then a single reload."""
        self._changed = True
        if self._timer is None:
            self._timer = GLib.timeout_add(DEBOUNCE_MS, self._flush)

    def _flush(self, *_args) -> bool:
        """Timer callback: perform the debounced reload. Returns False (one-shot)."""
        self._timer = None
        if self._changed:
            self._changed = False
            self.reload()
        return False

    def reload(self) -> None:
        """Re-read both files and re-apply them."""
        self.reloads += 1
        css = read_css(self.css_path)
        dark = read_dark(self.theme_path)
        self.theme_dark = dark
        self._apply(css, dark)


def install_spaced_palette(
    display: "Gdk.Display | None" = None,
    style_manager: "Adw.StyleManager | None" = None,
    config_dir: Path | None = None,
) -> "SpacedPalette | None":
    """Install the Spaced palette into the running app.

    Returns the live :class:`SpacedPalette` when the Spaced files are present,
    or ``None`` when they are not (Voxa keeps libadwaita's default behaviour).
    """
    palette = SpacedPalette(config_dir=config_dir, style_manager=style_manager, display=display)
    if palette.install():
        return palette
    return None
