"""Shared CSS for the new-interface shell (issue #6 target tree)."""

from __future__ import annotations

from gi.repository import Gdk, Gtk

SHELL_CSS = b"""
.voxa-task-panel {
  background-color: rgba(0, 0, 0, 0.28);
  border-radius: 12px;
  padding: 12px;
}
.voxa-panel-heading { font-weight: bold; font-size: 0.85em; }
.voxa-task-row { padding: 2px 0; }
.voxa-choice {
  background-color: rgba(0, 0, 0, 0.25);
  border-radius: 8px;
  padding: 8px;
}
.voxa-model-select { min-width: 170px; }
.voxa-mode-button { font-weight: bold; }
.voxa-active-on { background-image: none; background-color: #2e7d32; color: white; }
.voxa-offline-on { background-image: none; background-color: #c62828; color: white; }
"""


def install_css() -> None:
    """Register the shell stylesheet (no-op without a display)."""
    display = Gdk.Display.get_default()
    if display is None:
        return
    provider = Gtk.CssProvider()
    provider.load_from_data(SHELL_CSS)
    Gtk.StyleContext.add_provider_for_display(
        display,
        provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )
