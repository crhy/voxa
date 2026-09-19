"""Approved Voxa interface CSS, installed once per process.

The stylesheet extends the mockup CSS (issue #3) in the same restrained
spirit: task rows, section headings, choice buttons and the
ACTIVE/OFFLINE selection states. Colors are anchored to libadwaita's
theme variables so the panel follows the user's light/dark preference.
"""

from __future__ import annotations

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

VOXA_CSS = """
.voxa-brand {
    font-weight: 700;
    font-size: 15px;
}

.voxa-active,
.voxa-offline {
    min-width: 118px;
    min-height: 48px;
    border-radius: 12px;
    font-weight: 700;
}

.voxa-active {
    background: #238636;
    color: white;
}

.voxa-offline {
    background: #b3261e;
    color: white;
}

.voxa-active.selected,
.voxa-offline.selected {
    outline: 1px solid alpha(currentColor, 0.45);
    outline-offset: 2px;
}

.voxa-active.dimmed,
.voxa-offline.dimmed {
    opacity: 0.45;
}

.voxa-task-panel {
    /* GTK CSS has no max-width: the width cap is applied in TaskPanel. */
    min-width: 250px;
    padding: 14px;
    border-radius: 16px;
    background: alpha(@window_bg_color, 0.88);
}

.voxa-section {
    font-size: 11px;
    font-weight: 600;
    opacity: 0.55;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}

.voxa-task-title {
    font-weight: 700;
    font-size: 14px;
}

.voxa-task-detail {
    font-size: 12px;
    opacity: 0.62;
}

.voxa-task-detail.error {
    color: #d2645f;
    opacity: 0.9;
}

.voxa-choice {
    padding: 8px 14px;
    border-radius: 10px;
}

.voxa-state {
    font-size: 14px;
    opacity: 0.72;
}

.voxa-assistant-view.listening .voxa-state,
.voxa-assistant-view.thinking .voxa-state,
.voxa-assistant-view.speaking .voxa-state {
    opacity: 1;
}
"""

_provider: Gtk.CssProvider | None = None


def install_styles(display: Gdk.Display | None = None) -> Gtk.CssProvider:
    """Load the Voxa stylesheet into the display, once per process."""
    global _provider
    if display is None:
        display = Gdk.Display.get_default()
    if _provider is not None:
        return _provider
    provider = Gtk.CssProvider()
    provider.load_from_string(VOXA_CSS)
    if display is not None:
        Gtk.StyleContext.add_provider_for_display(
            display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
    _provider = provider
    return provider
