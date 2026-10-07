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
/* Preferences: libadwaita's defaults leave the cards almost the same shade as
   the dialog and the subtitles small and faint, which reads as low contrast on
   the dark theme. Lift the cards, give the rows room, and firm up the text. */
.voxa-preferences preferencesgroup > box > box > box > label.title,
.voxa-preferences preferencesgroup .heading {
    font-size: 15px;
    font-weight: 700;
    opacity: 1;
}

.voxa-preferences preferencesgroup .description,
.voxa-preferences preferencesgroup label.dim-label {
    opacity: 0.82;
}

.voxa-preferences list.boxed-list {
    background: alpha(currentColor, 0.075);
    border: 1px solid alpha(currentColor, 0.14);
    box-shadow: none;
}

.voxa-preferences list.boxed-list > row {
    min-height: 58px;
    border-color: alpha(currentColor, 0.12);
}

.voxa-preferences row .title {
    font-size: 14px;
    font-weight: 600;
}

.voxa-preferences row .subtitle {
    font-size: 12.5px;
    font-weight: 400;
    opacity: 0.85;
    margin-top: 2px;
}

.voxa-preferences preferencespage > scrolledwindow > viewport > clamp > box {
    margin-top: 12px;
    margin-bottom: 24px;
}

.voxa-preferences preferencesgroup {
    margin-bottom: 10px;
}

.voxa-brand {
    font-weight: 700;
    font-size: 15px;
}

.voxa-active,
.voxa-offline,
.voxa-pause {
    min-width: 84px;
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

.voxa-pause {
    background: #e5a50a;
    color: white;
}

.voxa-active.selected,
.voxa-offline.selected,
.voxa-pause.selected {
    outline: 1px solid alpha(currentColor, 0.45);
    outline-offset: 2px;
}

.voxa-active.dimmed,
.voxa-offline.dimmed,
.voxa-pause.dimmed {
    opacity: 0.45;
}

.voxa-task-panel {
    /* GTK CSS has no max-width: the width cap is applied in TaskPanel. */
    min-width: 250px;
    padding: 14px;
    border-radius: 16px;
    background: alpha(@window_bg_color, 0.88);
}

.voxa-section,
.voxa-control-caption {
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

.voxa-tips {
    /* GTK CSS has no max-width: the width cap is applied in TipsPanel. */
    font-size: 13px;
    opacity: 0.6;
}

.voxa-exchange-answer {
    font-size: 13px;
}

.voxa-notice {
    margin: 12px 24px;
    padding: 10px 18px;
    border-radius: 12px;
    background: alpha(@accent_bg_color, 0.92);
    color: @accent_fg_color;
    font-weight: 600;
}

.voxa-choice {
    padding: 8px 14px;
    border-radius: 10px;
}

.voxa-state {
    font-size: 30px;
    font-weight: 800;
    opacity: 1;
    margin-top: 16px;
    letter-spacing: 0.01em;
}

.voxa-state-hint {
    font-size: 15px;
    opacity: 0.75;
    margin-top: 4px;
}

.voxa-assistant-view.ready .voxa-state,
.voxa-assistant-view.listening .voxa-state,
.voxa-assistant-view.dictating .voxa-state {
    color: #3fb950;
}

.voxa-assistant-view.thinking .voxa-state,
.voxa-assistant-view.working .voxa-state {
    color: #d29922;
}

.voxa-assistant-view.speaking .voxa-state {
    color: #58a6ff;
}

.voxa-assistant-view.error .voxa-state {
    color: #f85149;
}

.voxa-assistant-view.offline .voxa-state {
    opacity: 0.6;
}

.voxa-assistant-view.paused .voxa-state {
    color: #e5a50a;
}

.voxa-avatar {
    opacity: 0.78;
    transition: opacity 0.18s ease;
    animation: none;
}

@keyframes voxa-avatar-idle {
    0%, 100% {
        opacity: 0.72;
        transform: scale(1.000);
    }
    50% {
        opacity: 0.82;
        transform: scale(1.020);
    }
}

@keyframes voxa-avatar-listening {
    0%, 100% {
        opacity: 0.78;
        transform: scale(1.010);
    }
    50% {
        opacity: 0.94;
        transform: scale(1.035);
    }
}

@keyframes voxa-avatar-thinking {
    0%, 100% {
        opacity: 0.70;
        transform: scale(1.010) rotate(0.2deg);
    }
    50% {
        opacity: 0.84;
        transform: scale(1.025) rotate(-0.2deg);
    }
}

@keyframes voxa-avatar-speaking {
    0%, 100% {
        opacity: 0.78;
        transform: scale(1.020);
    }
    50% {
        opacity: 0.92;
        transform: scale(1.045);
    }
}

@keyframes voxa-avatar-error {
    0%, 100% {
        opacity: 0.72;
        transform: scale(1.000);
    }
    50% {
        opacity: 0.96;
        transform: scale(1.018);
    }
}

.voxa-assistant-view.ready .voxa-avatar {
    animation: voxa-avatar-idle 5.0s ease-in-out infinite;
}

.voxa-assistant-view.listening .voxa-avatar,
.voxa-assistant-view.dictating .voxa-avatar {
    animation: voxa-avatar-listening 1.5s ease-in-out infinite;
}

.voxa-assistant-view.thinking .voxa-avatar {
    animation: voxa-avatar-thinking 7.5s ease-in-out infinite;
}

.voxa-assistant-view.speaking .voxa-avatar {
    animation: voxa-avatar-speaking 2.4s ease-in-out infinite;
    opacity: 0.92;
}

.voxa-assistant-view.error .voxa-avatar {
    animation: voxa-avatar-error 1.2s ease-in-out 3;
    opacity: 0.96;
}

.voxa-assistant-view.speaking .voxa-avatar.audio-low {
    animation-duration: 3.0s;
    opacity: 0.82;
}

.voxa-assistant-view.speaking .voxa-avatar.audio-medium {
    animation-duration: 1.7s;
    opacity: 0.88;
}

.voxa-assistant-view.speaking .voxa-avatar.audio-high {
    animation-duration: 1.2s;
    opacity: 0.96;
}

/* Portrait picker: a round photo in the header and a grid of cells in the popover. */
.voxa-portrait-button {
    border-radius: 999px;
}

.voxa-portrait-cell {
    padding: 4px;
    border-radius: 10px;
}

.voxa-portrait-cell.selected-character {
    outline: 2px solid @accent_color;
    outline-offset: 2px;
}

/* Focus pop-up: a circular talking-head cut-out with a status caption, shown
   in the top-right only while the window is not focused. */
.voxa-focus-popup {
    background: alpha(currentColor, 0.06);
    border: 1px solid alpha(currentColor, 0.12);
    border-radius: 12px;
    padding: 8px;
}

.voxa-focus-circle {
    border-radius: 999px;
    overflow: hidden;
}

.voxa-focus-caption {
    font-size: 11px;
    opacity: 0.85;
}

window.voxa-focus-window {
    background: transparent;
    box-shadow: none;
}

.voxa-focus-face {
    border-radius: 999px;
}

.voxa-focus-window-caption {
    background: rgba(20, 20, 24, 0.82);
    color: #ffffff;
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 600;
}

.caption {
    font-size: 11px;
    opacity: 0.8;
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
