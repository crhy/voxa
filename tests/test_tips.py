"""Tests for the contextual tips module and the tips panel widget."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from voxa.tips import tips_for  # noqa: E402
from voxa.ui.tips_panel import MAX_TIPS, TipsPanel  # noqa: E402


def _children(panel: Gtk.Box) -> list[Gtk.Widget]:
    kids = []
    child = panel.get_first_child()
    while child is not None:
        kids.append(child)
        child = child.get_next_sibling()
    return kids


def test_ready_tips_are_the_three_lines():
    assert tips_for("READY", {}) == [
        "Say “Voxa”, then your request",
        "Try: “Voxa, play some jazz”",
        "Try: “Voxa, get me the cheapest ticket to Hawaii”",
    ]


def test_ready_rotation_by_tick():
    tick_1 = tips_for("READY", {"tick": 1})
    assert tick_1 == [
        "Try: “Voxa, play some jazz”",
        "Try: “Voxa, get me the cheapest ticket to Hawaii”",
        "Say “Voxa”, then your request",
    ]
    assert tips_for("READY", {"tick": 3}) == tips_for("READY", {"tick": 0})


def test_offline_tips():
    tips = tips_for("OFFLINE", {})
    assert tips[0] == "Press ACTIVE to start"
    assert len(tips) == 2


def test_listening_tips():
    assert tips_for("LISTENING", {}) == [
        "I am listening — just say it",
        "Say “never mind” to cancel",
    ]


def test_speaking_tips():
    assert tips_for("SPEAKING", {}) == [
        "Talk over me to interrupt",
        "Say “Voxa, pause” to make me wait",
    ]


def test_paused_tips():
    assert tips_for("PAUSED", {}) == ["Say “Voxa” to continue"]


def test_dictating_tips():
    assert tips_for("DICTATING", {}) == [
        "Say “stop dictation” to exit dictation mode",
        "Say “new line”, “comma”, “period” for punctuation",
    ]


def test_thinking_tips():
    tips = tips_for("THINKING", {})
    assert len(tips) == 2
    assert tips[0] == "One moment — thinking it through"


def test_media_playing_adds_music_tip():
    tips = tips_for("DICTATING", {"media_playing": True})
    assert "Say “stop music” or “pause the music”" in tips
    assert len(tips) == 3


def test_no_models_adds_installer_tip():
    tips = tips_for("OFFLINE", {"no_models": True})
    assert "Open Preferences → Local AI to install a model" in tips


def test_live_face_mode_adds_nothing():
    assert tips_for("READY", {"face_mode": "live"}) == tips_for("READY", {})


def test_at_most_three_lines():
    tips = tips_for("READY", {"media_playing": True, "no_models": True, "tick": 2})
    assert len(tips) <= MAX_TIPS
    assert "Say “stop music” or “pause the music”" in tips
    assert "Open Preferences → Local AI to install a model" in tips


def test_unknown_state_has_no_tips():
    assert tips_for("BOGUS", {}) == []
    assert tips_for("BOGUS", {"media_playing": True}) == [
        "Say “stop music” or “pause the music”"
    ]


def test_panel_shows_one_label_per_tip():
    panel = TipsPanel()
    panel.set_tips(["one", "two"])
    assert len(_children(panel)) == 2
    assert panel.get_visible()


def test_panel_hidden_when_empty():
    panel = TipsPanel()
    panel.set_tips([])
    assert not panel.get_visible()
    assert len(_children(panel)) == 0


def test_panel_unchanged_list_is_a_noop():
    panel = TipsPanel()
    panel.set_tips(["one", "two"])
    first = _children(panel)
    panel.set_tips(["one", "two"])
    assert _children(panel) == first


def test_panel_caps_at_max_tips():
    panel = TipsPanel()
    panel.set_tips(["a", "b", "c", "d"])
    assert len(_children(panel)) == MAX_TIPS
