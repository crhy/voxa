from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

Adw.init()

from voxa.ui.model_selector import AI_STATUS, ModelSelector, ai_status_for  # noqa: E402


def test_status_from_model_list():
    assert ai_status_for(["qwen2.5:0.5b"]) == "online"
    assert ai_status_for([]) == "offline"
    assert ai_status_for(["Strata is not running"]) == "offline"
    assert ai_status_for(["No models available - Ollama"]) == "offline"


def test_indicator_shows_and_hides():
    selector = ModelSelector()
    assert selector.get_status() == ""
    for state in ("starting", "online", "offline"):
        selector.set_status(state)
        assert selector.get_status() == state
        assert selector._status.get_visible() and selector._status.get_text() == AI_STATUS[state]
        assert selector._status.has_css_class(f"voxa-ai-{state}")
    selector.set_status("online")
    assert not selector._status.has_css_class("voxa-ai-offline")
    selector.set_status("nonsense")
    assert not selector._status.get_visible()
