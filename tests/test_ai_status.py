from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

Adw.init()

from voxa.ui.model_selector import AI_STATUS, ModelSelector, ai_status_for, ai_status_for_backend  # noqa: E402
from voxa.server import SERVER_FAILED, SERVER_IDLE, SERVER_READY, SERVER_STARTING, SERVER_UNAVAILABLE  # noqa: E402


def test_status_from_model_list():
    assert ai_status_for(["qwen2.5:0.5b"]) == "online"
    assert ai_status_for([]) == "offline"
    assert ai_status_for(["Strata is not running"]) == "offline"
    assert ai_status_for(["No models available - Ollama"]) == "offline"


def test_ollama_dot_follows_the_server_not_the_disk_list():
    # Installed models are now listed from disk, so a non-empty Ollama list no
    # longer proves the server is answering: its health decides, as for llama.cpp.
    disk = ["qwen2.5:0.5b", "llama3:latest"]
    assert ai_status_for_backend("ollama", disk, SERVER_READY) == "online"
    assert ai_status_for_backend("ollama", disk, SERVER_STARTING) == "starting"
    for down in (SERVER_FAILED, SERVER_UNAVAILABLE, SERVER_IDLE, ""):
        assert ai_status_for_backend("ollama", disk, down) == "offline"


def test_llamacpp_dot_still_follows_the_engine_health():
    gguf = "/home/rhy/.local/share/voxa/models/voxa-small.gguf"
    assert ai_status_for_backend("llamacpp", [gguf], SERVER_READY) == "online"
    assert ai_status_for_backend("llamacpp", [gguf], SERVER_STARTING) == "starting"
    assert ai_status_for_backend("llamacpp", [gguf], SERVER_FAILED) == "offline"


def test_a_backend_voxa_does_not_manage_keeps_the_list_rule():
    assert ai_status_for_backend("strata", ["qwen3.8"], SERVER_READY) == "online"
    assert ai_status_for_backend("strata", ["Strata is not running"], SERVER_READY) == "offline"


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
