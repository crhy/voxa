from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.model_selector import MAX_BUTTON_CHARS, ModelSelector  # noqa: E402


def test_programmatic_update_does_not_fire_callback() -> None:
    fired: list[str] = []
    selector = ModelSelector(on_model_selected=fired.append)
    selector.set_models(["qwen3.8", "llama3"], selected="llama3")
    assert selector.get_selected() == "llama3"
    assert fired == []
    selector.set_models(["qwen3.8"])
    assert fired == []


def test_user_selection_fires_callback() -> None:
    fired: list[str] = []
    selector = ModelSelector(on_model_selected=fired.append)
    selector.set_models(["qwen3.8", "llama3"])
    # set_models() leaves the first item selected, so a user picking the other
    # one is the real change that must fire the callback.
    selector._dropdown.set_selected(1)
    assert fired == ["llama3"]


def test_empty_model_list() -> None:
    fired: list[str] = []
    selector = ModelSelector(on_model_selected=fired.append)
    selector.set_models([])
    assert not selector._dropdown.get_sensitive()
    assert selector._items.get_n_items() == 1
    assert selector._items.get_item(0).get_string() == "No models available"
    assert selector.get_selected() == ""
    selector._dropdown.set_selected(0)
    assert fired == []


def test_backend_dropdown_exists_and_defaults_to_llamacpp() -> None:
    selector = ModelSelector()
    backend = selector._backend_dropdown
    assert isinstance(backend, Gtk.DropDown)
    items = [backend.get_model().get_string(i) for i in range(backend.get_model().get_n_items())]
    assert items == ["llama.cpp", "Ollama"]
    assert backend.get_selected() == 0
    assert selector.get_backend() == "llamacpp"


def test_set_backend_does_not_fire_callback() -> None:
    fired: list[str] = []
    selector = ModelSelector(on_backend_selected=fired.append)
    selector.set_backend("ollama")
    assert selector.get_backend() == "ollama"
    assert fired == []
    selector.set_backend("llamacpp")
    assert fired == []


def test_user_backend_choice_fires_callback_with_backend_id() -> None:
    fired: list[str] = []
    selector = ModelSelector(on_backend_selected=fired.append)
    selector._backend_dropdown.set_selected(1)
    assert fired == ["ollama"]
    selector._backend_dropdown.set_selected(0)
    assert fired == ["ollama", "llamacpp"]


def test_backend_dropdown_is_a_real_control_not_a_label() -> None:
    selector = ModelSelector()
    backend = selector._backend_dropdown
    assert backend.get_sensitive()
    assert not isinstance(backend, Gtk.Label)


def test_long_model_name_get_selected_returns_full_name() -> None:
    long_name = "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S" * 2
    selector = ModelSelector()
    selector.set_models([long_name])
    assert selector.get_selected() == long_name


def test_dropdown_tooltip_is_full_name_after_set_models() -> None:
    long_name = "hf.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S" * 2
    selector = ModelSelector()
    selector.set_models([long_name])
    assert selector._dropdown.get_tooltip_text() == long_name


def test_button_and_list_factories_are_set_and_different() -> None:
    selector = ModelSelector()
    button_factory = selector._dropdown.get_factory()
    list_factory = selector._dropdown.get_list_factory()
    assert button_factory is not None
    assert list_factory is not None
    assert button_factory is not list_factory


def test_max_button_chars_constant() -> None:
    assert MAX_BUTTON_CHARS == 24
