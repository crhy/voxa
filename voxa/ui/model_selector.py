"""Compact "Local AI" picker: a model dropdown plus a backend dropdown."""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

EMPTY_LABEL = "No models available"
BACKENDS = ["llama.cpp", "Ollama"]
BACKEND_VALUES = ["llamacpp", "ollama"]


class ModelSelector(Gtk.Box):
    """Dropdowns over the available local models and the AI backend.

    Programmatic updates (``set_models`` / ``set_backend``) never fire the
    callbacks; only a real user selection does.
    """

    def __init__(
        self,
        on_model_selected: Callable[[str], None] | None = None,
        on_backend_selected: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        label = Gtk.Label(label="Local AI")
        label.add_css_class("voxa-section")

        self._items = Gtk.StringList.new([])
        self._dropdown = Gtk.DropDown(model=self._items)
        self._dropdown.set_sensitive(False)

        self._backend_items = Gtk.StringList.new(BACKENDS)
        self._backend_dropdown = Gtk.DropDown(model=self._backend_items)

        self.on_model_selected = on_model_selected
        self.on_backend_selected = on_backend_selected
        self._models: list[str] = []
        self._updating = False
        self._backend_updating = False
        self._dropdown.connect("notify::selected", self._on_selected)
        self._backend_dropdown.connect("notify::selected", self._on_backend_selected)

        self.append(label)
        self.append(self._dropdown)
        self.append(self._backend_dropdown)

    def set_models(self, models: list[str], selected: str = "") -> None:
        """Fill the dropdown from a model list without triggering the callback."""
        self._updating = True
        try:
            self._models = list(models)
            if not self._models:
                self._items.splice(0, self._items.get_n_items(), [EMPTY_LABEL])
            else:
                self._items.splice(0, self._items.get_n_items(), self._models)
                if selected in self._models:
                    self._dropdown.set_selected(self._models.index(selected))
            self._dropdown.set_sensitive(bool(self._models))
        finally:
            self._updating = False

    def get_selected(self) -> str:
        if not self._models:
            return ""
        item = self._dropdown.get_selected_item()
        return item.get_string() if item is not None else ""

    def set_backend(self, backend: str) -> None:
        """Select the backend ("llamacpp" or "ollama") without triggering the callback."""
        index = 1 if backend == "ollama" else 0
        self._backend_updating = True
        try:
            self._backend_dropdown.set_selected(index)
        finally:
            self._backend_updating = False

    def get_backend(self) -> str:
        item = self._backend_dropdown.get_selected_item()
        text = item.get_string() if item is not None else ""
        return "ollama" if text == "Ollama" else "llamacpp"

    def _on_selected(self, *_args) -> None:
        if self._updating or not self._models or self.on_model_selected is None:
            return
        self.on_model_selected(self.get_selected())

    def _on_backend_selected(self, *_args) -> None:
        if self._backend_updating or self.on_backend_selected is None:
            return
        index = self._backend_dropdown.get_selected()
        backend = BACKEND_VALUES[index] if 0 <= index < len(BACKEND_VALUES) else "llamacpp"
        self.on_backend_selected(backend)
