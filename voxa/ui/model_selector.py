"""Compact "Local AI" model picker with a backend label."""

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

EMPTY_LABEL = "No models available"


class ModelSelector(Gtk.Box):
    """Dropdown over the available local models; programmatic updates never fire the callback."""

    def __init__(self, on_model_selected: Callable[[str], None] | None = None) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        label = Gtk.Label(label="Local AI")
        label.add_css_class("voxa-section")

        self._items = Gtk.StringList.new([])
        self._dropdown = Gtk.DropDown(model=self._items)
        self._dropdown.set_sensitive(False)

        self._backend_label = Gtk.Label(label="")
        self._backend_label.add_css_class("voxa-task-detail")

        self.on_model_selected = on_model_selected
        self._models: list[str] = []
        self._updating = False
        self._dropdown.connect("notify::selected", self._on_selected)

        self.append(label)
        self.append(self._dropdown)
        self.append(self._backend_label)

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

    def set_backend_label(self, text: str) -> None:
        self._backend_label.set_text(text)

    def _on_selected(self, *_args) -> None:
        if self._updating or not self._models or self.on_model_selected is None:
            return
        self.on_model_selected(self.get_selected())
