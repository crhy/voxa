"""Compact Ollama model selector (issue #5).

Lives in the bottom bar, not the header. Persists through the callback the
owner wires (MainWindow saves the setting, as before).
"""

from __future__ import annotations

from collections.abc import Callable

from gi.repository import Gtk


class ModelSelector(Gtk.Box):
    """"Local AI:" label plus a model dropdown."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._models: list[str] = []
        self._updating = False
        self._callback: Callable[[str], None] | None = None

        caption = Gtk.Label(label="Local AI:")
        caption.add_css_class("dim-label")
        self.dropdown = Gtk.DropDown()
        self.dropdown.add_css_class("voxa-model-select")
        self.dropdown.set_tooltip_text("Ollama model used for answers")
        self.dropdown.set_visible(False)  # shown once models arrive
        self.dropdown.connect("notify::selected", self._on_selected)
        self.append(caption)
        self.append(self.dropdown)

    def set_models(self, models: list[str], selected: str | None) -> None:
        """Replace the list; no callback fires for this programmatic set."""
        self._models = list(models)
        self.dropdown.set_visible(bool(models))
        if not models:
            return
        index = models.index(selected) if selected in models else 0
        self._updating = True
        self.dropdown.set_model(Gtk.StringList.new(models))
        self.dropdown.set_selected(index)
        self._updating = False

    def connect_changed(self, callback: Callable[[str], None]) -> None:
        self._callback = callback

    def _on_selected(self, _dropdown: Gtk.DropDown, _property: str) -> None:
        if self._updating or not self._models or self._callback is None:
            return
        index = min(self.dropdown.get_selected(), len(self._models) - 1)
        self._callback(self._models[index])
