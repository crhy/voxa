"""Station list (radio stations, live TV channels): its own small window, one clean row each."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Pango", "1.0")
from gi.repository import Gdk, Gtk, Pango  # noqa: E402

TITLE = "Radio stations"
HINT = "Say a number or a name, or click a station."
WIDTH = 440
ROW_HEIGHT = 58
MAX_HEIGHT = 560


def station_rows(stations: Sequence) -> list[tuple[str, str, str]]:
    """Pure helper: (number, name, genre and description) for each station, in order."""
    rows = []
    for index, station in enumerate(stations, 1):
        genre = ", ".join(str(getattr(station, "genre", "")).split()[:3])
        about = str(getattr(station, "description", "")).strip()
        rows.append((str(index), station.name, " — ".join(part for part in (genre, about) if part)))
    return rows


class RadioPicker(Gtk.Window):
    """A pop-up listing the stations; choosing one calls ``on_station(name)`` and closes it."""

    def __init__(self, parent: Gtk.Window | None, on_station: Callable[[str], None]) -> None:
        super().__init__(title=TITLE)
        self.on_station = on_station
        if parent is not None:
            self.set_transient_for(parent)
        self.set_modal(False)
        self.set_hide_on_close(True)
        self.add_css_class("voxa-radio-picker")

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        heading = Gtk.Label(label=TITLE, xalign=0)
        heading.add_css_class("voxa-radio-heading")
        hint = Gtk.Label(label=HINT, xalign=0)
        hint.add_css_class("voxa-radio-hint")
        self._heading, self._hint = heading, hint
        self._list = Gtk.ListBox()
        self._list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._list.add_css_class("voxa-radio-list")
        self._list.connect("row-activated", self._on_row)
        self._scroller = Gtk.ScrolledWindow()
        self._scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._scroller.set_child(self._list)
        self._scroller.set_vexpand(True)
        box.append(heading)
        box.append(hint)
        box.append(self._scroller)
        self.set_child(box)

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)

    def show_stations(self, stations: Sequence, title: str = TITLE, hint: str = HINT) -> None:
        """Fill the list and bring the window up."""
        self.set_title(title)
        self._heading.set_text(title)
        self._hint.set_text(hint)
        child = self._list.get_first_child()
        while child is not None:
            following = child.get_next_sibling()
            self._list.remove(child)
            child = following
        rows = station_rows(stations)
        for number, name, about in rows:
            self._list.append(self._row(number, name, about))
        self.set_default_size(WIDTH, min(MAX_HEIGHT, 96 + ROW_HEIGHT * max(1, len(rows))))
        self.present()

    def _row(self, number: str, name: str, about: str) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.station_name = name
        line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=14)
        line.add_css_class("voxa-radio-row")
        badge = Gtk.Label(label=number)
        badge.add_css_class("voxa-radio-number")
        badge.set_size_request(34, -1)
        text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        text.set_hexpand(True)
        title = Gtk.Label(label=name, xalign=0)
        title.add_css_class("voxa-radio-name")
        title.set_ellipsize(Pango.EllipsizeMode.END)
        text.append(title)
        if about:
            detail = Gtk.Label(label=about, xalign=0)
            detail.add_css_class("voxa-radio-about")
            detail.set_ellipsize(Pango.EllipsizeMode.END)
            text.append(detail)
        line.append(badge)
        line.append(text)
        row.set_child(line)
        return row

    def _on_row(self, _listbox, row) -> None:
        self.set_visible(False)
        self.on_station(row.station_name)

    def _on_key(self, _controller, keyval, _keycode, _mods) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.set_visible(False)
            return True
        return False
