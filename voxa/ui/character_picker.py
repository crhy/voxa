"""Character picker: a dropdown over the avatars that actually have a model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Pango", "1.0")
gi.require_version("Gdk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, Gtk, Pango  # noqa: E402

from .avatars import available_avatars  # noqa: E402

EMPTY_LABEL = "No characters available"
MAX_BUTTON_CHARS = 24

PORTRAIT_BUTTON_SIZE = 36
PORTRAIT_SHARP_PX = 144
PORTRAIT_CELL_SIZE = 64
POPOVER_MAX_HEIGHT = 420
FACE_MODE_LABELS = (("live", "Live"), ("prerendered", "Pre-rendered"), ("still", "Still portrait"))
FACE_LABEL_TO_MODE = {label: mode for mode, label in FACE_MODE_LABELS}
DEFAULT_PORTRAIT_ICON = "avatar-default-symbolic"


def picker_entries(avatars=None) -> list[tuple[str, str, Path]]:
    """Pure helper: (id, display name, portrait path) sorted by display name."""
    source = list(avatars) if avatars is not None else list(available_avatars())
    return [(avatar.id, avatar.display_name, Path(avatar.portrait_path)) for avatar in
            sorted(source, key=lambda avatar: avatar.display_name)]


class CharacterPicker(Gtk.Box):
    """Dropdown over the available characters.

    Programmatic updates (``refresh`` / ``set_selected``) never fire the
    callback; only a real user selection does.
    """

    def __init__(self, on_character_selected: Callable[[str], None] | None = None) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        label = Gtk.Label(label="Character")
        label.add_css_class("voxa-section")

        self._items = Gtk.StringList.new([])
        self._dropdown = Gtk.DropDown(model=self._items)
        self._dropdown.set_sensitive(False)

        button_factory = Gtk.SignalListItemFactory()
        button_factory.connect("setup", self._setup_button_label)
        button_factory.connect("bind", self._bind_button_label)
        self._dropdown.set_factory(button_factory)

        list_factory = Gtk.SignalListItemFactory()
        list_factory.connect("setup", self._setup_list_label)
        list_factory.connect("bind", self._bind_list_label)
        self._dropdown.set_list_factory(list_factory)

        self.on_character_selected = on_character_selected
        self._avatars: list = []
        self._updating = False
        self._dropdown.connect("notify::selected", self._on_selected)

        self.append(label)
        self.append(self._dropdown)

    def _setup_button_label(self, _factory, list_item: Gtk.ListItem) -> None:
        lbl = Gtk.Label(xalign=0)
        lbl.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        lbl.set_width_chars(12)
        lbl.set_max_width_chars(MAX_BUTTON_CHARS)
        list_item.set_child(lbl)

    def _bind_button_label(self, _factory, list_item: Gtk.ListItem) -> None:
        item = list_item.get_item()
        if item is None:
            return
        text = item.get_string()
        lbl = list_item.get_child()
        lbl.set_text(text)
        lbl.set_tooltip_text(text)

    def _setup_list_label(self, _factory, list_item: Gtk.ListItem) -> None:
        lbl = Gtk.Label(xalign=0)
        list_item.set_child(lbl)

    def _bind_list_label(self, _factory, list_item: Gtk.ListItem) -> None:
        item = list_item.get_item()
        if item is None:
            return
        lbl = list_item.get_child()
        lbl.set_text(item.get_string())

    def refresh(self, avatars=None) -> None:
        """Fill the dropdown from available avatars without firing the callback."""
        self._updating = True
        try:
            self._avatars = list(avatars) if avatars is not None else list(available_avatars())
            if not self._avatars:
                self._items.splice(0, self._items.get_n_items(), [EMPTY_LABEL])
            else:
                labels = [avatar.display_name for avatar in self._avatars]
                self._items.splice(0, self._items.get_n_items(), labels)
            self._dropdown.set_sensitive(bool(self._avatars))
            self._dropdown.set_tooltip_text(self.get_selected() or None)
        finally:
            self._updating = False

    def set_selected(self, character_id: str) -> None:
        """Select a character by id without firing the callback."""
        index = next((i for i, avatar in enumerate(self._avatars) if avatar.id == character_id), -1)
        self._updating = True
        try:
            if index >= 0:
                self._dropdown.set_selected(index)
        finally:
            self._updating = False

    def get_selected(self) -> str:
        """Character id of the current selection, or "" when none is selected."""
        if not self._avatars:
            return ""
        index = self._dropdown.get_selected()
        if 0 <= index < len(self._avatars):
            return self._avatars[index].id
        return ""

    def _on_selected(self, *_args) -> None:
        self._dropdown.set_tooltip_text(self.get_selected() or None)
        if self._updating or not self._avatars or self.on_character_selected is None:
            return
        self.on_character_selected(self.get_selected())


def _portrait_texture(path: Path) -> Gdk.Texture | None:
    """Decode a portrait JPEG and scale it up for a sharp small render."""
    try:
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(path))
        scaled = pixbuf.scale_simple(PORTRAIT_SHARP_PX, PORTRAIT_SHARP_PX, GdkPixbuf.InterpType.BILINEAR)
        return Gdk.Texture.new_for_pixbuf(scaled)
    except Exception:
        return None


class PortraitPicker(Gtk.MenuButton):
    """Top-right portrait button: a round photo of the selected character.

    Clicking opens a popover with a 3-column grid of portraits (single selection)
    and a Face drop-down. Programmatic ``set_selected`` / ``set_face_mode`` never
    fire the callbacks; only a real user selection does.
    """

    def __init__(self, on_character_selected=None, on_face_mode_selected=None) -> None:
        super().__init__()
        self.on_character_selected = on_character_selected
        self.on_face_mode_selected = on_face_mode_selected

        self._avatars: list = []
        self._selected_id = ""
        self._face_mode = "prerendered"
        self._updating = False
        self._cell_buttons: dict[str, Gtk.Button] = {}
        self._textures: dict[str, Gdk.Texture] = {}

        # Portraits are decoded the first time the popover is shown, not at window start-up.
        self._popover = Gtk.Popover()
        self._popover.connect("show", lambda *_args: self._load_textures())
        self.set_popover(self._popover)

        self._content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self._scrolled = Gtk.ScrolledWindow()
        self._scrolled.set_propagate_natural_height(True)
        self._scrolled.set_max_content_height(POPOVER_MAX_HEIGHT)
        self._scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._flow = Gtk.FlowBox()
        self._flow.set_min_children_per_line(3)
        self._flow.set_max_children_per_line(3)
        self._flow.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._scrolled.set_child(self._flow)
        self._content.append(self._scrolled)
        self._content.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        face_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        face_label = Gtk.Label(label="Face")
        face_label.set_xalign(0)
        self._face_items = Gtk.StringList.new([mode for mode, _label in FACE_MODE_LABELS])
        self._face_dropdown = Gtk.DropDown(model=self._face_items)
        self._face_dropdown.set_sensitive(True)
        list_factory = Gtk.SignalListItemFactory()
        list_factory.connect("setup", self._face_setup_label)
        list_factory.connect("bind", self._face_bind_label)
        self._face_dropdown.set_factory(list_factory)
        self._face_dropdown.connect("notify::selected", self._on_face_selected)
        face_row.append(face_label)
        face_row.append(self._face_dropdown)
        self._content.append(face_row)

        self._popover.set_child(self._content)
        self._refresh_button()

    def refresh(self, avatars=None) -> None:
        """Rebuild the grid and selection from avatars without firing callbacks."""
        self._updating = True
        try:
            self._avatars = list(avatars) if avatars is not None else list(available_avatars())
            self._cell_buttons = {}
            child = self._flow.get_first_child()
            while child is not None:
                nxt = child.get_next_sibling()
                self._flow.remove(child)
                child = nxt
            for avatar in sorted(self._avatars, key=lambda item: item.display_name):
                button = Gtk.Button()
                button.add_css_class("voxa-portrait-cell")
                cell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
                picture = Gtk.Image()
                picture.set_pixel_size(PORTRAIT_CELL_SIZE)
                picture.set_overflow(Gtk.Overflow.HIDDEN)
                picture.add_css_class("voxa-portrait-button")
                cell.append(picture)
                name = Gtk.Label(label=avatar.display_name)
                name.set_xalign(0.5)
                name.add_css_class("caption")
                cell.append(name)
                button.set_child(cell)
                button.set_tooltip_text(avatar.display_name)
                button.connect("clicked", self._make_character_handler(avatar.id))
                self._flow.append(button)
                self._cell_buttons[avatar.id] = button
            self._refresh_button()
        finally:
            self._updating = False

    def set_selected(self, character_id: str) -> None:
        """Select a character by id without firing the callback."""
        self._selected_id = character_id if character_id in self._cell_buttons else ""
        self._refresh_button()

    def get_selected(self) -> str:
        return self._selected_id

    def set_face_mode(self, mode: str) -> None:
        """Select a face mode by value without firing the callback."""
        if mode not in FACE_LABEL_TO_MODE.values():
            return
        self._face_mode = mode
        self._updating = True
        try:
            index = next(i for i, (value, _label) in enumerate(FACE_MODE_LABELS) if value == mode)
            self._face_dropdown.set_selected(index)
        finally:
            self._updating = False

    def get_face_mode(self) -> str:
        return self._face_mode

    def _make_character_handler(self, character_id: str):
        def _handler(button: Gtk.Button, cid: str = character_id) -> None:
            if self._updating:
                return
            self._selected_id = cid
            self._refresh_button()
            self.popdown()
            if self.on_character_selected is not None:
                self.on_character_selected(cid)
        return _handler

    def _on_face_selected(self, *_args) -> None:
        if self._updating:
            return
        item = self._face_dropdown.get_selected_item()
        if item is None:
            return
        self._face_mode = item.get_string()
        if self.on_face_mode_selected is not None:
            self.on_face_mode_selected(self._face_mode)

    def _face_setup_label(self, _factory, list_item: Gtk.ListItem) -> None:
        list_item.set_child(Gtk.Label(label=""))

    def _face_bind_label(self, _factory, list_item: Gtk.ListItem) -> None:
        item = list_item.get_item()
        if item is None:
            return
        label = FACE_LABEL_TO_MODE.get(item.get_string(), item.get_string())
        list_item.get_child().set_text(label)

    def _load_textures(self) -> None:
        """Decode portrait JPEGs only when the popover actually opens."""
        for avatar in self._avatars:
            if avatar.id in self._textures:
                continue
            texture = _portrait_texture(Path(avatar.portrait_path))
            if texture is not None:
                self._textures[avatar.id] = texture
        for avatar_id, button in self._cell_buttons.items():
            texture = self._textures.get(avatar_id)
            if texture is None:
                continue
            cell = button.get_child()
            if cell is not None:
                picture = cell.get_first_child()
                if picture is not None:
                    picture.set_from_paintable(texture)

    def _refresh_button(self) -> None:
        """Show the selected character's portrait, or the default icon."""
        texture = self._textures.get(self._selected_id) if self._selected_id else None
        if texture is None and self._selected_id:
            texture = _portrait_texture(self._portrait_path_for(self._selected_id))
        if texture is not None:
            picture = Gtk.Image.new_from_paintable(texture)
            picture.set_pixel_size(PORTRAIT_BUTTON_SIZE)
            picture.add_css_class("voxa-portrait-button")
            picture.set_overflow(Gtk.Overflow.HIDDEN)
            self.set_child(picture)
            avatar = next((item for item in self._avatars if item.id == self._selected_id), None)
            self.set_tooltip_text(avatar.display_name if avatar else "")
        else:
            self.set_child(None)
            self.set_icon_name(DEFAULT_PORTRAIT_ICON)
            self.set_tooltip_text("")
        for avatar_id, button in self._cell_buttons.items():
            if avatar_id == self._selected_id:
                button.add_css_class("selected-character")
            else:
                button.remove_css_class("selected-character")

    def _portrait_path_for(self, character_id: str) -> Path:
        avatar = next((item for item in self._avatars if item.id == character_id), None)
        return Path(avatar.portrait_path) if avatar else Path()
