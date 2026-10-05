from __future__ import annotations

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no GTK display available")

from voxa.ui.avatars import AvatarDescriptor  # noqa: E402
from voxa.ui.character_picker import (  # noqa: E402
    EMPTY_LABEL,
    CharacterPicker,
    PortraitPicker,
    picker_entries,  # noqa: E402
)


def _avatar(avatar_id: str, display_name: str) -> AvatarDescriptor:
    return AvatarDescriptor(
        id=avatar_id,
        display_name=display_name,
        model_path=f"/models/{avatar_id}.glb",
        thumbnail_path=f"/models/{avatar_id}.png",
        voice="en-US-GuyNeural",
        renderer="gl_area",
    )


def test_refresh_does_not_fire_callback() -> None:
    fired: list[str] = []
    picker = CharacterPicker(on_character_selected=fired.append)
    picker.refresh([_avatar("grace", "Grace"), _avatar("jack", "Jack")])
    assert fired == []
    picker.refresh([_avatar("grace", "Grace")])
    assert fired == []


def test_set_selected_does_not_fire_callback() -> None:
    fired: list[str] = []
    picker = CharacterPicker(on_character_selected=fired.append)
    picker.refresh([_avatar("grace", "Grace"), _avatar("jack", "Jack")])
    picker.set_selected("jack")
    assert picker.get_selected() == "jack"
    assert fired == []


def test_user_selection_fires_callback_with_character_id() -> None:
    fired: list[str] = []
    picker = CharacterPicker(on_character_selected=fired.append)
    picker.refresh([_avatar("grace", "Grace"), _avatar("jack", "Jack")])
    picker._dropdown.set_selected(1)
    assert fired == ["jack"]


def test_empty_avatar_list() -> None:
    fired: list[str] = []
    picker = CharacterPicker(on_character_selected=fired.append)
    picker.refresh([])
    assert not picker._dropdown.get_sensitive()
    assert picker._items.get_n_items() == 1
    assert picker._items.get_item(0).get_string() == EMPTY_LABEL
    assert picker.get_selected() == ""
    picker._dropdown.set_selected(0)
    assert fired == []


def test_refresh_uses_available_avatars_by_default() -> None:
    picker = CharacterPicker()
    picker.refresh()
    assert picker._dropdown.get_sensitive()
    assert picker._items.get_n_items() >= 1


def test_dropdown_tooltip_tracks_selected_id() -> None:
    picker = CharacterPicker()
    picker.refresh([_avatar("grace", "Grace"), _avatar("jack", "Jack")])
    picker.set_selected("jack")
    assert picker._dropdown.get_tooltip_text() == "jack"


def test_dropdown_is_a_real_control_not_a_label() -> None:
    picker = CharacterPicker()
    picker.refresh([_avatar("grace", "Grace")])
    assert picker._dropdown.get_sensitive()
    assert not isinstance(picker._dropdown, Gtk.Label)


def test_portrait_picker_is_a_menu_button() -> None:
    picker = PortraitPicker()
    assert isinstance(picker, Gtk.MenuButton)


def test_portrait_refresh_does_not_fire_callbacks() -> None:
    fired: list[str] = []
    picker = PortraitPicker(on_character_selected=fired.append)
    picker.refresh([_avatar("grace", "Grace"), _avatar("jack", "Jack")])
    assert fired == []


def test_portrait_set_selected_does_not_fire_callback() -> None:
    fired: list[str] = []
    picker = PortraitPicker(on_character_selected=fired.append)
    picker.refresh([_avatar("grace", "Grace"), _avatar("jack", "Jack")])
    picker.set_selected("jack")
    assert picker.get_selected() == "jack"
    assert fired == []


def test_portrait_set_selected_ignores_unknown_id() -> None:
    picker = PortraitPicker()
    picker.refresh([_avatar("grace", "Grace")])
    picker.set_selected("nope")
    assert picker.get_selected() == ""


def test_portrait_empty_avatars() -> None:
    picker = PortraitPicker()
    picker.refresh([])
    assert picker.get_selected() == ""
    assert picker._flow.get_first_child() is None


def test_picker_entries_sorted_by_display_name() -> None:
    avatars = [_avatar("jack", "Jack"), _avatar("grace", "Grace")]
    entries = picker_entries(avatars)
    assert [name for _id, name, _path in entries] == ["Grace", "Jack"]
    assert [cid for cid, _name, _path in entries] == ["grace", "jack"]


def test_portrait_character_click_fires_callback_once() -> None:
    fired: list[str] = []
    picker = PortraitPicker(on_character_selected=fired.append)
    picker.refresh([_avatar("grace", "Grace"), _avatar("jack", "Jack")])
    handler = picker._make_character_handler("jack")
    handler(picker._cell_buttons["jack"])
    assert fired == ["jack"]
    assert picker.get_selected() == "jack"
