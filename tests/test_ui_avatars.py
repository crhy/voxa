from __future__ import annotations

from pathlib import Path

import pytest

from voxa.ui import avatars
from voxa.ui.avatars import (
    AVATAR_DIRECTORY,
    AVATARS,
    RENDERER_GL3D,
    RENDERER_PHOTO,
    AvatarDescriptor,
    all_avatars,
    available_avatars,
    avatar_choice_label,
    avatar_directory,
    character_choices,
    default_avatar,
    discover_avatars,
    get_avatar,
    model_is_downloaded,
    portrait_is_available,
)


def test_registry_has_grace_by_default() -> None:
    avatar = default_avatar()
    assert avatar.id == "grace"
    assert avatar.display_name == "Grace"
    assert str(avatar.model_path).endswith("grace.glb")
    assert str(avatar.thumbnail_path).endswith("grace.png")
    assert avatar.renderer == RENDERER_PHOTO
    assert avatar in AVATARS


def test_roster_has_twenty_two_characters() -> None:
    assert len(AVATARS) == 22
    ids = {avatar.id for avatar in AVATARS}
    assert ids == {
        "jack",
        "grace",
        "seamus",
        "aoife",
        "oliver",
        "charlotte",
        "juan",
        "valentina",
        "etienne",
        "camille",
        "klaus",
        "greta",
        "hiroshi",
        "sakura",
        "omar",
        "layla",
        "arjun",
        "priya",
        "koda",
        "aiyana",
        "chidi",
        "amara",
    }
    assert len(ids) == 22


def test_roster_avatars_have_cosmetics() -> None:
    for avatar in AVATARS:
        assert len(avatar.skin_tone) == 3
        assert all(0.0 <= channel <= 1.0 for channel in avatar.skin_tone)
        if avatar.hair_color is not None:
            assert len(avatar.hair_color) == 3
            assert all(0.0 <= channel <= 1.0 for channel in avatar.hair_color)


def test_get_avatar_returns_registered_id_or_none() -> None:
    assert get_avatar("grace") == default_avatar()
    assert get_avatar("not-a-real-avatar") is None
    assert get_avatar("") is None
    assert get_avatar(None) is None


def test_character_choices_start_with_classic_badge() -> None:
    choices = character_choices()
    assert choices[0] == ("", "Classic badge")
    for character_id, _label in choices[1:]:
        avatar = get_avatar(character_id)
        assert portrait_is_available(avatar) or model_is_downloaded(avatar)


def test_character_choice_label_includes_gender_and_culture() -> None:
    by_id = {avatar.id: avatar for avatar in AVATARS}
    seamus = avatar_choice_label(by_id["seamus"])
    assert seamus == "Seamus (Male, Ireland)"
    sakura = avatar_choice_label(by_id["sakura"])
    assert sakura == "Sakura (Female, Japan)"


def test_available_avatars_include_roster_when_portraits_exist(monkeypatch, tmp_path) -> None:
    directory = tmp_path / "avatars"
    directory.mkdir()
    monkeypatch.setattr(avatars, "AVATAR_DIRECTORY", str(directory))
    (directory / "stray-bust.glb").write_bytes(b"glb")

    available = available_avatars()
    ids = {avatar.id for avatar in available}
    assert {avatar.id for avatar in AVATARS} <= ids
    assert "stray-bust" not in ids
    assert all(portrait_is_available(avatar) or model_is_downloaded(avatar) for avatar in available)


def test_character_choices_offer_roster_with_portraits(monkeypatch, tmp_path) -> None:
    directory = tmp_path / "avatars"
    directory.mkdir()
    monkeypatch.setattr(avatars, "AVATAR_DIRECTORY", str(directory))
    (directory / "stray-bust.glb").write_bytes(b"glb")

    offered = {character_id for character_id, _label in character_choices()}
    assert offered == {""} | {avatar.id for avatar in AVATARS}


def test_roster_voice_ids_are_in_tts_voice_choices() -> None:
    try:
        from voxa.window import TTS_VOICES
    except Exception as exc:  # noqa: BLE001 - window/GTK is optional in headless tests
        pytest.skip(f"could not import TTS_VOICES: {exc}")

    available = {voice for _label, voice in TTS_VOICES}
    original_ids = {
        "jack",
        "grace",
        "seamus",
        "aoife",
        "oliver",
        "charlotte",
        "juan",
        "valentina",
        "etienne",
        "camille",
        "klaus",
        "greta",
        "hiroshi",
        "sakura",
    }
    original_voices = {avatar.voice for avatar in AVATARS if avatar.id in original_ids}
    assert original_voices <= available


def test_model_is_downloaded_checks_the_actual_file(tmp_path) -> None:
    avatar = AvatarDescriptor(
        id="test-avatar",
        display_name="Test Avatar",
        model_path=str(tmp_path / "test-avatar.glb"),
        thumbnail_path=str(tmp_path / "test-avatar.png"),
        voice="en-US-AriaNeural",
        renderer=RENDERER_GL3D,
    )

    assert not model_is_downloaded(avatar)

    model_path = tmp_path / "test-avatar.glb"
    model_path.write_bytes(b"glb")
    assert model_is_downloaded(avatar)


def test_discover_avatars_includes_stray_models(monkeypatch, tmp_path) -> None:
    directory = tmp_path / "avatars"
    directory.mkdir()
    monkeypatch.setattr(avatars, "AVATAR_DIRECTORY", str(directory))
    (directory / "extra.glb").write_bytes(b"glb")

    discovered = discover_avatars()
    assert any(avatar.id == "extra" for avatar in discovered)
    assert any(avatar.id == "grace" for avatar in all_avatars())


def test_avatar_directory_prefers_xdg_data_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    assert avatar_directory() == Path(tmp_path / "xdg-data", "voxa", "avatars")


def test_avatar_directory_falls_back_to_local_share_when_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert avatar_directory() == Path(tmp_path, ".local", "share", "voxa", "avatars")


def test_avatar_directory_ignores_an_empty_xdg_data_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert avatar_directory() == Path(tmp_path, ".local", "share", "voxa", "avatars")


def test_registry_paths_use_the_resolved_avatar_directory() -> None:
    assert AVATAR_DIRECTORY == avatar_directory()
    for avatar in AVATARS:
        assert avatar.model_path.is_relative_to(AVATAR_DIRECTORY)
        assert avatar.thumbnail_path.is_relative_to(AVATAR_DIRECTORY)


def test_roster_portraits_exist_and_are_jpegs() -> None:
    for avatar in AVATARS:
        assert avatar.portrait_path.is_file()
        with avatar.portrait_path.open("rb") as handle:
            assert handle.read(2) == b"\xff\xd8"


def test_roster_ids_are_unique() -> None:
    ids = [avatar.id for avatar in AVATARS]
    assert len(ids) == len(set(ids))


def test_amara_voice_matches_roster() -> None:
    assert get_avatar("amara").voice == "en-NG-EzinneNeural"


def test_old_roster_ids_still_present() -> None:
    ids = {avatar.id for avatar in AVATARS}
    old = {
        "jack",
        "grace",
        "seamus",
        "aoife",
        "oliver",
        "charlotte",
        "juan",
        "valentina",
        "etienne",
        "camille",
        "klaus",
        "greta",
        "hiroshi",
        "sakura",
    }
    assert old <= ids


def test_aoife_choice_label_text() -> None:
    by_id = {avatar.id: avatar for avatar in AVATARS}
    assert avatar_choice_label(by_id["aoife"]) == "Aoife (Female, Ireland)"


def test_available_avatars_contains_all_roster_without_models(monkeypatch, tmp_path) -> None:
    directory = tmp_path / "avatars"
    directory.mkdir()
    monkeypatch.setattr(avatars, "AVATAR_DIRECTORY", str(directory))
    available = available_avatars()
    assert {avatar.id for avatar in available} >= {avatar.id for avatar in AVATARS}


def test_roster_avatars_use_photo_renderer() -> None:
    for avatar in AVATARS:
        assert avatar.renderer == RENDERER_PHOTO
