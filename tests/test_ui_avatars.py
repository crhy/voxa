from __future__ import annotations

import pytest

from voxa.ui.avatars import (
    AVATARS,
    RENDERER_GL3D,
    AvatarDescriptor,
    default_avatar,
    get_avatar,
    model_is_downloaded,
)


def test_registry_has_porcelain_grace_by_default() -> None:
    avatar = default_avatar()
    assert avatar.id == "porcelain-grace"
    assert avatar.display_name == "Porcelain Grace"
    assert avatar.model_path.endswith("porcelain-grace.glb")
    assert avatar.thumbnail_path.endswith("porcelain-grace.png")
    assert avatar.renderer == RENDERER_GL3D
    assert avatar in AVATARS


def test_get_avatar_returns_registered_id_or_none() -> None:
    assert get_avatar("porcelain-grace") is default_avatar()
    assert get_avatar("not-a-real-avatar") is None


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


def test_porcelain_grace_voice_is_a_known_tts_voice() -> None:
    try:
        from voxa.window import TTS_VOICES
    except Exception as exc:  # noqa: BLE001 - window/GTK is optional in headless tests
        pytest.skip(f"could not import TTS_VOICES: {exc}")

    assert default_avatar().voice in {voice for _label, voice in TTS_VOICES}
