"""Avatar model registry and download location.

This module is intentionally small: it defines the canonical avatar list,
resolves the directory where downloaded avatar models live, and exposes a
single helper used by the renderer and the download service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def avatar_directory() -> Path:
    """Directory where downloaded avatar models are stored."""
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(data_home) / "voxa" / "avatars"


AVATAR_DIRECTORY = avatar_directory()

_LOCALE_NAMES = {
    "en-US": "English — United States",
    "en-GB": "English — United Kingdom",
    "en-IE": "English — Ireland",
    "es-MX": "Spanish — Mexico",
    "fr-FR": "French — France",
    "de-DE": "German — Germany",
    "ja-JP": "Japanese — Japan",
}

RENDERER_GL3D = "gl_area"
_DEFAULT_SKIN_TONE = (0.52, 0.70, 0.84)
_DEFAULT_HAIR_COLOR = None


@dataclass(frozen=True)
class AvatarDescriptor:
    """One character the user can pick from the preferences dialog."""

    id: str
    display_name: str
    model_path: Path
    thumbnail_path: Path
    voice: str
    renderer: str
    skin_tone: tuple[float, float, float] | None = _DEFAULT_SKIN_TONE
    hair_color: tuple[float, float, float] | None = _DEFAULT_HAIR_COLOR
    locale: str = "en-US"
    gender: str = "Other"


def _avatar(
    avatar_id: str,
    display_name: str,
    voice: str,
    locale: str,
    gender: str,
    skin_tone: tuple[float, float, float],
    hair_color: tuple[float, float, float] | None = None,
) -> AvatarDescriptor:
    return AvatarDescriptor(
        id=avatar_id,
        display_name=display_name,
        model_path=AVATAR_DIRECTORY / f"{avatar_id}.glb",
        thumbnail_path=AVATAR_DIRECTORY / f"{avatar_id}.png",
        voice=voice,
        renderer="gl_area",
        skin_tone=skin_tone,
        hair_color=hair_color,
        locale=locale,
        gender=gender,
    )


AVATARS: tuple[AvatarDescriptor, ...] = (
    _avatar(
        "jack",
        "Jack",
        "en-US-GuyNeural",
        "en-US",
        "Male",
        (0.60, 0.48, 0.38),
        (0.35, 0.26, 0.18),
    ),
    _avatar(
        "grace",
        "Grace",
        "en-US-AriaNeural",
        "en-US",
        "Female",
        _DEFAULT_SKIN_TONE,
        (0.90, 0.92, 0.94),
    ),
    _avatar(
        "seamus",
        "Seamus",
        "en-IE-ConnorNeural",
        "en-IE",
        "Male",
        (0.62, 0.48, 0.36),
        (0.84, 0.76, 0.32),
    ),
    _avatar(
        "aoife",
        "Aoife",
        "en-IE-EmilyNeural",
        "en-IE",
        "Female",
        (0.55, 0.44, 0.34),
        (0.48, 0.31, 0.24),
    ),
    _avatar(
        "oliver",
        "Oliver",
        "en-GB-RyanNeural",
        "en-GB",
        "Male",
        (0.58, 0.45, 0.35),
        (0.28, 0.22, 0.18),
    ),
    _avatar(
        "charlotte",
        "Charlotte",
        "en-GB-SoniaNeural",
        "en-GB",
        "Female",
        (0.64, 0.52, 0.42),
        (0.88, 0.82, 0.66),
    ),
    _avatar(
        "juan",
        "Juan",
        "es-MX-JorgeNeural",
        "es-MX",
        "Male",
        (0.52, 0.42, 0.34),
        (0.24, 0.18, 0.15),
    ),
    _avatar(
        "valentina",
        "Valentina",
        "es-MX-DaliaNeural",
        "es-MX",
        "Female",
        (0.62, 0.49, 0.40),
        (0.27, 0.20, 0.16),
    ),
    _avatar(
        "etienne",
        "Étienne",
        "fr-FR-HenriNeural",
        "fr-FR",
        "Male",
        (0.61, 0.51, 0.40),
        (0.38, 0.30, 0.25),
    ),
    _avatar(
        "camille",
        "Camille",
        "fr-FR-DeniseNeural",
        "fr-FR",
        "Female",
        (0.67, 0.56, 0.47),
        (0.94, 0.82, 0.58),
    ),
    _avatar(
        "klaus",
        "Klaus",
        "de-DE-ConradNeural",
        "de-DE",
        "Male",
        (0.60, 0.49, 0.40),
        (0.26, 0.21, 0.17),
    ),
    _avatar(
        "greta",
        "Greta",
        "de-DE-KatjaNeural",
        "de-DE",
        "Female",
        (0.63, 0.52, 0.44),
        (0.96, 0.90, 0.75),
    ),
    _avatar(
        "hiroshi",
        "Hiroshi",
        "ja-JP-KeitaNeural",
        "ja-JP",
        "Male",
        (0.55, 0.46, 0.38),
        (0.18, 0.16, 0.15),
    ),
    _avatar(
        "sakura",
        "Sakura",
        "ja-JP-NanamiNeural",
        "ja-JP",
        "Female",
        (0.66, 0.56, 0.47),
        (0.72, 0.34, 0.42),
    ),
)


def discover_avatars() -> list[AvatarDescriptor]:
    """Return any local .glb models in the avatar directory not already listed."""
    directory = Path(AVATAR_DIRECTORY)
    if not directory.is_dir():
        return []

    known = {avatar.id for avatar in AVATARS}
    discovered: list[AvatarDescriptor] = []

    for path in sorted(directory.glob("*.glb")):
        avatar_id = path.stem
        if avatar_id in known:
            continue

        discovered.append(
            AvatarDescriptor(
                id=avatar_id,
                display_name=avatar_id.replace("-", " ").title(),
                model_path=path,
                thumbnail_path=directory / f"{avatar_id}.png",
                voice="en-US-AriaNeural",
                renderer="gl_area",
                skin_tone=_DEFAULT_SKIN_TONE,
                hair_color=None,
                locale="en-US",
                gender="Other",
            )
        )

    return discovered


def all_avatars() -> tuple[AvatarDescriptor, ...]:
    """Return the canonical roster plus any ad-hoc local models."""
    return AVATARS + tuple(discover_avatars())


def get_avatar(character_id: str | None) -> AvatarDescriptor | None:
    """Return an avatar descriptor by character id, or None for the Classic badge."""
    if not character_id:
        return None

    for avatar in all_avatars():
        if avatar.id == character_id:
            return avatar

    return None


def default_avatar() -> AvatarDescriptor:
    """Return Grace, the default 3D avatar."""
    return get_avatar("grace") or AVATARS[0]


def avatar_choice_label(avatar: AvatarDescriptor) -> str:
    """Human-readable preference label including gender and language region."""
    locale = _LOCALE_NAMES.get(avatar.locale, avatar.locale)
    return f"{avatar.display_name} ({avatar.gender}, {locale})"


def character_choices() -> list[tuple[str, str]]:
    """Choices shown in the Preferences dialog.

    The first entry is the current static badge, represented by an empty
    character id.
    """
    choices: list[tuple[str, str]] = [("", "Classic badge")]
    choices.extend((avatar.id, avatar_choice_label(avatar)) for avatar in all_avatars())
    return choices


def model_is_downloaded(avatar: AvatarDescriptor) -> bool:
    """True when the avatar's model file is present in the avatar directory."""
    return Path(avatar.model_path).exists()
