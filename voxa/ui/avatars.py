"""Avatar model registry and download location.

This module is intentionally small: it defines the canonical avatar list,
resolves the directory where downloaded avatar models live, and exposes a
single helper used by the renderer and the download service.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path


def avatar_directory() -> Path:
    """Directory where downloaded avatar models are stored."""
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(data_home) / "voxa" / "avatars"


AVATAR_DIRECTORY = avatar_directory()

PORTRAIT_DIRECTORY = Path(__file__).parent / "assets" / "portraits"

_LOCALE_NAMES = {
    "en-US": "English — United States",
    "en-GB": "English — United Kingdom",
    "en-IE": "English — Ireland",
    "es-MX": "Spanish — Mexico",
    "fr-FR": "French — France",
    "de-DE": "German — Germany",
    "ja-JP": "Japanese — Japan",
    "ar-SA": "Arabic — Saudi Arabia",
    "en-IN": "English — India",
    "en-NG": "English — Nigeria",
}

RENDERER_GL3D = "gl_area"
RENDERER_PHOTO = "photo"
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
    portrait_path: Path = Path()
    culture: str = ""


def _avatar(
    avatar_id: str,
    display_name: str,
    voice: str,
    locale: str,
    gender: str,
    culture: str,
    skin_tone: tuple[float, float, float] = _DEFAULT_SKIN_TONE,
    hair_color: tuple[float, float, float] | None = None,
) -> AvatarDescriptor:
    return AvatarDescriptor(
        id=avatar_id,
        display_name=display_name,
        model_path=AVATAR_DIRECTORY / f"{avatar_id}.glb",
        thumbnail_path=AVATAR_DIRECTORY / f"{avatar_id}.png",
        portrait_path=PORTRAIT_DIRECTORY / f"{avatar_id}.jpg",
        voice=voice,
        renderer=RENDERER_PHOTO,
        skin_tone=skin_tone,
        hair_color=hair_color,
        locale=locale,
        gender=gender,
        culture=culture,
    )


AVATARS: tuple[AvatarDescriptor, ...] = (
    _avatar(
        "jack",
        "Jack",
        "en-US-GuyNeural",
        "en-US",
        "Male",
        "United States",
        (0.60, 0.48, 0.38),
        (0.35, 0.26, 0.18),
    ),
    _avatar(
        "grace",
        "Grace",
        "en-US-AriaNeural",
        "en-US",
        "Female",
        "United States",
        _DEFAULT_SKIN_TONE,
        (0.90, 0.92, 0.94),
    ),
    _avatar(
        "seamus",
        "Seamus",
        "en-IE-ConnorNeural",
        "en-IE",
        "Male",
        "Ireland",
        (0.62, 0.48, 0.36),
        (0.84, 0.76, 0.32),
    ),
    _avatar(
        "aoife",
        "Aoife",
        "en-IE-EmilyNeural",
        "en-IE",
        "Female",
        "Ireland",
        (0.55, 0.44, 0.34),
        (0.48, 0.31, 0.24),
    ),
    _avatar(
        "oliver",
        "Oliver",
        "en-GB-RyanNeural",
        "en-GB",
        "Male",
        "Britain",
        (0.58, 0.45, 0.35),
        (0.28, 0.22, 0.18),
    ),
    _avatar(
        "charlotte",
        "Charlotte",
        "en-GB-SoniaNeural",
        "en-GB",
        "Female",
        "Britain",
        (0.64, 0.52, 0.42),
        (0.88, 0.82, 0.66),
    ),
    _avatar(
        "juan",
        "Juan",
        "es-MX-JorgeNeural",
        "es-MX",
        "Male",
        "Mexico",
        (0.52, 0.42, 0.34),
        (0.24, 0.18, 0.15),
    ),
    _avatar(
        "valentina",
        "Valentina",
        "es-MX-DaliaNeural",
        "es-MX",
        "Female",
        "Mexico",
        (0.62, 0.49, 0.40),
        (0.27, 0.20, 0.16),
    ),
    _avatar(
        "etienne",
        "Étienne",
        "fr-FR-HenriNeural",
        "fr-FR",
        "Male",
        "France",
        (0.61, 0.51, 0.40),
        (0.38, 0.30, 0.25),
    ),
    _avatar(
        "camille",
        "Camille",
        "fr-FR-DeniseNeural",
        "fr-FR",
        "Female",
        "France",
        (0.67, 0.56, 0.47),
        (0.94, 0.82, 0.58),
    ),
    _avatar(
        "klaus",
        "Klaus",
        "de-DE-ConradNeural",
        "de-DE",
        "Male",
        "Germany",
        (0.60, 0.49, 0.40),
        (0.26, 0.21, 0.17),
    ),
    _avatar(
        "greta",
        "Greta",
        "de-DE-KatjaNeural",
        "de-DE",
        "Female",
        "Germany",
        (0.63, 0.52, 0.44),
        (0.96, 0.90, 0.75),
    ),
    _avatar(
        "hiroshi",
        "Hiroshi",
        "ja-JP-KeitaNeural",
        "ja-JP",
        "Male",
        "Japan",
        (0.55, 0.46, 0.38),
        (0.18, 0.16, 0.15),
    ),
    _avatar(
        "sakura",
        "Sakura",
        "ja-JP-NanamiNeural",
        "ja-JP",
        "Female",
        "Japan",
        (0.66, 0.56, 0.47),
        (0.72, 0.34, 0.42),
    ),
    _avatar("omar", "Omar", "ar-SA-HamedNeural", "ar-SA", "Male", "Arabia"),
    _avatar("layla", "Layla", "ar-SA-ZariyahNeural", "ar-SA", "Female", "Arabia"),
    _avatar("arjun", "Arjun", "en-IN-PrabhatNeural", "en-IN", "Male", "India"),
    _avatar("priya", "Priya", "en-IN-NeerjaNeural", "en-IN", "Female", "India"),
    _avatar("koda", "Koda", "en-US-ChristopherNeural", "en-US", "Male", "Native America"),
    _avatar("aiyana", "Aiyana", "en-US-JennyNeural", "en-US", "Female", "Native America"),
    _avatar("chidi", "Chidi", "en-NG-AbeoNeural", "en-NG", "Male", "Nigeria"),
    _avatar("amara", "Amara", "en-NG-EzinneNeural", "en-NG", "Female", "Nigeria"),
)


def _avatar_overrides(directory: Path) -> dict[str, dict[str, str]]:
    """Optional per-model overrides from <avatar dir>/avatars.json.

    Shape: ``{"<stem>": {"name": ..., "voice": ..., "gender": ...}}``.
    A missing or unreadable file is ignored.
    """
    path = directory / "avatars.json"
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        logging.debug("ignoring avatar overrides file %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _friendly_display_name(stem: str) -> str:
    """Human name for a discovered model stem.

    Suffix/marker tokens ("-bust", "-head", "-tripo", "-animated") are dropped
    and the rest is title-cased: "bald-father-bust" -> "Bald Father".
    """
    name = stem
    for token in ("-bust", "-head", "-tripo", "-animated"):
        name = name.replace(token, "")
    return name.replace("-", " ").title()


def _discovered_gender(stem: str) -> str:
    """Female when the stem mentions a woman, otherwise Male."""
    lowered = stem.lower()
    if any(word in lowered for word in ("female", "woman", "girl")):
        return "Female"
    return "Male"


def discover_avatars() -> list[AvatarDescriptor]:
    """Return any local .glb models in the avatar directory not already listed."""
    directory = Path(AVATAR_DIRECTORY)
    if not directory.is_dir():
        return []

    known = {avatar.id for avatar in AVATARS}
    overrides = _avatar_overrides(directory)
    discovered: list[AvatarDescriptor] = []
    used_names: dict[str, int] = {}

    for path in sorted(directory.glob("*.glb")):
        avatar_id = path.stem
        if avatar_id in known:
            continue

        override = overrides.get(avatar_id, {})
        display_name = override.get("name") or _friendly_display_name(avatar_id)
        gender = override.get("gender") or _discovered_gender(avatar_id)
        voice = override.get("voice") or (
            "en-US-AriaNeural" if gender == "Female" else "en-US-GuyNeural"
        )

        count = used_names.get(display_name, 0) + 1
        used_names[display_name] = count
        if count > 1:
            display_name = f"{display_name} {count}"

        discovered.append(
            AvatarDescriptor(
                id=avatar_id,
                display_name=display_name,
                model_path=path,
                thumbnail_path=directory / f"{avatar_id}.png",
                voice=voice,
                renderer="gl_area",
                skin_tone=_DEFAULT_SKIN_TONE,
                hair_color=None,
                locale="en-US",
                gender=gender,
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


def portrait_is_available(avatar: AvatarDescriptor) -> bool:
    """True when the avatar's portrait file exists on disk."""
    return Path(avatar.portrait_path).exists()


def available_avatars() -> tuple[AvatarDescriptor, ...]:
    """Every roster avatar whose portrait is on disk, sorted by name.

    Leftover 3D model files in the avatar directory are no longer offered: the faces are photo portraits now.
    """
    usable = [avatar for avatar in AVATARS if portrait_is_available(avatar)]
    return tuple(sorted(usable, key=lambda avatar: avatar.display_name))


def avatar_choice_label(avatar: AvatarDescriptor) -> str:
    """Human-readable preference label.

    Roster avatars get gender and culture appended; discovered local
    models are labelled by their display name alone.
    """
    if avatar not in AVATARS:
        return avatar.display_name
    return f"{avatar.display_name} ({avatar.gender}, {avatar.culture})"


def character_choices() -> list[tuple[str, str]]:
    """Choices shown in the Preferences dialog.

    The first entry is the current static badge, represented by an empty
    character id; only avatars with a downloaded model are offered after it.
    """
    choices: list[tuple[str, str]] = [("", "Classic badge")]
    choices.extend((avatar.id, avatar_choice_label(avatar)) for avatar in available_avatars())
    return choices


def model_is_downloaded(avatar: AvatarDescriptor) -> bool:
    """True when the avatar's model file is present in the avatar directory.

    Roster avatars are resolved against the current avatar directory so a
    relocated directory is honoured; discovered models carry their own path.
    """
    if avatar in AVATARS:
        return (Path(AVATAR_DIRECTORY) / f"{avatar.id}.glb").exists()
    return Path(avatar.model_path).exists()
