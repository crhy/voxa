"""Registry of downloadable 3D avatars (PHASE 22 groundwork).

Nothing consumes this yet: with one avatar there is nothing to choose between,
and the Preferences picker is future work.  The descriptor and registry exist so
the multi-avatar feature is designed once rather than twice - id, display name,
model path, thumbnail, voice preference and renderer kind are exactly what the
roadmap asks for.

The registry is deliberately GTK-free so it can be tested without a display.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Where downloaded avatar models live.  Kept in one place so the future picker
# and the renderer agree on it.
AVATAR_DIRECTORY = os.path.expanduser("~/.local/share/voxa/avatars")

RENDERER_GL3D = "gl3d"


@dataclass(frozen=True)
class AvatarDescriptor:
    """Everything an avatar needs to be selected, rendered and voiced."""

    id: str
    display_name: str
    model_path: str
    thumbnail_path: str
    voice: str
    renderer: str


AVATARS: tuple[AvatarDescriptor, ...] = (
    AvatarDescriptor(
        id="porcelain-grace",
        display_name="Porcelain Grace",
        # CC0 model from issue #2; the file may not be downloaded yet, and the
        # renderer treats a missing model as "use the procedural face".
        model_path=os.path.join(AVATAR_DIRECTORY, "porcelain-grace.glb"),
        thumbnail_path=os.path.join(AVATAR_DIRECTORY, "porcelain-grace.png"),
        voice="en-US-AriaNeural",
        renderer=RENDERER_GL3D,
    ),
)


def get_avatar(avatar_id: str) -> AvatarDescriptor | None:
    """Return the avatar with this id, or None when it is not in the registry."""
    for avatar in AVATARS:
        if avatar.id == avatar_id:
            return avatar
    return None


def default_avatar() -> AvatarDescriptor:
    """The avatar used when the user has not picked one."""
    return AVATARS[0]


def model_is_downloaded(avatar: AvatarDescriptor) -> bool:
    """True when this avatar's model file is actually present on disk."""
    return os.path.isfile(os.path.expanduser(avatar.model_path))
