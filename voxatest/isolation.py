"""Fixture isolation: VoxaTest's playback must be recognisable, so Voxa can keep it out of her ear.

VoxaTest speaks the test "user" through the speakers. To Voxa's canceller that sound is just another
stream on the output; if it lands in her echo reference it is subtracted from the microphone and the
hearing measurement measures nothing (handoff 8.2: "keep test-fixture playback out of Voxa's
echo-reference cancellation policy when it represents the human"). So every stream this harness plays
carries a tag Voxa can see - ``application.name`` / ``media.role`` = ``voxatest`` - and
:mod:`voxa.reference` refuses a reference that contains it, logging a WARNING when it cannot.

The tag itself is defined on the product side (:func:`voxa.reference.stream_properties`), because the
dependency runs one way: voxatest imports Voxa, never the reverse. This module is the harness's own
vocabulary for it, plus the check ``voxatest doctor`` reports.

No GStreamer import is needed to use this module: the element and the Gst module are passed in, so the
property handling is testable with a fake GStreamer, exactly like ``tests/test_audio_webrtc.py``.
"""

from __future__ import annotations

from typing import Any

from voxa.reference import (
    TEST_MEDIA_NAME,
    TEST_STREAM_DESCRIPTION,
    TEST_STREAM_TAG,
    apply_stream_tag,
    stream_properties,
    test_inputs,
)

#: The tag as the harness names it. All three spellings mean the same stream property.
APPLICATION_NAME = TEST_STREAM_TAG
MEDIA_ROLE = TEST_STREAM_TAG
MEDIA_NAME = TEST_MEDIA_NAME
STREAM_DESCRIPTION = TEST_STREAM_DESCRIPTION


def playback_properties() -> dict[str, str]:
    """The stream properties every playback of this harness carries."""
    return stream_properties()


def property_names() -> list[str]:
    """The property names Voxa's reference policy looks for in a sink input."""
    return sorted(stream_properties())


def tag_playback(playbin: Any, gst: Any = None) -> dict[str, bool]:
    """Tag a GStreamer player as the test voice; returns what reached it (see ``apply_stream_tag``)."""
    return apply_stream_tag(playbin, gst)


def tag_isolation(applied: dict[str, bool]) -> bool:
    """True when the tag reached the stream: the property structure, the sink's own properties,
    or the individual stream properties."""
    if applied.get("property-structure") or applied.get("properties"):
        return True
    return all(applied.get(key) for key in stream_properties())


def is_test_stream(row: dict) -> bool:
    """Is this sink-input row the test voice? (The same rule Voxa's reference policy uses.)"""
    return bool(test_inputs([row]))


def describe() -> str:
    """One line naming the tag, for ``voxatest doctor`` and the soak report."""
    return ", ".join(f"{key}={value}" for key, value in stream_properties().items())
