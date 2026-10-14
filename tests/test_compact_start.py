"""Issue #98: compact start size and a darker window background."""

from __future__ import annotations

from voxa.theme import BUBBLE_BG_DARK, BUBBLE_BG_LIGHT, initial_size, window_bg_color


def _luminance(color: str) -> float:
    r = int(color[1:3], 16) / 255
    g = int(color[3:5], 16) / 255
    b = int(color[5:7], 16) / 255

    def channel(value: float) -> float:
        if value <= 0.03928:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def test_initial_size_without_saved_size_is_compact() -> None:
    assert initial_size(None) == (1, 1)


def test_saved_size_is_returned_unchanged() -> None:
    saved = (1200, 760)
    assert initial_size(saved) == saved


def test_window_background_differs_from_bubble() -> None:
    assert window_bg_color(True) != BUBBLE_BG_DARK
    assert window_bg_color(False) != BUBBLE_BG_LIGHT


def test_window_background_is_darker_than_bubble() -> None:
    assert _luminance(window_bg_color(True)) < _luminance(BUBBLE_BG_DARK)
    assert _luminance(window_bg_color(False)) < _luminance(BUBBLE_BG_LIGHT)
