"""The floating head must show the High (neural) face exactly as the main window does."""
import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

from voxa.ui.focus_window import FocusWindow  # noqa: E402


class Frames:
    utterance_id = 7

    def index_at(self, seconds):
        return 0

    def get(self, index):
        return b"jpeg"


@pytest.fixture
def window():
    if not Gtk.init_check():
        pytest.skip("no display")
    return FocusWindow()


def test_pop_up_follows_the_face_mode_and_takes_live_frames(window):
    assert window.renderer.get_mode() == "prerendered"
    window.set_face_mode("live")
    frames = Frames()
    window.set_live_frames(frames)
    assert window.renderer.get_mode() == "live"
    assert window.renderer._live_buffer is frames
    window.set_live_frames(None)
    assert window.renderer._live_buffer is None


def test_shell_hands_the_same_frames_to_both_faces():
    from types import SimpleNamespace

    from voxa.ui.shell import AssistantShell

    main_face = SimpleNamespace(_live_buffer=None, set_live_frames=lambda buffer: setattr(main_face, "_live_buffer", buffer))
    seen = {}
    shell = SimpleNamespace(
        assistant_view=SimpleNamespace(_photo_renderer=main_face, get_face_mode=lambda: "live"),
        focus_window=SimpleNamespace(
            set_live_frames=lambda buffer: seen.__setitem__("frames", buffer),
            set_face_mode=lambda mode: seen.__setitem__("mode", mode),
        ),
    )
    frames = Frames()
    AssistantShell.show_live_frames(shell, frames)
    assert main_face._live_buffer is frames and seen["frames"] is frames
    AssistantShell.sync_focus_face(shell)
    assert seen == {"frames": frames, "mode": "live"}
