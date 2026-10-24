import threading
from types import SimpleNamespace

from voxa.faceinstall import Manifest, Part, gb, offer_text
from voxa.ui.face_quality import FaceQualitySwitch

SHA = "a" * 64


def _manifest() -> Manifest:
    return Manifest(
        version="1",
        download_bytes=5_000_000_000,
        install_bytes=9_000_000_000,
        parts=(Part(name="p0", url="https://x.com/p0", bytes=5_000_000_000, sha256=SHA),),
    )


def test_gb_one_decimal():
    assert gb(5_000_000_000) == "4.7"
    assert gb(9_000_000_000) == "8.4"
    assert gb(0) == "0.0"


def test_offer_text_returns_heading_and_body():
    heading, body = offer_text(_manifest())
    assert heading == "Install the High face?"
    assert "4.7 GB" in body
    assert "8.4 GB" in body
    assert "NVIDIA" in body


def test_set_loading_progress_custom_text():
    switch = FaceQualitySwitch()
    switch.set_loading_progress(40, "Downloading…")
    assert switch._status.get_text() == "Downloading…"
    assert switch._progress.percentage == 40
    assert switch._loading.get_visible()


def test_set_loading_progress_default_text():
    switch = FaceQualitySwitch()
    switch.set_loading_progress(50)
    assert switch._status.get_text() == "Loading high-resolution face…"


def test_go_back_to_medium_sets_prerendered():
    saved: list = []
    applied: list = []
    fake = SimpleNamespace(
        settings=SimpleNamespace(face_mode="live"),
        config_store=SimpleNamespace(save=lambda s: saved.append(s)),
        shell=SimpleNamespace(face_quality=FaceQualitySwitch()),
        _apply_face_mode=lambda: applied.append(True),
    )
    from voxa.window import MainWindow

    MainWindow._go_back_to_medium(fake)
    assert fake.settings.face_mode == "prerendered"
    assert saved == [fake.settings]
    assert applied == [True]


def test_on_shell_face_mode_selected_cancels_install():
    cancel = threading.Event()
    fake = SimpleNamespace(
        settings=SimpleNamespace(face_mode="live"),
        config_store=SimpleNamespace(save=lambda s: None),
        _face_install_cancel=cancel,
        _apply_face_mode=lambda: None,
    )
    from voxa.window import MainWindow

    MainWindow._on_shell_face_mode_selected(fake, "prerendered")
    assert cancel.is_set()
    assert fake.settings.face_mode == "prerendered"


def test_on_shell_face_mode_selected_no_cancel_for_live():
    cancel = threading.Event()
    fake = SimpleNamespace(
        settings=SimpleNamespace(face_mode="prerendered"),
        config_store=SimpleNamespace(save=lambda s: None),
        _face_install_cancel=cancel,
        _apply_face_mode=lambda: None,
    )
    from voxa.window import MainWindow

    MainWindow._on_shell_face_mode_selected(fake, "live")
    assert not cancel.is_set()
