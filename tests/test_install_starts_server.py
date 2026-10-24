"""#112, #71: after Voxa installs Ollama it must start the server, which failed at launch while Ollama was missing."""

from types import SimpleNamespace

import pytest

window = pytest.importorskip("voxa.window")


def _fake(backend: str, welcome_followup: bool) -> SimpleNamespace:
    calls: list[str] = []
    fake = SimpleNamespace(
        settings=SimpleNamespace(ai_backend=backend),
        _installing=True,
        _welcome_followup=welcome_followup,
        _toast=lambda _message: None,
        _restart_ai_server_async=lambda: calls.append("restart"),
        _refresh_ollama_models=lambda: calls.append("refresh"),
        _download_welcome_model=lambda: calls.append("download"),
        calls=calls,
    )
    return fake


def _finish(fake: SimpleNamespace, success: bool) -> None:
    dialog = SimpleNamespace(close=lambda: None)
    window.MainWindow._on_install_finished(fake, dialog, success, "done")


def test_a_finished_ollama_install_starts_the_server_before_the_first_download() -> None:
    fake = _fake("ollama", welcome_followup=True)
    _finish(fake, True)
    assert fake.calls == ["restart", "refresh", "download"]
    assert fake._installing is False


def test_a_failed_install_starts_nothing() -> None:
    fake = _fake("ollama", welcome_followup=True)
    _finish(fake, False)
    assert fake.calls == []


def test_another_backend_is_not_restarted() -> None:
    fake = _fake("llamacpp", welcome_followup=False)
    _finish(fake, True)
    assert fake.calls == ["refresh"]
