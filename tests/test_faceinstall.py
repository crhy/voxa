import subprocess
import threading

import pytest

from voxa.faceinstall import (
    INSTALL_SCRIPT,
    Manifest,
    Part,
    gpu_problem,
    install,
    install_command,
    parse_manifest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _good_manifest_text() -> str:
    return (
        '{"version": "1", "download_bytes": 5000000000, "install_bytes": 9000000000,'
        ' "parts": [{"name": "face-server-1.tar.xz.part00",'
        ' "url": "https://example.com/p0", "bytes": 1900000000,'
        ' "sha256": "' + SHA_A + '"}],'
        ' "external": [{"path": "ditto-talkinghead/checkpoints/ditto_pytorch/aux_models/det_10g.onnx",'
        ' "url": "https://huggingface.co/x", "bytes": 16923827,'
        ' "sha256": "' + SHA_B + '"}]}'
    )


def test_parse_manifest_good():
    m = parse_manifest(_good_manifest_text())
    assert m.version == "1"
    assert m.download_bytes == 5000000000
    assert m.install_bytes == 9000000000
    assert len(m.parts) == 1
    assert m.parts[0].name == "face-server-1.tar.xz.part00"
    assert m.parts[0].sha256 == SHA_A
    assert len(m.external) == 1
    assert m.external[0].name == "ditto-talkinghead/checkpoints/ditto_pytorch/aux_models/det_10g.onnx"


def test_parse_bad_json():
    with pytest.raises(ValueError):
        parse_manifest("not json")


def test_parse_no_parts():
    with pytest.raises(ValueError):
        parse_manifest('{"version": "1"}')
    with pytest.raises(ValueError):
        parse_manifest('{"version": "1", "parts": []}')


def test_parse_part_name_slash():
    with pytest.raises(ValueError):
        parse_manifest('{"parts": [{"name": "a/b", "url": "https://x", "bytes": 1, "sha256": "' + SHA_A + '"}]}')


def test_parse_part_name_dot():
    with pytest.raises(ValueError):
        parse_manifest('{"parts": [{"name": ".hidden", "url": "https://x", "bytes": 1, "sha256": "' + SHA_A + '"}]}')


def test_parse_external_absolute():
    with pytest.raises(ValueError):
        parse_manifest(
            '{"parts": [{"name": "p0", "url": "https://x", "bytes": 1, "sha256": "' + SHA_A + '"}],'
            ' "external": [{"path": "/etc/passwd", "url": "https://x", "bytes": 1, "sha256": "' + SHA_B + '"}]}'
        )


def test_parse_external_dotdot():
    with pytest.raises(ValueError):
        parse_manifest(
            '{"parts": [{"name": "p0", "url": "https://x", "bytes": 1, "sha256": "' + SHA_A + '"}],'
            ' "external": [{"path": "a/../b", "url": "https://x", "bytes": 1, "sha256": "' + SHA_B + '"}]}'
        )


def test_parse_non_https_url():
    with pytest.raises(ValueError):
        parse_manifest('{"parts": [{"name": "p0", "url": "http://x", "bytes": 1, "sha256": "' + SHA_A + '"}]}')


def test_parse_bad_sha():
    with pytest.raises(ValueError):
        parse_manifest('{"parts": [{"name": "p0", "url": "https://x", "bytes": 1, "sha256": "abc"}]}')


def test_gpu_problem(monkeypatch):
    import voxa.hardware as hw

    monkeypatch.setattr(hw, "_nvidia_vram_gb", lambda: None)
    assert gpu_problem() == "The High face needs an NVIDIA graphics card with at least 6 GB of video memory."
    monkeypatch.setattr(hw, "_nvidia_vram_gb", lambda: 4.0)
    assert gpu_problem() == "The High face needs an NVIDIA graphics card with at least 6 GB of video memory."
    monkeypatch.setattr(hw, "_nvidia_vram_gb", lambda: 16.0)
    assert gpu_problem() == ""


def test_install_command_order():
    m = Manifest(
        version="7",
        download_bytes=0,
        install_bytes=0,
        parts=(
            Part(name="p0", url="https://x/0", bytes=100, sha256=SHA_A),
            Part(name="p1", url="https://x/1", bytes=200, sha256=SHA_B),
        ),
        external=(Part(name="sub/e.onnx", url="https://x/e", bytes=5, sha256="c" * 64),),
    )
    cmd = install_command(m)
    assert cmd[:3] == ["bash", "-c", INSTALL_SCRIPT]
    assert cmd[3] == "bash"
    assert cmd[4] == "7"
    assert cmd[5] == "2"
    tail = cmd[6:]
    assert tail == [
        "p0", "https://x/0", "100", SHA_A,
        "p1", "https://x/1", "200", SHA_B,
        "sub/e.onnx", "https://x/e", "5", "c" * 64,
    ]


class FakeProc:
    def __init__(self, lines):
        self.stdout = lines
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self):
        self.waited = True


def _popen(lines):
    def make(cmd, **kwargs):
        proc = FakeProc(lines)
        proc.cmd = cmd
        proc.kwargs = kwargs
        return proc

    return make


def test_install_done():
    m = Manifest(version="1", download_bytes=0, install_bytes=0, parts=(Part("p0", "https://x", 1, SHA_A),))
    calls = []
    proc_holder = {}

    def popen(cmd, **kwargs):
        proc = FakeProc(["PROGRESS 50 100\n", "UNPACKING\n", "DONE\n"])
        proc_holder["p"] = proc
        return proc

    ok, msg = install(m, lambda frac, text: calls.append((frac, text)), threading.Event(), popen=popen)
    assert ok is True
    assert msg == ""
    assert calls[0] == (0.95 * 50 / 100, "Downloading the High face… 50%")
    assert calls[1] == (0.96, "Unpacking the High face…")
    assert calls[2] == (1.0, "High face installed.")


def test_install_error_checksum():
    m = Manifest(version="1", download_bytes=0, install_bytes=0, parts=(Part("p0", "https://x", 1, SHA_A),))
    ok, msg = install(m, lambda *a: None, threading.Event(), popen=_popen(["ERROR checksum part00\n"]))
    assert ok is False
    assert msg == "The High face file part00 failed its checksum."


def test_install_no_done():
    m = Manifest(version="1", download_bytes=0, install_bytes=0, parts=(Part("p0", "https://x", 1, SHA_A),))
    ok, msg = install(m, lambda *a: None, threading.Event(), popen=_popen(["PROGRESS 10 100\n"]))
    assert ok is False
    assert msg == "The High face download stopped."


def test_script_syntax():
    subprocess.run(["bash", "-n", "-c", INSTALL_SCRIPT], check=True)


def test_install_refuses_when_the_disk_is_too_full():
    from voxa.faceinstall import install

    m = Manifest(version="1", download_bytes=5 * 10**9, install_bytes=9 * 10**9, parts=(Part("p0", "https://x", 1, SHA_A),))

    def no_popen(*_args, **_kwargs):
        raise AssertionError("must not start the download")

    ok, message = install(m, lambda *_a: None, threading.Event(), popen=no_popen, free_bytes=lambda: 3 * 10**9)
    assert not ok
    assert message == "The High face needs about 14 GB of free disk space, and there is 3 GB."


def test_cancel_stops_the_background_download(tmp_path):
    # The script's trap must take its background jobs down with it (a cancelled install must not keep downloading).
    from voxa.faceinstall import INSTALL_SCRIPT

    trap = next(line for line in INSTALL_SCRIPT.splitlines() if line.startswith("trap "))
    marker = tmp_path / "still-running"
    script = f"{trap}\n(sleep 2; touch {marker}) &\nwhile :; do sleep 0.2; done\n"  # polls like the real script
    proc = subprocess.Popen(["bash", "-c", script])
    import time

    time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=5)
    time.sleep(2.5)
    assert not marker.exists()
