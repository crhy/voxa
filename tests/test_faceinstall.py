import hashlib
import os
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


def _argv_before_issue_117(m: Manifest) -> list[str]:
    """How install_command built the argv before the torrent work: the regression guard."""
    args = [m.version, str(len(m.parts))]
    for p in m.parts:
        args += [p.name, p.url, str(p.bytes), p.sha256]
    for e in m.external:
        args += [e.name, e.url, str(e.bytes), e.sha256]
    return ["bash", "-c", INSTALL_SCRIPT, "bash", *args]


def test_install_command_is_unchanged_when_the_release_publishes_no_torrent():
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
    assert install_command(m) == _argv_before_issue_117(m)
    assert install_command(m, None) == _argv_before_issue_117(m)


def test_install_command_adds_only_web_seed_urls_when_a_torrent_is_published():
    from voxa.torrentsource import read_torrent

    def encode(value: object) -> bytes:
        if isinstance(value, int):
            return b"i%de" % value
        if isinstance(value, bytes):
            return b"%d:%s" % (len(value), value)
        if isinstance(value, list):
            return b"l" + b"".join(encode(item) for item in value) + b"e"
        return b"d" + b"".join(encode(k) + encode(v) for k, v in sorted(value.items())) + b"e"

    seed = "https://github.com/crhy/voxa/releases/download/face-server-1/p0"
    meta = read_torrent(encode({
        b"info": {b"length": 100, b"piece length": 100},
        b"name": b"p0",
        b"url-list": [seed.encode()],
    }))
    m = Manifest(
        version="7",
        download_bytes=0,
        install_bytes=0,
        parts=(Part(name="p0", url="https://x/0", bytes=100, sha256=SHA_A),),
    )
    cmd = install_command(m, meta)
    assert cmd[:6] == _argv_before_issue_117(m)[:6]
    assert cmd[6:] == ["p0", "https://x/0", "100", SHA_A, seed]


def test_the_checksum_checks_survive_the_web_seed_wiring():
    # A peer or a web seed may only make the download faster, never different: the script must
    # still check the bytes before resuming and after downloading.
    assert 'sha256sum "$f"' in INSTALL_SCRIPT
    assert 'sha256sum "$NEW/$epath"' in INSTALL_SCRIPT
    assert 'echo "ERROR checksum $name"' in INSTALL_SCRIPT
    assert 'echo "ERROR checksum $epath"' in INSTALL_SCRIPT


def test_the_script_hands_the_web_seeds_to_curl_and_still_refuses_wrong_bytes(tmp_path):
    # The published URL fails and the web seed serves the file: the install succeeds. Then the
    # web seed serves bytes that are not the release's: the checksum must refuse them.
    home = tmp_path / "home"
    home.mkdir()
    stub = tmp_path / "bin"
    stub.mkdir()
    served = tmp_path / "served"
    served.mkdir()
    payload = _tiny_tar_xz()
    (served / "p0").write_bytes(payload)
    log = tmp_path / "curl-urls.log"
    (stub / "curl").write_text(
        'out=""; urls=()\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then shift; out=$1; shift; continue; fi\n'
        '  if [ "${1#-}" != "$1" ]; then shift; continue; fi\n'
        '  urls+=("$1"); shift\n'
        'done\n'
        'for u in "${urls[@]}"; do printf \'%s\\n\' "$u" >> "' + str(log) + '"; done\n'
        'for u in "${urls[@]}"; do\n'
        '  case "$u" in *missing*) continue;; esac\n'
        '  f="' + str(served) + '/${u#https://seed/}"\n'
        '  [ -f "$f" ] || continue\n'
        '  cat "$f" > "$out"; exit 0\n'
        'done\n'
        'exit 22\n'
    )
    (stub / "curl").chmod(0o755)
    sha = hashlib.sha256(payload).hexdigest()
    env = dict(os.environ, HOME=str(home), PATH=f"{stub}:{os.environ['PATH']}")
    argv = [
        "bash", "-c", INSTALL_SCRIPT, "bash", "1", "1",
        "p0", "https://release/missing", str(len(payload)), sha,
        "https://seed/p0",
    ]
    result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60)
    assert result.stdout.split()[-1] == "DONE"
    assert log.read_text().split() == ["https://release/missing", "https://seed/p0"]

    (served / "p0").write_bytes(b"someone else's bytes")
    result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60)
    assert "ERROR checksum p0" in result.stdout
    assert not (home / ".local/share/voxa/face-server.download/p0").exists()


def _tiny_tar_xz() -> bytes:
    """A valid tar.xz, so the script's unpack step has something real to unpack."""
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        info = tarfile.TarInfo("hello.txt")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"hello"))
    return buf.getvalue()
