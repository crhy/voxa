import subprocess

from voxa.agent import intents
from voxa.agent.tools import camera


def _fake(installed, ok=True, written=True):
    calls = []

    def run(command, stdin=None, check=True):
        calls.append(command)
        if command[0] == "date":
            return subprocess.CompletedProcess(command, 0, "2026-10-07 18-30-00\n", "")
        if command[:2] == ["sh", "-c"]:
            return subprocess.CompletedProcess(command, 0 if command[-1] in installed else 1, "", "")
        if command[0] == "test":
            return subprocess.CompletedProcess(command, 0 if written else 1, "", "")
        return subprocess.CompletedProcess(command, 0 if ok else 1, "", "")

    return run, calls


def _setup(monkeypatch, **kw):
    run, calls = _fake(**kw)
    monkeypatch.setattr(camera, "_run", run)
    monkeypatch.setattr(camera.filemanage, "_folder", lambda spoken: "/home/u/Pictures")
    monkeypatch.setattr(camera, "last_picture", "")
    return calls


def test_python_gstreamer_is_used_when_nothing_else_is_installed(monkeypatch):
    calls = _setup(monkeypatch, installed={"python3"})
    result = camera.take_picture({})
    assert result.ok and result.detail == "/home/u/Pictures/Photo 2026-10-07 18-30-00.jpg"
    grab = [c for c in calls if c[0] == "python3"][0]
    assert grab[-2:] == ["v4l2src", "/home/u/Pictures/Photo 2026-10-07 18-30-00.jpg"]
    assert camera.show_picture({}).ok
    assert calls[-1] == ["gio", "open", "/home/u/Pictures/Photo 2026-10-07 18-30-00.jpg"]


def test_ffmpeg_first_and_no_false_success(monkeypatch):
    calls = _setup(monkeypatch, installed={"ffmpeg", "python3"})
    assert camera.take_picture({}).ok
    assert not [c for c in calls if c[0] == "python3"]
    _setup(monkeypatch, installed={"python3"}, written=False)
    assert camera.take_picture({}).speech == camera.NO_CAMERA
    _setup(monkeypatch, installed=set())
    assert not camera.take_picture({}).ok
    assert not camera.show_picture({}).ok


def test_router():
    for phrase in ("take a picture", "Take a photo.", "take a selfie", "snap a photo", "take my picture"):
        assert intents.route(phrase).tool == "take_picture", phrase
    assert intents.route("show me the picture").tool == "show_picture"
    assert intents.route("take a screenshot").tool == "take_screenshot"
    assert intents.route("take a picture of my screen").tool == "take_screenshot"
    shown = intents.route("show me pictures of cats")
    assert shown is not None and shown.tool == "image_search"
