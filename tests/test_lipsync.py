"""Tests for voxa.lipsync (Rhubarb mouth-shape timelines)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from voxa import lipsync
from voxa.lipsync import (
    LipSyncError,
    MouthCue,
    analyze,
    ensure_valid_wav,
    parse_cues,
    shape_at,
)

# The 'Hi.' example from Rhubarb's own README (JSON export format).
HI_JSON = {
    "metadata": {"soundFile": "hi.wav", "duration": 0.47},
    "mouthCues": [
        {"start": 0.00, "end": 0.05, "value": "X"},
        {"start": 0.05, "end": 0.27, "value": "D"},
        {"start": 0.27, "end": 0.31, "value": "C"},
        {"start": 0.31, "end": 0.43, "value": "B"},
        {"start": 0.43, "end": 0.47, "value": "X"},
    ],
}


def test_parse_readme_example() -> None:
    cues = parse_cues(HI_JSON)
    assert cues == [
        MouthCue(0.00, 0.05, "X"),
        MouthCue(0.05, 0.27, "D"),
        MouthCue(0.27, 0.31, "C"),
        MouthCue(0.31, 0.43, "B"),
        MouthCue(0.43, 0.47, "X"),
    ]


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"mouthCues": []},
        {"mouthCues": [{"start": 0.0, "end": 0.1, "value": "Q"}]},
        {"mouthCues": [{"start": 0.2, "end": 0.1, "value": "A"}]},
        {"mouthCues": [{"start": "soon", "end": 0.1, "value": "A"}]},
        {"mouthCues": [{"start": 0.0, "end": 0.1}]},
        {"mouthCues": "nope"},
    ],
)
def test_parse_rejects_malformed(document: dict) -> None:
    with pytest.raises(LipSyncError):
        parse_cues(document)


def test_shape_at_boundaries() -> None:
    cues = parse_cues(HI_JSON)
    assert shape_at(cues, 0.0) == "X"
    assert shape_at(cues, 0.05) == "D"
    assert shape_at(cues, 0.30) == "C"
    assert shape_at(cues, 0.47) == "X"
    assert shape_at(cues, 99.0) == "X"
    assert shape_at([], 0.1) == "X"


def test_analyze_without_binary_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lipsync, "find_binary", lambda: None)
    assert analyze("whatever.wav", "hi") is None


def _fake_rhubarb(tmp_path: Path, payload: str, exit_code: int = 0) -> Path:
    """An executable impersonating rhubarb: records argv, prints payload."""
    script = tmp_path / "rhubarb"
    argv_log = tmp_path / "argv.txt"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"open({str(argv_log)!r}, 'w').write('\\n'.join(sys.argv[1:]))\n"
        "dialog = open(sys.argv[sys.argv.index('-d') + 1]).read()\n"
        f"open({str(tmp_path / 'dialog.txt')!r}, 'w').write(dialog)\n"
        f"sys.stdout.write(open({str(tmp_path / 'out.json')!r}).read())\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "out.json").write_text(payload)
    (tmp_path / "argv.txt").write_text("")
    os.environ["FAKE_ARGV"] = str(argv_log)
    return script


def test_analyze_runs_binary_and_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_rhubarb(tmp_path, json.dumps(HI_JSON))
    monkeypatch.setattr(lipsync, "find_binary", lambda: str(script))
    wav = tmp_path / "hi.wav"
    _write_wav(wav)

    cues = analyze(wav, "Hi.")
    assert cues is not None and cues[1] == MouthCue(0.05, 0.27, "D")

    argv = (tmp_path / "argv.txt").read_text().splitlines()
    assert argv[:4] == ["-f", "json", "--quiet", "-r"]
    assert argv[4] == "pocketSphinx"
    assert argv[5] == "-d"
    assert (tmp_path / "dialog.txt").read_text(encoding="utf-8") == "Hi."
    assert argv[7] == str(wav)


def test_analyze_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_rhubarb(tmp_path, "boom", exit_code=1)
    monkeypatch.setattr(lipsync, "find_binary", lambda: str(script))
    wav = tmp_path / "hi.wav"
    _write_wav(wav)
    with pytest.raises(LipSyncError):
        analyze(wav, "Hi.")


def test_analyze_bad_json_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_rhubarb(tmp_path, "not json")
    monkeypatch.setattr(lipsync, "find_binary", lambda: str(script))
    wav = tmp_path / "hi.wav"
    _write_wav(wav)
    with pytest.raises(LipSyncError):
        analyze(wav, "Hi.")


def _write_wav(path: Path, *, data_size: int | None = None) -> None:
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"\x00\x01" * 1000)
    if data_size is not None:  # corrupt the data-chunk length like espeak does
        raw = bytearray(path.read_bytes())
        raw[40:44] = data_size.to_bytes(4, "little")
        path.write_bytes(bytes(raw))


def test_valid_wav_passes_through(tmp_path: Path) -> None:
    wav = tmp_path / "ok.wav"
    _write_wav(wav)
    assert ensure_valid_wav(wav) == wav


def test_bogus_header_gets_fixed_copy(tmp_path: Path) -> None:
    import wave

    wav = tmp_path / "espeak.wav"
    _write_wav(wav, data_size=0xFFFFFFFF)
    fixed = ensure_valid_wav(wav)
    assert fixed != wav and fixed.name == "espeak.fixed.wav"
    with wave.open(str(fixed)) as parsed:
        assert parsed.getnframes() == 1000
    # Idempotent: the fixed copy validates as-is.
    assert ensure_valid_wav(fixed) == fixed


def test_non_wav_rejected(tmp_path: Path) -> None:
    text = tmp_path / "note.txt"
    text.write_text("nope")
    with pytest.raises(LipSyncError):
        ensure_valid_wav(text)


def test_analyze_fixes_header_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_rhubarb(tmp_path, json.dumps(HI_JSON))
    monkeypatch.setattr(lipsync, "find_binary", lambda: str(script))
    wav = tmp_path / "espeak.wav"
    _write_wav(wav, data_size=0xFFFFFFFF)
    assert analyze(wav, "Hi.") is not None
    argv = (tmp_path / "argv.txt").read_text().splitlines()
    assert argv[-1].endswith("espeak.fixed.wav")
