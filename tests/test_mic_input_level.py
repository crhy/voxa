"""Task v11e step 3b: Voxa calibrates her OWN microphone input gain (issue #72).

The pactl conversation is injected, so every rule is checked with no sound server:
only the chosen source's volume is ever set, the level is remembered per
microphone_id, a user change wins and stops auto-adjust, and it never adjusts while
she is speaking.
"""

from __future__ import annotations

import types

from voxa.config import Settings
from voxa.miclevel import (
    MAX_SOURCE_PERCENT,
    calibrate,
    note_user_change,
    parse_source_volume,
    target_percent,
    user_owns_level,
)

MIC_ID = "device.serial:Image+_Galyimage_Live_camera_HU12345679876543"
SOURCE = "alsa_input.usb-Image__Galyimage_Live_camera_HU12345679876543-02.analog-stereo"


class FakePactl:
    """A tiny stand-in for pactl: a fixed source volume it remembers, plus the set calls."""

    def __init__(self, volume: int = 66) -> None:
        self.volume = volume
        self.set_calls: list[tuple[str, int]] = []

    def __call__(self, cmd, *_args, **_kwargs):
        # host_command may wrap pactl; find the pactl sub-args.
        if "pactl" not in cmd:
            return types.SimpleNamespace(returncode=1, stdout="")
        args = cmd[cmd.index("pactl") + 1:]
        verb = " ".join(args[:2]) if args[:2] == ["list", "short"] else args[0]
        if verb == "list short":
            out = f"1\t{SOURCE}\talsa-sink\n2\talsa_output.pci.monitor\talsa-sink\n"
            return types.SimpleNamespace(returncode=0, stdout=out)
        if args[0] == "get-source-volume":
            return types.SimpleNamespace(returncode=0,
                                         stdout=f"Source volume: front-left: {self.volume * 655 / 100:.0f} / {self.volume}% / 0.72 (mutable)")
        if args[0] == "set-source-volume":
            self.set_calls.append((args[1], int(args[2].rstrip("%"))))
            self.volume = int(args[2].rstrip("%"))
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=1, stdout="")


def test_parse_source_volume() -> None:
    assert parse_source_volume("Source volume: front-left: 43253 / 66% / 0.72 (mutable)") == 66
    assert parse_source_volume("no percent here") is None


def test_target_percent_moves_toward_the_target_without_clipping() -> None:
    # quiet capture at 40 dB below target -> push it up (but never above the ceiling)
    assert target_percent(50, -55.0) == MAX_SOURCE_PERCENT or target_percent(50, -55.0) > 50
    # too hot and clipping -> pull it down
    assert target_percent(100, -2.0) < 100
    # about right -> stays near current
    assert abs(target_percent(80, -20.0) - 80) <= 4


def test_calibration_sets_only_that_source_and_remembers_the_level() -> None:
    runner = FakePactl(volume=40)
    levels: dict = {}
    result = calibrate(MIC_ID, levels, peak_dbfs=-45.0, source=SOURCE, runner=runner)
    assert result["changed"] is True
    assert result["source"] == SOURCE
    assert runner.set_calls, "must set a volume"
    # only ever her own source, never the output or another app
    assert all(name == SOURCE for name, _ in runner.set_calls)
    assert levels[MIC_ID]["user_picked"] is False
    assert levels[MIC_ID]["volume_percent"] > 40


def test_quiet_source_not_found_does_not_touch_anything() -> None:
    runner = FakePactl()
    levels: dict = {}
    result = calibrate("device.serial:not_there_zzz", levels, peak_dbfs=-45.0, runner=runner)
    assert result["changed"] is False
    assert result["reason"] == "microphone source not found"
    assert runner.set_calls == []


def test_never_adjusts_while_speaking() -> None:
    runner = FakePactl(volume=40)
    levels: dict = {}
    result = calibrate(MIC_ID, levels, peak_dbfs=-45.0, speaking=True, source=SOURCE, runner=runner)
    assert result["changed"] is False
    assert result["reason"] == "she is speaking"
    assert runner.set_calls == []


def test_user_change_wins_and_stops_auto_adjust() -> None:
    runner = FakePactl(volume=40)
    levels: dict = {}
    note_user_change(MIC_ID, levels, 70)
    assert user_owns_level(MIC_ID, levels)
    result = calibrate(MIC_ID, levels, peak_dbfs=-45.0, source=SOURCE, runner=runner)
    assert result["changed"] is False
    assert result["reason"] == "the user set this microphone's level"
    assert levels[MIC_ID]["volume_percent"] == 70
    assert runner.set_calls == []


def test_level_is_stored_per_microphone_id_and_survives_config_roundtrip(tmp_path) -> None:
    settings = Settings()
    settings.mic_levels[MIC_ID] = {"volume_percent": 90, "user_picked": True}
    settings.mic_levels["other-mic"] = {"volume_percent": 55, "user_picked": False}
    settings.normalized()
    assert settings.mic_levels["other-mic"]["volume_percent"] == 55

    from voxa.config import ConfigStore

    store = ConfigStore(tmp_path / "config.json")
    store.save(settings)
    loaded = store.load()
    assert loaded.mic_levels[MIC_ID] == {"volume_percent": 90, "user_picked": True}
    assert loaded.mic_levels["other-mic"]["user_picked"] is False


def test_a_bad_entry_is_dropped_not_the_whole_map() -> None:
    settings = Settings()
    settings.mic_levels = {MIC_ID: {"volume_percent": "not a number"},
                           "good": {"volume_percent": 50}}
    settings.normalized()
    assert settings.mic_levels == {"good": {"volume_percent": 50, "user_picked": False}}
