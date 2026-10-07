"""Echo helpers: the pactl conversation, with a scripted fake pactl."""

from __future__ import annotations

import types

from voxa.echo import (
    SINK_NAME,
    SOURCE_NAME,
    default_monitor_source,
    parse_module_id,
    parse_sink_inputs,
    parse_sink_master,
)


def make_runner(modules: str = "", default_sink: str = "speakers", code: int = 0):
    def runner(argv, **_kwargs):
        out = ""
        if "get-default-sink" in argv:
            out = default_sink
        elif "modules" in argv:
            out = modules
        return types.SimpleNamespace(returncode=code, stdout=out, stderr="")

    return runner


def test_parse_helpers() -> None:
    listing = f"3\tmodule-alsa-card\tx\n24\tmodule-echo-cancel\taec_method=webrtc source_name={SOURCE_NAME} sink_name={SINK_NAME} sink_master=alsa_output.real"
    assert parse_module_id(listing) == "24"
    assert parse_module_id("24\tmodule-echo-cancel\tsource_name=someone_elses") is None
    assert parse_sink_inputs("7\t1\t55\tx\n\n9\t1\t60\ty") == ["7", "9"]


def test_parse_sink_master() -> None:
    listing = f"24\tmodule-echo-cancel\tsource_name={SOURCE_NAME} sink_master=alsa_output.usb-Blue.analog-stereo"
    assert parse_sink_master(listing) == "alsa_output.usb-Blue.analog-stereo"
    assert parse_sink_master(f"24\tmodule-echo-cancel\tsource_name={SOURCE_NAME}") is None
    assert parse_sink_master("24\tmodule-echo-cancel\tsource_name=someone_elses sink_master=x") is None


def test_default_monitor_source_uses_the_real_output() -> None:
    assert default_monitor_source(make_runner(default_sink="alsa_output.real")) == "alsa_output.real.monitor"
    assert default_monitor_source(make_runner(default_sink=SINK_NAME)) is None


def test_default_monitor_source_none_when_pactl_fails() -> None:
    assert default_monitor_source(make_runner(code=1)) is None

    def broken(_argv, **_kwargs):
        raise OSError("no pactl")

    assert default_monitor_source(broken) is None


def test_the_chosen_microphone_is_found_by_its_serial() -> None:
    from voxa.echo import source_for

    listing = (
        "2\talsa_input.usb-Image__Galyimage_Live_camera_HU1-02.analog-stereo\tx\n"
        "9\talsa_input.usb-Generic_Blue_Microphones_2111-00.analog-stereo\tx\n"
        "8\talsa_output.pci.analog-stereo.monitor\tx\n"
        f"7\t{SOURCE_NAME}\tx"
    )

    def runner(argv, **_kwargs):
        return types.SimpleNamespace(returncode=0, stdout=listing, stderr="")

    assert source_for("device.serial:Generic_Blue_Microphones_2111", runner).startswith("alsa_input.usb-Generic_Blue")
    assert source_for("device.serial:Image+_Galyimage_Live_camera_HU1", runner).startswith("alsa_input.usb-Image__Galy")
    assert source_for("device.serial:Nothing_Like_It", runner) is None and source_for("", runner) is None
