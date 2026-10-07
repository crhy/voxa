"""Echo cancellation: the pactl conversation, with a scripted fake pactl."""

from __future__ import annotations

import types

from voxa.echo import SINK_NAME, SOURCE_NAME, EchoCanceller, parse_module_id, parse_sink_inputs


class FakePactl:
    def __init__(self, modules: str = "", default_sink: str = "speakers", load_ok: bool = True) -> None:
        self.modules, self.default_sink, self.load_ok = modules, default_sink, load_ok
        self.calls: list[list[str]] = []

    def __call__(self, argv, **_kwargs):
        args = argv[argv.index("pactl") + 1:]
        self.calls.append(args)
        out, code = "", 0
        if args[:3] == ["list", "short", "modules"]:
            out = self.modules
        elif args == ["get-default-sink"]:
            out = self.default_sink
        elif args[0] == "load-module":
            out, code = ("31", 0) if self.load_ok else ("Failure: Module initialization failed", 1)
        elif args[:3] == ["list", "short", "sink-inputs"]:
            out = "7\t1\t55\tprotocol-native.c\ts16le 2ch 44100Hz\n9\t1\t60\tprotocol-native.c\ts16le 2ch 48000Hz"
        elif args[0] == "set-default-sink":
            self.default_sink = args[1]
        return types.SimpleNamespace(returncode=code, stdout=out, stderr="")


def test_parse_helpers() -> None:
    listing = f"3\tmodule-alsa-card\tx\n24\tmodule-echo-cancel\taec_method=webrtc source_name={SOURCE_NAME} sink_name={SINK_NAME}"
    assert parse_module_id(listing) == "24"
    assert parse_module_id("24\tmodule-echo-cancel\tsource_name=someone_elses") is None
    assert parse_sink_inputs("7\t1\t55\tx\n\n9\t1\t60\ty") == ["7", "9"]


def test_enable_routes_everything_through_the_cancelling_output_and_disable_restores() -> None:
    pactl = FakePactl()
    echo = EchoCanceller(runner=pactl)
    assert echo.enable() is True and echo.active
    load = next(call for call in pactl.calls if call[0] == "load-module")
    assert f"source_name={SOURCE_NAME}" in load and "sink_master=speakers" in load
    assert any(arg.startswith('aec_args="') and arg.endswith('"') for arg in load)  # quoted: the value has spaces
    assert ["set-default-sink", SINK_NAME] in pactl.calls
    assert ["move-sink-input", "7", SINK_NAME] in pactl.calls and ["move-sink-input", "9", SINK_NAME] in pactl.calls

    echo.disable()
    assert pactl.default_sink == "speakers" and not echo.active
    assert ["move-sink-input", "7", "speakers"] in pactl.calls
    assert pactl.calls[-1] == ["unload-module", "31"]


def test_enable_twice_loads_once() -> None:
    pactl = FakePactl()
    echo = EchoCanceller(runner=pactl)
    assert echo.enable() and echo.enable()
    assert sum(1 for call in pactl.calls if call[0] == "load-module") == 1


def test_a_module_left_by_a_crashed_run_is_replaced_by_a_fresh_one() -> None:
    """A leftover may be tied to the wrong microphone or speakers: it is removed and rebuilt, never adopted."""
    leftover = f"24\tmodule-echo-cancel\tsource_name={SOURCE_NAME} sink_name={SINK_NAME}"
    pactl = FakePactl(modules=leftover, default_sink="speakers")
    echo = EchoCanceller(runner=pactl)
    assert echo.enable(source_master="blue_mic") is True
    kinds = [call[0] for call in pactl.calls]
    assert kinds.index("unload-module") < kinds.index("load-module")
    load = next(call for call in pactl.calls if call[0] == "load-module")
    assert "source_master=blue_mic" in load and "sink_master=speakers" in load
    echo.disable()
    assert pactl.calls[-1] == ["unload-module", "31"]


def test_the_chosen_microphone_is_found_by_its_serial() -> None:
    import types

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


def test_failures_leave_the_normal_microphone_in_use() -> None:
    assert EchoCanceller(runner=FakePactl(load_ok=False)).enable() is False

    def broken(_argv, **_kwargs):
        raise OSError("no pactl")

    echo = EchoCanceller(runner=broken)
    assert echo.enable() is False
    echo.disable()  # nothing to undo, must not raise
