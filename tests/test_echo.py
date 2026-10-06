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


def test_a_module_left_by_a_crashed_run_is_adopted_and_cleaned_up() -> None:
    leftover = f"24\tmodule-echo-cancel\tsource_name={SOURCE_NAME} sink_name={SINK_NAME}"
    pactl = FakePactl(modules=leftover, default_sink=SINK_NAME)
    echo = EchoCanceller(runner=pactl)
    assert echo.enable() is True
    assert not any(call[0] == "load-module" for call in pactl.calls)
    echo.disable()
    assert pactl.calls[-1] == ["unload-module", "24"]


def test_failures_leave_the_normal_microphone_in_use() -> None:
    assert EchoCanceller(runner=FakePactl(load_ok=False)).enable() is False

    def broken(_argv, **_kwargs):
        raise OSError("no pactl")

    echo = EchoCanceller(runner=broken)
    assert echo.enable() is False
    echo.disable()  # nothing to undo, must not raise
