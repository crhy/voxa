"""No echo device: leftover removal and the config rules, with a fake pactl recording every argv."""

from __future__ import annotations

import types
from pathlib import Path

from voxa.config import ECHO_MODES, Settings
from voxa.echo import SINK_NAME, SOURCE_NAME, remove_leftover_devices


def make_runner(modules: str = "", default_sink: str = "speakers", sinks: str = "", sink_inputs: str = "", code: int = 0):
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        out = ""
        if "get-default-sink" in argv:
            out = default_sink
        elif "modules" in argv:
            out = modules
        elif "sinks" in argv:
            out = sinks
        elif "sink-inputs" in argv:
            out = sink_inputs
        return types.SimpleNamespace(returncode=code, stdout=out, stderr="")

    runner.calls = calls
    return runner


def recorded(calls: list[list[str]], name: str) -> list[list[str]]:
    return [argv[argv.index(name):] for argv in calls if name in argv]


def test_no_leftover_module_touches_nothing() -> None:
    runner = make_runner(modules="3\tmodule-alsa-card\tx")
    assert remove_leftover_devices(runner) is False
    for name in ("set-default-sink", "move-sink-input", "unload-module", "load-module"):
        assert not recorded(runner.calls, name)


def test_leftover_device_is_removed_and_the_output_handed_back() -> None:
    master = "alsa_output.usb-Blue.analog-stereo"
    modules = (
        f"403\tmodule-echo-cancel\taec_method=webrtc source_name={SOURCE_NAME} "
        f"sink_name={SINK_NAME} sink_master={master}"
    )
    sinks = f"146\t{master}\tx\n380\t{SINK_NAME}\tx"
    sink_inputs = "3213\t380\tx\n3234\t146\tx"
    runner = make_runner(modules=modules, default_sink=SINK_NAME, sinks=sinks, sink_inputs=sink_inputs)
    assert remove_leftover_devices(runner) is True
    assert ["set-default-sink", master] in recorded(runner.calls, "set-default-sink")
    moves = recorded(runner.calls, "move-sink-input")
    assert ["move-sink-input", "3213", master] in moves
    assert not any("3234" in move for move in moves)
    assert ["unload-module", "403"] in recorded(runner.calls, "unload-module")


def test_default_already_the_real_output_still_unloads() -> None:
    master = "alsa_output.usb-Blue.analog-stereo"
    modules = f"403\tmodule-echo-cancel\tsource_name={SOURCE_NAME} sink_master={master}"
    runner = make_runner(modules=modules, default_sink=master)
    assert remove_leftover_devices(runner) is True
    assert not recorded(runner.calls, "set-default-sink")
    assert ["unload-module", "403"] in recorded(runner.calls, "unload-module")


def test_pactl_failing_is_false_and_never_raises() -> None:
    assert remove_leftover_devices(make_runner(code=1)) is False

    def broken(_argv, **_kwargs):
        raise OSError("no pactl")

    assert remove_leftover_devices(broken) is False


def test_system_mode_is_gone_from_the_config() -> None:
    assert "system" not in ECHO_MODES
    assert Settings().echo_mode == "voxa"
    assert Settings(echo_mode="system").normalized().echo_mode == "voxa"
    assert Settings(echo_cancel=False).normalized().echo_mode == "off"


def test_the_sources_never_create_a_device() -> None:
    echo_text = Path("voxa/echo.py").read_text().replace("unload-module", "")
    assert "load-module" not in echo_text
    assert "set-default-sink" not in Path("voxa/window.py").read_text()
