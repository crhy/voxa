"""Every GPU on the machine, and Voxa's brain on one the other workload is not using.

Four things are covered: listing the cards with an identity of their own,
choosing one for Voxa's small model, the exact environment or flags handed to
the server Voxa starts, and what the spoken status says about it.

There is no GPU on the machine these tests run on that they may trust, so
nothing here measures an offload: the probes are fed canned nvidia-smi and
rocm-smi output, the sysfs fallback is fed a throwaway directory, and the
assertions are about the command line and the sentences. No GPU tool, no
model, no network and no live audio are involved.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from voxa import engine, hardware
from voxa.config import ConfigStore, Settings
from voxa.engine import server_command
from voxa.hardware import GpuDevice, detect_gpu_devices, pick_brain_device, resolve_brain_device
from voxa.server import AiServerManager

# The machine crhy/voxa#113 is about: a big card held by another workload and
# a small free one Voxa can take.
TWO_CARDS = "0, NVIDIA GeForce RTX 4090, 24576, 22528, 97\n1, NVIDIA GeForce RTX 3060, 8192, 0, 0\n"


def _run(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _device(index: int, total: float | None = None, used: float | None = None, utilization: float | None = None) -> GpuDevice:
    return GpuDevice(
        index=index,
        vendor="nvidia",
        name=f"NVIDIA card {index}",
        total_vram_gb=total,
        used_vram_gb=used,
        utilization_percent=utilization,
        device_id=f"nvidia:{index}",
    )


def test_detect_gpu_devices_lists_both_cards_with_an_identity() -> None:
    with patch("subprocess.run", return_value=_run(TWO_CARDS)):
        devices = detect_gpu_devices()
    assert [device.device_id for device in devices] == ["nvidia:0", "nvidia:1"]
    assert devices[0].name == "NVIDIA GeForce RTX 4090"
    assert devices[0].total_vram_gb == 24.0
    assert devices[0].used_vram_gb == 22.0
    assert devices[0].utilization_percent == 97.0
    assert devices[1].total_vram_gb == 8.0


def test_detect_gpu_devices_asks_nvidia_smi_per_card_with_the_bundled_copy(tmp_path: Path) -> None:
    bundled = tmp_path / "nvml" / "nvidia-smi"
    bundled.parent.mkdir()
    bundled.touch()
    with (
        patch.object(hardware, "_BUNDLED_NVIDIA_SMI", bundled),
        patch("subprocess.run", return_value=_run(TWO_CARDS)) as run,
    ):
        assert len(detect_gpu_devices(sysfs_base=tmp_path)) == 2
    assert run.call_args.args[0][0] == str(bundled)
    assert "--query-gpu=index,name,memory.total,memory.used,utilization.gpu" in run.call_args.args[0]


def test_detect_gpu_devices_parses_rocm_smi_json() -> None:
    payload = (
        '{"/sys/dev/block/18:0:t-0": {"GPU": "gpu0", "VRAM Total Memory (B)": "8589934592",'
        ' "VRAM Total Used Memory (B)": "4294967296"},'
        ' "/sys/dev/block/18:0:t-1": {"GPU": "gpu1", "VRAM Total Memory (B)": "4294967296"}}'
    )
    with patch("subprocess.run", return_value=_run(payload)):
        devices = detect_gpu_devices()
    assert [device.device_id for device in devices] == ["amd:0", "amd:1"]
    assert devices[0].total_vram_gb == 8.0
    assert devices[0].used_vram_gb == 4.0


def test_detect_gpu_devices_reads_a_sysfs_tree_when_no_gpu_tool_exists(tmp_path: Path) -> None:
    for card, vendor, total, used in (("card0", "0x1002", 8 * 1024**3, 4 * 1024**3), ("card2", "0x10de", 24 * 1024**3, 0)):
        device_dir = tmp_path / card / "device"
        device_dir.mkdir(parents=True)
        (device_dir / "vendor").write_text(f"{vendor}\n", encoding="utf-8")
        (device_dir / "mem_info_vram_total").write_text(str(total), encoding="utf-8")
        (device_dir / "mem_info_vram_used").write_text(str(used), encoding="utf-8")
    # The display links /sys/class/drm points at are the same card again.
    (tmp_path / "card0-HDMI-1").mkdir()

    with patch("subprocess.run", side_effect=FileNotFoundError):
        devices = detect_gpu_devices(tmp_path)
    assert [device.device_id for device in devices] == ["amd:0", "nvidia:1"]
    assert devices[0].total_vram_gb == 8.0
    assert devices[0].used_vram_gb == 4.0


def test_detect_gpu_devices_lists_nothing_on_a_machine_without_cards(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_gpu_devices(tmp_path) == []


def test_pick_brain_device_skips_the_card_the_other_workload_holds() -> None:
    with patch("subprocess.run", return_value=_run(TWO_CARDS)):
        devices = detect_gpu_devices()
    chosen, reason = pick_brain_device(devices, reserved_gb=7.2, busy={"nvidia:0"})
    assert chosen is not None
    assert chosen.device_id == "nvidia:1"
    assert "GPU 1" in reason
    assert "GPU 0 is left to the other workload" in reason
    assert "0.8 of 8 GB free on it" in reason


def test_pick_brain_device_is_honest_when_the_machine_has_one_gpu() -> None:
    chosen, reason = pick_brain_device([_device(0, 16.0, 2.0, 40.0)], reserved_gb=7.2)
    assert chosen is not None
    assert chosen.device_id == "nvidia:0"
    assert reason == "this machine has one GPU, Voxa will share it"


def test_pick_brain_device_says_so_when_every_card_is_held() -> None:
    chosen, reason = pick_brain_device(
        [_device(0, 24.0, 22.0), _device(1, 8.0, 4.0)],
        reserved_gb=7.2,
        busy={"nvidia:0", "nvidia:1"},
    )
    assert chosen is None
    assert "already held by another workload" in reason


def test_pick_brain_device_says_so_when_no_card_has_room_left() -> None:
    chosen, reason = pick_brain_device([_device(0, 6.0, 3.0), _device(1, 4.0, 2.0)], reserved_gb=7.2)
    assert chosen is None
    assert "no card has room left" in reason


def test_pick_brain_device_prefers_the_card_with_the_most_room() -> None:
    chosen, _reason = pick_brain_device([_device(0, 8.0, 0.0), _device(1, 24.0, 2.0)], reserved_gb=2.0)
    assert chosen is not None
    assert chosen.device_id == "nvidia:1"


def test_pick_brain_device_admits_a_guess_when_memory_cannot_be_read() -> None:
    chosen, reason = pick_brain_device([_device(0), _device(1)], reserved_gb=0.0)
    assert chosen is not None
    assert chosen.device_id == "nvidia:0"
    assert "could not be read" in reason


def test_busy_devices_names_the_cards_that_already_hold_a_workload() -> None:
    with patch("subprocess.run", return_value=_run(TWO_CARDS)):
        devices = detect_gpu_devices()
    assert hardware.busy_devices(devices) == {"nvidia:0"}


def test_resolve_brain_device_keeps_a_chosen_card() -> None:
    with patch("subprocess.run", return_value=_run(TWO_CARDS)):
        devices = detect_gpu_devices()
    launch, reason = resolve_brain_device("nvidia:1", devices, reserved_gb=7.2)
    assert launch == "nvidia:1"
    assert "you chose GPU 1" in reason


def test_resolve_brain_device_falls_back_when_the_saved_card_is_gone() -> None:
    with patch("subprocess.run", return_value=_run(TWO_CARDS)):
        devices = detect_gpu_devices()
    launch, reason = resolve_brain_device("nvidia:5", devices, reserved_gb=7.2)
    assert launch == "auto"
    assert "the saved card nvidia:5 is not in this machine today" in reason


def test_resolve_brain_device_keeps_the_cpu_choice_and_says_why() -> None:
    launch, reason = resolve_brain_device("cpu", [_device(0, 24.0)], reserved_gb=7.2)
    assert launch == "cpu"
    assert reason == "you asked Voxa to keep her brain on the processor"


def test_resolve_brain_device_reports_the_automatic_choice_honestly() -> None:
    with patch("subprocess.run", return_value=_run(TWO_CARDS)):
        devices = detect_gpu_devices()
    launch, reason = resolve_brain_device("auto", devices, reserved_gb=7.2, busy=hardware.busy_devices(devices))
    assert launch == "auto"
    assert "GPU 1" in reason


def test_settings_keep_a_card_choice_and_repair_a_bad_one() -> None:
    assert Settings().gpu_device == "auto"
    assert Settings(gpu_device="nvidia:1").normalized().gpu_device == "nvidia:1"
    assert Settings(gpu_device="NVIDIA:0").normalized().gpu_device == "nvidia:0"
    assert Settings(gpu_device="gpu:2").normalized().gpu_device == "gpu:2"
    for stale in ("", "nvidia", "nvidia:one", "auto:1", "cpu:0", "nvidia:-1", "nvidia:1:2"):
        assert Settings(gpu_device=stale).normalized().gpu_device == "auto", stale


def test_saved_settings_round_trip_the_card_choice(tmp_path: Path) -> None:
    path = tmp_path / "voxa" / "config.json"
    store = ConfigStore(path)
    settings = Settings()
    settings.gpu_device = "nvidia:1"
    store.save(settings)
    assert store.load().gpu_device == "nvidia:1"


def _manager(settings: Settings, *, flatpak: bool = False):
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class FakePopen:
        def __call__(self, command, *, env=None, **_kwargs):
            calls.append((command, env))
            return None

    manager = AiServerManager(
        settings,
        popen=FakePopen(),
        is_reachable=lambda _url: False,
        flatpak=lambda: flatpak,
    )
    return manager, calls


def _command(settings: Settings, *, flatpak: bool = False) -> tuple[list[str], dict[str, str] | None]:
    with patch("voxa.hardware.detect_gpu_devices", return_value=[_device(0, 24.0, 22.0, 97.0), _device(1, 8.0, 0.0, 0.0)]):
        manager, calls = _manager(settings, flatpak=flatpak)
        assert manager.start() is True
    return calls[0]


def test_ollama_gets_the_chosen_card_through_cuda_visible_devices() -> None:
    command, env = _command(Settings(ai_backend="ollama", gpu_device="nvidia:1"))
    assert command == ["ollama", "serve"]
    assert env is not None
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"


def test_ollama_on_the_cpu_gets_no_gpu_flags_at_all() -> None:
    command, env = _command(Settings(ai_backend="ollama", gpu_device="cpu"))
    assert command == ["ollama", "serve"]
    assert env is not None
    assert env["OLLAMA_NUM_GPU"] == "0"
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_ollama_on_the_automatic_setting_sends_no_card() -> None:
    command, env = _command(Settings(ai_backend="ollama", gpu_device="auto"))
    assert command == ["ollama", "serve"]
    assert env is not None
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "OLLAMA_NUM_GPU" not in env


def test_a_stale_saved_card_launches_without_card_flags() -> None:
    command, env = _command(Settings(ai_backend="ollama", gpu_device="nvidia:7"))
    assert env is not None
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_flatpak_bakes_the_card_into_the_host_command_line() -> None:
    command, env = _command(Settings(ai_backend="ollama", gpu_device="nvidia:1"), flatpak=True)
    assert command[:4] == ["flatpak-spawn", "--host", "--directory=/", "--watch-bus"]
    assert "env" in command
    assert "CUDA_VISIBLE_DEVICES=1" in command
    assert "OLLAMA_HOST=127.0.0.1:11434" in command
    assert env is not None and env["CUDA_VISIBLE_DEVICES"] == "1"


def test_llama_server_gets_device_and_main_gpu_flags() -> None:
    command, env = _command(Settings(ai_backend="llamacpp", llamacpp_model="m.gguf", gpu_device="nvidia:1"))
    assert command == [
        "llama-server",
        "--model",
        "m.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8781",
        "--device",
        "cuda",
        "--main-gpu",
        "1",
    ]
    assert env is None


def test_llama_server_on_the_cpu_gets_no_gpu_flags_at_all() -> None:
    command, _env = _command(Settings(ai_backend="llamacpp", llamacpp_model="m.gguf", gpu_device="cpu"))
    assert "--device" not in command
    assert "--main-gpu" not in command
    assert command[:4] == ["llama-server", "--model", "m.gguf", "--host"]


def test_server_command_on_the_cpu_keeps_every_layer_on_the_processor() -> None:
    command = server_command("m.gguf", "127.0.0.1", 8781, server="/app/bin/llama-server", device="cpu")
    assert command[command.index("-ngl") + 1] == "0"
    assert "--device" not in command
    assert "--main-gpu" not in command


def test_server_command_keeps_all_layers_on_the_gpu() -> None:
    command = server_command("m.gguf", "127.0.0.1", 8781, server="/app/bin/llama-server", device="amd:0")
    assert command[:3] == ["/app/bin/llama-server", "--model", "m.gguf"]
    assert command[command.index("-ngl") + 1] == "99"
    assert command[-4:] == ["--device", "hip", "--main-gpu", "0"]


def test_server_command_without_a_choice_is_the_old_command() -> None:
    assert server_command("m.gguf", "127.0.0.1", 8781, server="/app/bin/llama-server") == [
        "/app/bin/llama-server",
        "--model",
        "m.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8781",
        "-ngl",
        "99",
        "-c",
        "8192",
        "--jinja",
    ]


def test_an_unknown_vendor_pins_only_the_index() -> None:
    assert engine.device_flags("gpu:1") == ["--main-gpu", "1"]
    assert engine.device_flags("auto") == []
    assert engine.device_flags("cpu") == []


def test_the_single_value_probes_still_read_the_largest_card() -> None:
    with patch("subprocess.run", return_value=_run(TWO_CARDS)):
        assert hardware.detect_gpu_vram_gb() == 24.0
        usage = hardware.sample_gpu_usage()
    assert usage is not None
    assert usage.utilization_percent == 97.0
    assert usage.memory_used_gb == 22.0
    assert usage.memory_total_gb == 24.0


def test_detect_graphics_adapter_still_reads_sysfs_alone(tmp_path: Path) -> None:
    vendor = tmp_path / "card0" / "device" / "vendor"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("0x8086\n", encoding="utf-8")
    assert hardware.detect_graphics_adapter(tmp_path) == "Intel integrated GPU"


class FakeStore:
    """Stands in for the settings file so a tool test never touches it."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.saved: Settings | None = None

    def load(self) -> Settings:
        return self._settings

    def save(self, settings: Settings) -> None:
        self.saved = settings


def _tools(monkeypatch, settings: Settings):
    from voxa.agent.tools import system

    store = FakeStore(settings)
    monkeypatch.setattr(system, "ConfigStore", lambda: store)
    monkeypatch.setattr(
        hardware,
        "detect_gpu_devices",
        lambda sysfs_base=Path("/sys/class/drm"): [_device(0, 24.0, 22.0, 97.0), _device(1, 8.0, 0.0, 0.0)],
    )
    return system, store


def test_gpu_status_names_the_card_voxa_takes(monkeypatch) -> None:
    system, store = _tools(monkeypatch, Settings())
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    result = system.gpu_status({})
    assert result.ok
    assert "GPU 1" in result.speech
    assert "46 degrees" in result.speech
    assert store.saved is None  # asking is not changing


def test_gpu_choose_moves_voxa_to_the_free_card(monkeypatch) -> None:
    system, store = _tools(monkeypatch, Settings())
    result = system.gpu_choose({"move": "other"})
    assert result.ok
    assert store.saved is not None
    assert store.saved.gpu_device == "nvidia:1"
    assert "GPU 1" in result.speech
    assert "when her AI server starts next" in result.speech


def test_gpu_choose_can_keep_voxa_on_the_cpu(monkeypatch) -> None:
    system, store = _tools(monkeypatch, Settings(gpu_device="nvidia:1"))
    result = system.gpu_choose({"move": "cpu"})
    assert result.ok
    assert store.saved is not None
    assert store.saved.gpu_device == "cpu"
    assert "processor" in result.speech


def test_gpu_choose_admits_when_there_is_no_other_card(monkeypatch) -> None:
    system, store = _tools(monkeypatch, Settings(gpu_device="nvidia:1"))
    monkeypatch.setattr(hardware, "detect_gpu_devices", lambda sysfs_base=Path("/sys/class/drm"): [_device(1, 8.0, 0.0, 0.0)])
    result = system.gpu_choose({"move": "other"})
    assert result.ok
    assert "one GPU" in result.speech
    assert store.saved is not None
    assert store.saved.gpu_device == "nvidia:1"


def test_gpu_choose_says_which_card_it_names(monkeypatch) -> None:
    system, store = _tools(monkeypatch, Settings())
    result = system.gpu_choose({"move": "nvidia:0"})
    assert result.ok
    assert store.saved is not None
    assert store.saved.gpu_device == "nvidia:0"


@pytest.mark.parametrize(
    ("spoken", "tool"),
    [
        ("which gpu is voxa using", "gpu_status"),
        ("what graphics card is voxa's brain on", "gpu_status"),
        ("put voxa on the other gpu", "gpu_choose"),
        ("move voxa to another graphics card", "gpu_choose"),
        ("keep voxa on the cpu", "gpu_choose"),
        ("gpu temperature", "gpu_status"),
        ("how hot is my graphics card", "gpu_status"),
    ],
)
def test_the_routes_are_spoken(spoken: str, tool: str) -> None:
    from voxa.agent.intents import route

    assert route(spoken).tool == tool
