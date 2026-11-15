"""Picking which graphics card the local AI runs on (crhy/voxa#9).

Three things are covered: listing the cards the machine has, storing the choice
("auto", "cpu", "gpu:<index>"), and the exact environment or flags handed to the
server Voxa starts.

There is no graphics card on the machine these tests run on, so nothing here
measures an offload: the probes are fed canned nvidia-smi/rocm-smi output, the
sysfs fallback is fed a throwaway directory, and the assertions are about the
command line. No GPU tool, no model, no network and no live audio are involved.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from voxa import engine, hardware
from voxa.config import ConfigStore, Settings
from voxa.engine import server_command
from voxa.hardware import Gpu, detect_gpus
from voxa.server import AiServerManager


def _run(stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def test_detect_gpus_parses_two_nvidia_cards() -> None:
    with patch("subprocess.run", return_value=_run(
        "0, NVIDIA GeForce RTX 4090, 24576\n1, NVIDIA GeForce RTX 3060, 8192\n"
    )):
        assert detect_gpus() == (
            Gpu(index=0, label="GPU 0: NVIDIA GeForce RTX 4090", vram_gb=24.0),
            Gpu(index=1, label="GPU 1: NVIDIA GeForce RTX 3060", vram_gb=8.0),
        )


def test_detect_gpus_asks_nvidia_smi_per_card_and_uses_the_bundled_copy(tmp_path: Path) -> None:
    # Inside the Flatpak the host's nvidia-smi is not on PATH, so the copy
    # bundled at /app/lib/nvml has to be the one that runs (crhy/voxa#9's
    # "already detected for the model suggestions" relies on this too).
    bundled = tmp_path / "nvml" / "nvidia-smi"
    bundled.parent.mkdir()
    bundled.touch()
    with (
        patch.object(hardware, "_BUNDLED_NVIDIA_SMI", bundled),
        patch("subprocess.run", return_value=_run("0, NVIDIA GeForce RTX 4090, 24576\n")) as run,
    ):
        assert detect_gpus(sysfs_base=tmp_path) == (
            Gpu(index=0, label="GPU 0: NVIDIA GeForce RTX 4090", vram_gb=24.0),
        )
    assert run.call_args.args[0][0] == str(bundled)
    assert "--query-gpu=index,name,memory.total" in run.call_args.args[0]


def test_detect_gpus_parses_rocm_smi_json() -> None:
    payload = {
        "/sys/dev/block/18:0:t-0": {
            "GPU": "gpu0",
            "VRAM Total Memory (B)": "8589934592",
            "VRAM Total Used Memory (B)": "0",
        },
        "/sys/dev/block/18:0:t-1": {"GPU": "gpu1", "VRAM Total Memory (B)": "4294967296"},
    }
    with patch("subprocess.run", return_value=_run(json.dumps(payload))):
        assert detect_gpus() == (
            Gpu(index=0, label="GPU 0: AMD GPU", vram_gb=8.0),
            Gpu(index=1, label="GPU 1: AMD GPU", vram_gb=4.0),
        )


def test_detect_gpus_reads_sysfs_cards_when_no_tool_exists(tmp_path: Path) -> None:
    for card, vram in (("card0", 8 * 1024**3), ("card2", 4 * 1024**3)):
        device = tmp_path / card / "device"
        device.mkdir(parents=True)
        (device / "mem_info_vram_total").write_text(str(vram), encoding="utf-8")
        (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_gpus(tmp_path) == (
            Gpu(index=0, label="GPU 0: AMD GPU", vram_gb=8.0),
            Gpu(index=1, label="GPU 1: AMD GPU", vram_gb=4.0),
        )


def test_detect_gpus_returns_nothing_when_no_tool_and_no_sysfs(tmp_path: Path) -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_gpus(tmp_path) == ()


def test_detect_gpus_skips_a_malformed_row_and_keeps_the_good_one() -> None:
    with patch("subprocess.run", return_value=_run(
        "not,a,card\n1, NVIDIA GeForce RTX 3060, 8192\n"
    )):
        assert detect_gpus() == (Gpu(index=1, label="GPU 1: NVIDIA GeForce RTX 3060", vram_gb=8.0),)


def test_detect_gpus_keeps_a_card_whose_memory_is_unreadable() -> None:
    # nvidia-smi prints "[NotSupported]" for some integrated or busy cards; the
    # card is still pickable, it just has no size to show.
    with patch("subprocess.run", return_value=_run("0, NVIDIA GeForce RTX 4090, [NotSupported]\n")):
        assert detect_gpus() == (Gpu(index=0, label="GPU 0: NVIDIA GeForce RTX 4090", vram_gb=None),)


def test_detect_gpus_never_raises_when_the_tool_fails() -> None:
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
        assert detect_gpus(Path("/nonexistent-drm")) == ()


def test_settings_keep_the_three_gpu_choice_forms() -> None:
    assert Settings().gpu_choice == "auto"
    assert Settings(gpu_choice="auto").normalized().gpu_choice == "auto"
    assert Settings(gpu_choice="cpu").normalized().gpu_choice == "cpu"
    assert Settings(gpu_choice="gpu:1").normalized().gpu_choice == "gpu:1"
    assert Settings(gpu_choice=" GPU:2 ").normalized().gpu_choice == "gpu:2"


def test_settings_repair_a_junk_gpu_choice_to_auto() -> None:
    for junk in ("", "card 0", "gpu:", "gpu:one", "gpu:-1", "GPU 1", None):
        assert Settings(gpu_choice=junk).normalized().gpu_choice == "auto", junk


def test_a_saved_card_choice_survives_the_config_round_trip(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    store.save(Settings(ai_backend="ollama", gpu_choice="gpu:1"))
    assert store.load().gpu_choice == "gpu:1"


def test_an_older_config_without_the_key_defaults_to_auto(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"ollama_model": "qwen2.5:0.5b", "ai_backend": "ollama"}), encoding="utf-8")
    assert ConfigStore(path).load().gpu_choice == "auto"


def _manager(settings: Settings, *, flatpak: bool = False) -> AiServerManager:
    return AiServerManager(
        settings,
        popen=lambda *args, **kwargs: None,
        is_reachable=lambda _url: False,
        flatpak=lambda: flatpak,
    )


def test_ollama_automatic_sends_no_card_variable() -> None:
    _command, env = _manager(Settings(ai_backend="ollama"))._command("127.0.0.1", 11434)
    assert env == {"OLLAMA_HOST": "127.0.0.1:11434"}
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_ollama_chosen_card_passes_its_index_through_the_environment() -> None:
    _command, env = _manager(Settings(ai_backend="ollama", gpu_choice="gpu:1"))._command("127.0.0.1", 11434)
    assert _command == ["ollama", "serve"]
    assert env == {"OLLAMA_HOST": "127.0.0.1:11434", "CUDA_VISIBLE_DEVICES": "1"}


def test_ollama_processor_only_hides_every_card() -> None:
    _command, env = _manager(Settings(ai_backend="ollama", gpu_choice="cpu"))._command("127.0.0.1", 11434)
    assert env == {"OLLAMA_HOST": "127.0.0.1:11434", "CUDA_VISIBLE_DEVICES": ""}


def test_ollama_flatpak_names_the_card_on_the_env_argument() -> None:
    # flatpak-spawn --host starts the host's ollama with the host's environment,
    # so the choice has to appear on the command line as well or it is lost.
    command, env = _manager(Settings(ai_backend="ollama", gpu_choice="gpu:1"), flatpak=True)._command(
        "127.0.0.1", 11434
    )
    assert command == [
        "flatpak-spawn",
        "--host",
        "--directory=/",
        "--watch-bus",
        "env",
        "OLLAMA_HOST=127.0.0.1:11434",
        "CUDA_VISIBLE_DEVICES=1",
        "ollama",
        "serve",
    ]
    assert env == {"OLLAMA_HOST": "127.0.0.1:11434", "CUDA_VISIBLE_DEVICES": "1"}


def test_ollama_flatpak_names_the_empty_card_list_for_processor_only() -> None:
    command, _env = _manager(Settings(ai_backend="ollama", gpu_choice="cpu"), flatpak=True)._command(
        "127.0.0.1", 11434
    )
    assert "CUDA_VISIBLE_DEVICES=" in command


def test_llamacpp_chosen_card_uses_its_device_option(monkeypatch) -> None:
    monkeypatch.setattr(engine, "bundled_server", lambda: "/app/bin/llama-server")
    command, env = _manager(
        Settings(ai_backend="llamacpp", llamacpp_model="m.gguf", gpu_choice="gpu:1")
    )._command("127.0.0.1", 8781)
    assert command[:2] == ["/app/bin/llama-server", "--model"]
    assert command[-2:] == ["--device", "cuda:1"]
    # CUDA_VISIBLE_DEVICES renumbers the cards a process can see, which would
    # make cuda:1 point at the wrong card, so llama.cpp is steered by its own
    # flag only.
    assert env is None


def test_llamacpp_processor_only_drops_the_layers_and_hides_the_cards(monkeypatch) -> None:
    monkeypatch.setattr(engine, "bundled_server", lambda: "/app/bin/llama-server")
    command, env = _manager(
        Settings(ai_backend="llamacpp", llamacpp_model="m.gguf", gpu_choice="cpu")
    )._command("127.0.0.1", 8781)
    assert "-ngl" in command and command[command.index("-ngl") + 1] == "0"
    assert "--device" not in command
    assert env == {"CUDA_VISIBLE_DEVICES": ""}


def test_llamacpp_automatic_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(engine, "bundled_server", lambda: "/app/bin/llama-server")
    command, env = _manager(Settings(ai_backend="llamacpp", llamacpp_model="m.gguf"))._command(
        "127.0.0.1", 8781
    )
    assert command[command.index("-ngl") + 1] == "99"
    assert "--device" not in command
    assert env is None


def test_host_llama_server_gets_the_choice_and_the_flatpak_env_argument(monkeypatch) -> None:
    monkeypatch.setattr(engine, "bundled_server", lambda: None)
    command, env = _manager(
        Settings(ai_backend="llamacpp", llamacpp_model="/models/m.gguf", gpu_choice="cpu"), flatpak=True
    )._command("127.0.0.1", 8781)
    assert command == [
        "flatpak-spawn",
        "--host",
        "--directory=/",
        "--watch-bus",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "llama-server",
        "--model",
        "/models/m.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8781",
    ]
    assert env == {"CUDA_VISIBLE_DEVICES": ""}


def test_server_command_defaults_to_todays_shape() -> None:
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


def test_server_command_for_each_choice() -> None:
    cpu = server_command("m.gguf", "127.0.0.1", 8781, server="/app/bin/llama-server", gpu_choice="cpu")
    assert cpu[cpu.index("-ngl") : cpu.index("-ngl") + 2] == ["-ngl", "0"]
    assert "--device" not in cpu

    card = server_command("m.gguf", "127.0.0.1", 8781, server="/app/bin/llama-server", gpu_choice="gpu:1")
    assert card[-2:] == ["--device", "cuda:1"]
    assert card[card.index("-ngl") + 1] == "99"

    # A junk choice behaves like "auto": nothing is added, nothing is removed.
    junk = server_command("m.gguf", "127.0.0.1", 8781, server="/app/bin/llama-server", gpu_choice="card 0")
    assert junk == server_command("m.gguf", "127.0.0.1", 8781, server="/app/bin/llama-server")


def test_the_status_line_names_the_device_in_use() -> None:
    # The wording lives in the picker module, which imports gi; where gi is not
    # installed (as on this machine) the check skips instead of passing silently
    # about nothing.
    pytest.importorskip("gi")
    from voxa.ui.model_selector import gpu_status_detail

    assert gpu_status_detail("gpu:1", (0, 1)) == "card 1"
    assert gpu_status_detail("cpu", (0, 1)) == "processor"
    assert gpu_status_detail("auto", (0, 1)) == ""
    assert gpu_status_detail("gpu:3", (0, 1)) == "card 3 (not in this machine)"
