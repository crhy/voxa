from __future__ import annotations

from voxa.agent.intents import route
from voxa.agent.sysinfo import (
    parse_cpu_temp,
    parse_gpu,
    parse_load,
    spoken_cpu,
    spoken_gpu,
    spoken_health,
)
from voxa.agent.tools import default_registry, system


def test_parse_gpu():
    assert parse_gpu("46, 4, 3454, 16376\n") == (46, 4, 3454, 16376)
    assert parse_gpu("") is None
    assert parse_gpu("not, a, gpu, line") is None
    assert parse_gpu("46, 4, 3454\n") is None


def test_parse_cpu_temp():
    assert parse_cpu_temp("CPU Package:  +67.0°C\n") == 67
    assert parse_cpu_temp("Tctl:         +58.9°C\n") == 59
    assert parse_cpu_temp("CPU:          +56.0°C\n") == 56
    assert parse_cpu_temp("CPU Package:  +67.0°C\nTctl: +90.0°C\n") == 67
    assert parse_cpu_temp("no temperature here\n") is None


def test_parse_load():
    assert parse_load("1.92 1.10 0.98 3/1234 5678\n", "16\n") == 12
    assert parse_load("16.0 1.0 1.0 1/1 1", "16") == 100
    assert parse_load("junk", "16") is None
    assert parse_load("1.0 1.0 1.0", "0") is None


def test_spoken_gpu():
    assert spoken_gpu((46, 4, 3454, 16376)) == (
        "Your graphics card is at 46 degrees, 4 percent busy, "
        "with 3 of 16 gigabytes of memory in use."
    )
    assert spoken_gpu(None) == "I can't read a graphics card on this computer."


def test_spoken_cpu():
    assert spoken_cpu(67, 12) == "Your processor is at 67 degrees and 12 percent busy."
    assert spoken_cpu(67, None) == "Your processor is at 67 degrees."
    assert spoken_cpu(None, 12) == "Your processor is 12 percent busy."
    assert spoken_cpu(None, None) == "I can't read the processor's temperature on this computer."


def test_spoken_health():
    text = spoken_health(67, 12, (46, 4, 3454, 16376), "mem.", "disk.")
    assert text == (
        "Your processor is at 67 degrees and 12 percent busy. "
        "Your graphics card is at 46 degrees, 4 percent busy, "
        "with 3 of 16 gigabytes of memory in use. mem. disk."
    )
    hot = spoken_health(95, None, None, "mem.", "disk.")
    assert hot.startswith("Something is running hot. ")
    assert hot == "Something is running hot. Your processor is at 95 degrees. mem. disk."
    gpu_hot = spoken_health(None, None, (120, 0, 0, 1), "mem.", "disk.")
    assert gpu_hot.startswith("Something is running hot. ")


def test_router_phrases():
    assert route("gpu temperature").tool == "gpu_status"
    assert route("cpu usage").tool == "cpu_status"
    assert route("health check").tool == "system_health"
    assert route("how much memory is free").tool == "memory_free"


def test_registry_lists_new_tools():
    names = default_registry().names()
    assert "gpu_status" in names
    assert "cpu_status" in names
    assert "system_health" in names


def test_gpu_status_simulated(monkeypatch):
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    result = system.gpu_status({})
    assert result.ok
    assert "46 degrees" in result.speech


def test_cpu_status_simulated(monkeypatch):
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    result = system.cpu_status({})
    assert result.ok
    assert "67 degrees" in result.speech


def test_system_health_simulated(monkeypatch):
    monkeypatch.setenv("VOXA_SIMULATE_ACTIONS", "1")
    result = system.system_health({})
    assert result.ok
    assert "46 degrees" in result.speech
    assert "67 degrees" in result.speech
