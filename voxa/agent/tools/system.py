from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime

from voxa import hardware, simulation
from voxa.agent import worldtime
from voxa.agent.host import host_command
from voxa.agent.policy import RiskLevel
from voxa.agent.registry import Tool
from voxa.agent.result import ToolResult
from voxa.agent.sysinfo import (
    find_problems,
    parse_cpu_temp,
    parse_df,
    parse_gpu,
    parse_load,
    parse_meminfo,
    parse_ps,
    parse_upower,
    spoken_battery,
    spoken_cpu,
    spoken_date,
    spoken_disk,
    spoken_gpu,
    spoken_health,
    spoken_memory,
    spoken_problems,
    spoken_time,
    spoken_top_cpu,
    spoken_top_memory,
)
from voxa.config import ConfigStore

log = logging.getLogger("voxa.agent.tools.system")

_FAILURE = "I could not read that from this computer."


_SIMULATED = {
    ("date", "+%H %M"): "12 00\n",
    ("date", "+%A|%-d|%B|%Y"): "Monday|1|January|2024\n",
    ("df", "-B1", "--output=avail,size"): "        Avail         1B-blocks\n128849018880 536870912000\n",
    ("cat", "/proc/meminfo"): "MemTotal: 32000000 kB\nMemAvailable: 24000000 kB\n",
    ("upower", "-e"): "",
    (
        "nvidia-smi",
        "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ): "46, 4, 3454, 16376\n",
    ("sensors",): "CPU Package:  +67.0°C\n",
    ("cat", "/proc/loadavg"): "1.92 1.10 0.98 3/1234 5678\n",
    ("nproc",): "16\n",
    (
        "ps",
        "-eo",
        "comm,%cpu,%mem",
        "--sort=-%cpu",
        "--no-headers",
    ): "brave  20.5  12.0\nbrave  10.5  6.0\npython3  12.0  8.0\nXorg  4.0  2.0\n",
    (
        "ps",
        "-eo",
        "comm,%cpu,%mem",
        "--sort=-%mem",
        "--no-headers",
    ): "brave  18.0  12.0\npython3  9.0  8.0\n",
}


def _run(command: list[str], stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if simulation.actions_simulated():
        log.info("simulated action: would run %s", command)
        if command[:2] == ["sh", "-c"]:
            canned = "/home/me\n" if "$HOME" in command[2] else ""
        else:
            canned = _SIMULATED.get(tuple(command), _SIMULATED.get(tuple(command[:3]), ""))
        return subprocess.CompletedProcess(command, 0, canned, "")
    return subprocess.run(
        host_command(command),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=6,
        check=check,
    )


def tell_time(args: dict[str, str]) -> ToolResult:
    try:
        hour, minute = _run(["date", "+%H %M"], check=False).stdout.split()
        return ToolResult.success(spoken_time(int(hour), int(minute)))
    except Exception:
        return ToolResult.failure(_FAILURE)


def tell_date(args: dict[str, str]) -> ToolResult:
    try:
        weekday, day, month, year = _run(["date", "+%A|%-d|%B|%Y"], check=False).stdout.split("|")
        return ToolResult.success(spoken_date(weekday, int(day), month, int(year)))
    except Exception:
        return ToolResult.failure(_FAILURE)


def disk_space(args: dict[str, str]) -> ToolResult:
    try:
        home = _run(["sh", "-c", "echo $HOME"], check=False).stdout.strip()
        free, total = parse_df(_run(["df", "-B1", "--output=avail,size", home], check=False).stdout)
        return ToolResult.success(spoken_disk(free, total))
    except Exception:
        return ToolResult.failure(_FAILURE)


def memory_free(args: dict[str, str]) -> ToolResult:
    try:
        available, total = parse_meminfo(_run(["cat", "/proc/meminfo"], check=False).stdout)
        return ToolResult.success(spoken_memory(available, total))
    except Exception:
        return ToolResult.failure(_FAILURE)


def battery_level(args: dict[str, str]) -> ToolResult:
    try:
        found = _run(["sh", "-c", "upower -e | grep -m1 BAT"], check=False).stdout.strip()
        if not found:
            return ToolResult.success(spoken_battery(None, ""))
        percent, state = parse_upower(_run(["upower", "-i", found], check=False).stdout)
        return ToolResult.success(spoken_battery(percent, state))
    except Exception:
        return ToolResult.failure(_FAILURE)


_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def gpu_status(args: dict[str, str]) -> ToolResult:
    try:
        gpu = parse_gpu(_run(_GPU_QUERY, check=False).stdout)
        return ToolResult.success(f"{spoken_gpu(gpu)} {_brain_report()}")
    except Exception:
        return ToolResult.failure(_FAILURE)


# The machine a lab run pretends to have: two cards, the main one held by
# another workload, which is exactly the case crhy/voxa#113 is about. Without
# it a simulated run would report whatever card the developer's machine has,
# and the sentence would change from run to run.
_SIMULATED_DEVICES = [
    hardware.GpuDevice(
        index=0,
        vendor="nvidia",
        name="NVIDIA GeForce RTX 4090",
        total_vram_gb=24.0,
        used_vram_gb=22.0,
        utilization_percent=97.0,
        device_id="nvidia:0",
    ),
    hardware.GpuDevice(
        index=1,
        vendor="nvidia",
        name="NVIDIA GeForce RTX 3060",
        total_vram_gb=8.0,
        used_vram_gb=0.0,
        utilization_percent=0.0,
        device_id="nvidia:1",
    ),
]


def _brain_devices() -> list[hardware.GpuDevice]:
    if simulation.actions_simulated():
        return list(_SIMULATED_DEVICES)
    return hardware.detect_gpu_devices()


def _said(reason: str) -> str:
    """A reason from hardware.py starts lower-case; after a full stop it must not."""
    return reason[:1].upper() + reason[1:] if reason else ""


def _brain_report() -> str:
    """Which card Voxa's brain is on, and why, in one spoken sentence."""
    devices = _brain_devices()
    settings = ConfigStore().load()
    reserved = hardware.reserved_gpu_gb(settings.face_mode, True)
    busy = hardware.busy_devices(devices)
    launch, reason = hardware.resolve_brain_device(
        settings.gpu_device,
        devices,
        reserved_gb=reserved,
        busy=busy,
    )
    if not devices:
        return f"Voxa's brain runs on the processor: {reason}."
    if launch == "cpu":
        return f"Voxa keeps her brain on the processor: {reason}."
    chosen, _reason = hardware.pick_brain_device(devices, reserved_gb=reserved, busy=busy)
    saved = next((device for device in devices if device.device_id == launch), None)
    card = saved if saved is not None else chosen
    prefix = "Voxa is on" if saved is not None else "On the automatic setting Voxa takes"
    return f"{prefix} {hardware.describe_device(card)}: {reason}."


def gpu_choose(args: dict[str, str]) -> ToolResult:
    """Move Voxa's brain to another card, or to the processor, and say where it went.

    The choice is saved, not applied to a server that is already running: the
    reply says so rather than implying the card changed under Voxa's feet.
    """
    try:
        move = (args.get("move") or "other").strip().casefold()
        devices = _brain_devices()
        settings = ConfigStore().load()
        reserved = hardware.reserved_gpu_gb(settings.face_mode, True)
        if move == "cpu":
            wanted, where, reason = "cpu", "the processor", "you asked for the processor"
        elif move == "auto":
            wanted, where, reason = "auto", "whatever card the backend picks", "the choice is automatic again"
        else:
            named = hardware.parse_device_id(move)
            if named is not None:
                wanted = f"{named[0]}:{named[1]}"
                found = next((device for device in devices if device.device_id == wanted), None)
                if found is None:
                    return ToolResult.success(f"This machine has no card {wanted}, so Voxa stays where she is.")
                where, reason = hardware.describe_device(found), f"you named GPU {found.index}"
            else:
                # The card Voxa is on now is not "the other one", so it is left
                # out of the busy set even though her own model is loaded on it.
                busy = hardware.busy_devices(devices) - {settings.gpu_device}
                chosen, reason = hardware.pick_brain_device(devices, reserved_gb=reserved, busy=busy)
                if chosen is None:
                    return ToolResult.success(f"I can't move Voxa to another graphics card. {_said(reason)}.")
                wanted, where = chosen.device_id, hardware.describe_device(chosen)
        settings.gpu_device = wanted
        ConfigStore().save(settings)
        return ToolResult.success(
            f"Voxa will run her brain on {where} when her AI server starts next. {_said(reason)}. "
            "Until then the server she is using keeps the card it started on."
        )
    except Exception:
        return ToolResult.failure(_FAILURE)


def cpu_status(args: dict[str, str]) -> ToolResult:
    try:
        temp = parse_cpu_temp(_run(["sensors"], check=False).stdout)
        load = parse_load(
            _run(["cat", "/proc/loadavg"], check=False).stdout,
            _run(["nproc"], check=False).stdout,
        )
        return ToolResult.success(spoken_cpu(temp, load))
    except Exception:
        return ToolResult.failure(_FAILURE)


def system_health(args: dict[str, str]) -> ToolResult:
    try:
        temp = parse_cpu_temp(_run(["sensors"], check=False).stdout)
        load = parse_load(
            _run(["cat", "/proc/loadavg"], check=False).stdout,
            _run(["nproc"], check=False).stdout,
        )
        gpu = parse_gpu(_run(_GPU_QUERY, check=False).stdout)
        available, total = parse_meminfo(_run(["cat", "/proc/meminfo"], check=False).stdout)
        memory_sentence = spoken_memory(available, total)
        home = _run(["sh", "-c", "echo $HOME"], check=False).stdout.strip()
        free, total_disk = parse_df(
            _run(["df", "-B1", "--output=avail,size", home], check=False).stdout
        )
        disk_sentence = spoken_disk(free, total_disk)
        return ToolResult.success(spoken_health(temp, load, gpu, memory_sentence, disk_sentence))
    except Exception:
        return ToolResult.failure(_FAILURE)


_PS_FIELDS = ["ps", "-eo", "comm,%cpu,%mem"]


def top_cpu(args: dict[str, str]) -> ToolResult:
    try:
        out = _run(_PS_FIELDS + ["--sort=-%cpu", "--no-headers"], check=False).stdout
        return ToolResult.success(spoken_top_cpu(parse_ps(out)))
    except Exception:
        return ToolResult.failure(_FAILURE)


def top_memory(args: dict[str, str]) -> ToolResult:
    try:
        out = _run(_PS_FIELDS + ["--sort=-%mem", "--no-headers"], check=False).stdout
        return ToolResult.success(spoken_top_memory(parse_ps(out)))
    except Exception:
        return ToolResult.failure(_FAILURE)


def system_problems(args: dict[str, str]) -> ToolResult:
    try:
        temp = parse_cpu_temp(_run(["sensors"], check=False).stdout)
        load = parse_load(
            _run(["cat", "/proc/loadavg"], check=False).stdout,
            _run(["nproc"], check=False).stdout,
        )
        gpu = parse_gpu(_run(_GPU_QUERY, check=False).stdout)
        memory = parse_meminfo(_run(["cat", "/proc/meminfo"], check=False).stdout)
        home = _run(["sh", "-c", "echo $HOME"], check=False).stdout.strip()
        disk = parse_df(_run(["df", "-B1", "--output=avail,size", home], check=False).stdout)
        problems = find_problems(
            temp,
            load,
            gpu,
            memory[0] if memory else None,
            memory[1] if memory else None,
            disk[0] if disk else None,
            disk[1] if disk else None,
        )
        return ToolResult.success(spoken_problems(problems))
    except Exception:
        return ToolResult.failure(_FAILURE)


def world_time(args: dict[str, str]) -> ToolResult:
    place = args["place"]
    if worldtime.zone_for(place) is None:
        return ToolResult.failure(f"I don't know the time zone for {place}.")
    now = datetime.now(UTC)
    local_weekday = datetime.now().strftime("%A")
    return ToolResult.success(worldtime.spoken_time_in(place, now, local_weekday))


def time_difference(args: dict[str, str]) -> ToolResult:
    place = args["place"]
    if worldtime.zone_for(place) is None:
        return ToolResult.failure(f"I don't know the time zone for {place}.")
    now = datetime.now(UTC)
    local_offset = datetime.now().astimezone().utcoffset().total_seconds() / 3600
    return ToolResult.success(worldtime.spoken_difference(place, now, local_offset))


def system_tools() -> list[Tool]:
    return [
        Tool(
            name="tell_time",
            description="Say the current time on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=tell_time,
        ),
        Tool(
            name="tell_date",
            description="Say today's date on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=tell_date,
        ),
        Tool(
            name="disk_space",
            description="Say how much disk space is free on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=disk_space,
        ),
        Tool(
            name="memory_free",
            description="Say how much memory is free on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=memory_free,
        ),
        Tool(
            name="battery_level",
            description="Say how much battery is left on this computer.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=battery_level,
        ),
        Tool(
            name="gpu_status",
            description="Say how hot the graphics card is, how busy it is, and which card Voxa's brain runs on.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=gpu_status,
        ),
        Tool(
            name="gpu_choose",
            description="Put Voxa's brain on another graphics card, on a card you name, or back on the processor.",
            parameters={"move": "other|cpu|auto|<vendor>:<index>"},
            risk=RiskLevel.REVERSIBLE,
            handler=gpu_choose,
        ),
        Tool(
            name="cpu_status",
            description="Say how hot the processor is and how busy it is.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=cpu_status,
        ),
        Tool(
            name="system_health",
            description="Say how hot and busy this computer is, with memory and disk.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=system_health,
        ),
        Tool(
            name="top_cpu",
            description="Name the programs using the most processor right now.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=top_cpu,
        ),
        Tool(
            name="top_memory",
            description="Name the programs using the most memory right now.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=top_memory,
        ),
        Tool(
            name="system_problems",
            description="Look for trouble: hot or busy processor, low memory, full disk.",
            parameters={},
            risk=RiskLevel.READ_ONLY,
            handler=system_problems,
        ),
        Tool(
            name="world_time",
            description="Say what time it is right now in another place.",
            parameters={"place": "the place to look up"},
            risk=RiskLevel.READ_ONLY,
            handler=world_time,
            required=("place",),
        ),
        Tool(
            name="time_difference",
            description="Say how many hours ahead or behind another place is from you.",
            parameters={"place": "the place to compare with"},
            risk=RiskLevel.READ_ONLY,
            handler=time_difference,
            required=("place",),
        ),
    ]
