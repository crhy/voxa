from __future__ import annotations

import re

_MEGA = 1024**2
_GIGA = 1024**3
_TERA = 1024**4


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _part_of_day(hour: int, minute: int) -> str:
    if hour < 12:
        return "in the morning"
    if hour < 18:
        return "in the afternoon"
    if hour < 22:
        return "in the evening"
    return "at night"


def spoken_time(hour: int, minute: int) -> str:
    hour12 = hour % 12 or 12
    if minute == 0:
        if hour == 0:
            return "It's midnight."
        if hour == 12:
            return "It's noon."
        return f"It's {hour12} o'clock {_part_of_day(hour, minute)}."
    return f"It's {hour12}:{minute:02d} {_part_of_day(hour, minute)}."


def spoken_date(weekday: str, day: int, month: str, year: int) -> str:
    return f"It's {weekday}, the {ordinal(day)} of {month} {year}."


def _unit(value: float, name: str, decimals: int) -> str:
    if decimals == 0:
        num = str(int(round(value)))
    else:
        num = f"{value:.1f}"
        if num.endswith(".0"):
            num = num[:-2]
    unit = name if num != "1" else name[:-1]
    return f"{num} {unit}"


def human_size(n_bytes: int) -> str:
    if n_bytes >= _TERA:
        return _unit(n_bytes / _TERA, "terabytes", 1)
    if n_bytes >= _GIGA:
        value = n_bytes / _GIGA
        return _unit(value, "gigabytes", 1 if value < 10 else 0)
    if n_bytes >= _MEGA:
        return _unit(n_bytes / _MEGA, "megabytes", 0)
    return f"{n_bytes} bytes"


def spoken_disk(free_bytes: int, total_bytes: int) -> str:
    percent = round(free_bytes / total_bytes * 100) if total_bytes else 0
    return (
        f"You have {human_size(free_bytes)} free of {human_size(total_bytes)}, {percent} percent."
    )


def spoken_memory(available_bytes: int, total_bytes: int) -> str:
    return f"{human_size(available_bytes)} of memory are free, out of {human_size(total_bytes)}."


def spoken_battery(percent: int | None, state: str) -> str:
    if percent is None:
        return "This computer has no battery."
    if state == "fully-charged":
        return "The battery is full."
    if state == "charging":
        return f"The battery is at {percent} percent and charging."
    return f"The battery is at {percent} percent."


def parse_df(output: str) -> tuple[int, int] | None:
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1])
    return None


def parse_meminfo(text: str) -> tuple[int, int] | None:
    available = total = None
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        value = rest.strip().split()
        if not value:
            continue
        if key.strip() == "MemAvailable":
            available = int(value[0]) * 1024
        elif key.strip() == "MemTotal":
            total = int(value[0]) * 1024
    if available is None or total is None:
        return None
    return available, total


def parse_upower(text: str) -> tuple[int | None, str]:
    percent = None
    state = ""
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        key = key.strip()
        if key == "percentage":
            digits = "".join(ch for ch in rest if ch.isdigit())
            if digits:
                percent = int(digits)
        elif key == "state":
            state = rest.strip()
    return percent, state


def parse_gpu(text: str) -> tuple[int, int, int, int] | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 4:
        return None
    try:
        values = [int(part) for part in parts]
    except ValueError:
        return None
    return (values[0], values[1], values[2], values[3])


def parse_cpu_temp(text: str) -> int | None:
    for label in ("CPU Package", "Tctl", "Package id 0", "CPU"):
        for line in text.splitlines():
            key, _, rest = line.partition(":")
            if key.strip() == label:
                match = re.search(r"[-+]?\d+(?:\.\d+)?", rest)
                if match:
                    return round(float(match.group()))
    return None


def parse_load(loadavg: str, threads: str) -> int | None:
    try:
        load = float(loadavg.split()[0])
        count = int(threads.strip())
    except (ValueError, IndexError):
        return None
    if count <= 0:
        return None
    return min(round(load / count * 100), 100)


def spoken_gpu(gpu: tuple[int, int, int, int] | None) -> str:
    if gpu is None:
        return "I can't read a graphics card on this computer."
    temp, util, used, total = gpu
    used_gb = round(used / 1024)
    total_gb = round(total / 1024)
    return (
        f"Your graphics card is at {temp} degrees, {util} percent busy, "
        f"with {used_gb} of {total_gb} gigabytes of memory in use."
    )


def spoken_cpu(temp: int | None, load: int | None) -> str:
    if temp is None and load is None:
        return "I can't read the processor's temperature on this computer."
    if temp is None:
        return f"Your processor is {load} percent busy."
    if load is None:
        return f"Your processor is at {temp} degrees."
    return f"Your processor is at {temp} degrees and {load} percent busy."


def spoken_health(
    temp: int | None,
    load: int | None,
    gpu: tuple[int, int, int, int] | None,
    memory_sentence: str,
    disk_sentence: str,
) -> str:
    parts = [spoken_cpu(temp, load)]
    if gpu is not None:
        parts.append(spoken_gpu(gpu))
    parts.append(memory_sentence)
    parts.append(disk_sentence)
    text = " ".join(parts)
    gpu_temp = gpu[0] if gpu is not None else None
    if (temp is not None and temp >= 90) or (gpu_temp is not None and gpu_temp >= 90):
        text = "Something is running hot. " + text
    return text


def parse_ps(text: str, top: int = 3) -> list[tuple[str, float, float]]:
    summed: dict[str, list[float]] = {}
    order: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0]
        if name.startswith("["):
            continue
        try:
            cpu = float(parts[1])
            mem = float(parts[2])
        except ValueError:
            continue
        if name in summed:
            summed[name][0] += cpu
            summed[name][1] += mem
        else:
            summed[name] = [cpu, mem]
            order.append(name)
    rows = [(name, summed[name][0], summed[name][1]) for name in order]
    return rows[:top]


def _and_join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def spoken_top_cpu(rows: list[tuple[str, float, float]]) -> str:
    hot = sorted(((name, cpu) for name, cpu, _ in rows if cpu > 1.0), key=lambda r: -r[1])
    if not hot:
        return "Nothing is using much of the processor right now."
    parts = [f"{name} at {round(cpu)} percent" for name, cpu in hot]
    if len(parts) == 1:
        return f"The busiest program is {parts[0]}."
    return f"The busiest programs are {_and_join(parts)}."


def spoken_top_memory(rows: list[tuple[str, float, float]]) -> str:
    ranked = sorted(((name, mem) for name, _, mem in rows if mem > 0.0), key=lambda r: -r[1])
    if not ranked:
        return "Nothing is using much memory right now."
    parts = [f"{name} with {round(mem)} percent" for name, mem in ranked]
    if len(parts) == 1:
        return f"The most memory is used by {parts[0]}."
    return f"The most memory is used by {', then '.join(parts)}."


def find_problems(
    temp: int | None,
    load: int | None,
    gpu: tuple[int, int, int, int] | None,
    memory_available: int | None,
    memory_total: int | None,
    disk_free: int | None,
    disk_total: int | None,
) -> list[str]:
    problems: list[str] = []
    if temp is not None and temp >= 85:
        problems.append(f"the processor is hot at {temp} degrees")
    if gpu is not None and gpu[0] >= 85:
        problems.append(f"the graphics card is hot at {gpu[0]} degrees")
    if load is not None and load >= 90:
        problems.append("the processor is fully busy")
    if (
        memory_available is not None
        and memory_total is not None
        and memory_total > 0
        and memory_available / memory_total < 0.10
    ):
        problems.append("memory is almost full")
    if (
        disk_free is not None
        and disk_total is not None
        and disk_total > 0
        and disk_free / disk_total < 0.05
    ):
        pct = round(disk_free / disk_total * 100)
        problems.append(f"the disk is almost full ({pct} percent free)")
    return problems


def spoken_problems(problems: list[str]) -> str:
    if not problems:
        return "Everything looks normal."
    if len(problems) == 1:
        return f"I see one thing: {problems[0]}."
    joined = ", ".join(problems[:-1]) + ", and " + problems[-1]
    return f"I see {len(problems)} things: {joined}."
