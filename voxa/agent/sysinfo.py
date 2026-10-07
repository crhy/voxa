from __future__ import annotations

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
