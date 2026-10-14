from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from voxa.agent.sysinfo import spoken_time

ZONES: dict[str, str] = {
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "london": "Europe/London",
    "uk": "Europe/London",
    "england": "Europe/London",
    "paris": "Europe/Paris",
    "france": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "germany": "Europe/Berlin",
    "madrid": "Europe/Madrid",
    "spain": "Europe/Madrid",
    "rome": "Europe/Rome",
    "italy": "Europe/Rome",
    "amsterdam": "Europe/Amsterdam",
    "moscow": "Europe/Moscow",
    "dubai": "Asia/Dubai",
    "india": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "beijing": "Asia/Shanghai",
    "china": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong",
    "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "australia": "Australia/Sydney",
    "auckland": "Pacific/Auckland",
    "new zealand": "Pacific/Auckland",
    "new york": "America/New_York",
    "boston": "America/New_York",
    "miami": "America/New_York",
    "chicago": "America/Chicago",
    "dallas": "America/Chicago",
    "denver": "America/Denver",
    "phoenix": "America/Phoenix",
    "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "san diego": "America/Los_Angeles",
    "honolulu": "Pacific/Honolulu",
    "hawaii": "Pacific/Honolulu",
    "anchorage": "America/Anchorage",
    "toronto": "America/Toronto",
    "vancouver": "America/Vancouver",
    "mexico city": "America/Mexico_City",
    "sao paulo": "America/Sao_Paulo",
    "brazil": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
    "cairo": "Africa/Cairo",
    "johannesburg": "Africa/Johannesburg",
    "lagos": "Africa/Lagos",
    "nairobi": "Africa/Nairobi",
    "istanbul": "Europe/Istanbul",
    "seoul": "Asia/Seoul",
    "bangkok": "Asia/Bangkok",
    "jakarta": "Asia/Jakarta",
    "manila": "Asia/Manila",
    "utc": "UTC",
}


def zone_for(place: str) -> str | None:
    key = place.strip().lower()
    if key.startswith("the "):
        key = key[len("the "):]
    return ZONES.get(key)


def time_in(place: str, now_utc: datetime) -> tuple[int, int, str] | None:
    zone = zone_for(place)
    if zone is None:
        return None
    local = now_utc.astimezone(ZoneInfo(zone))
    return local.hour, local.minute, local.strftime("%A")


def _title_case(place: str) -> str:
    return place.strip().title()


def spoken_time_in(place: str, now_utc: datetime, local_weekday: str) -> str | None:
    found = time_in(place, now_utc)
    if found is None:
        return None
    hour, minute, weekday = found
    phrase = spoken_time(hour, minute)
    inner = phrase[len("It's "):-1]
    suffix = f" on {weekday}" if weekday != local_weekday else ""
    return f"In {_title_case(place)} it's {inner}{suffix}."


def difference_hours(place: str, now_utc: datetime, local_offset_hours: float) -> float | None:
    zone = zone_for(place)
    if zone is None:
        return None
    local = now_utc.astimezone(ZoneInfo(zone))
    offset = local.utcoffset().total_seconds() / 3600
    return offset - local_offset_hours


def _spoken_hours(value: float) -> str:
    whole = int(value)
    if value - whole >= 0.25 and value - whole <= 0.75:
        if whole == 0:
            return "half an hour"
        return f"{whole} and a half hours"
    if whole == 1:
        return "1 hour"
    return f"{whole} hours"


def spoken_difference(place: str, now_utc: datetime, local_offset_hours: float) -> str | None:
    diff = difference_hours(place, now_utc, local_offset_hours)
    if diff is None:
        return None
    title = _title_case(place)
    if diff == 0:
        return f"{title} is on the same time as you."
    hours = _spoken_hours(abs(diff))
    if diff > 0:
        return f"{title} is {hours} ahead of you."
    return f"{title} is {hours} behind you."
