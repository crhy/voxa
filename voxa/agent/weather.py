"""Open-Meteo (https://open-meteo.com/): the weather without an API key (issue #101).

Geocoding turns a typed city into coordinates; the forecast answers the current
hour, today and tomorrow in one call. A typed "latitude, longitude" skips the
geocoding step. Parsing is pure so the tests can feed recorded answers.
"""

from __future__ import annotations

import json
import re
import urllib.parse

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather icons shared by the current hour and the daily arrays.
CODES = {
    0: "clear",
    1: "clear",
    2: "partly cloudy",
    3: "cloudy",
    45: "fog",
    48: "fog",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow",
    80: "showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm",
    99: "a thunderstorm",
}

_LATLON = re.compile(r"^\s*(-?\d{1,2}(?:\.\d+)?)[, ]+(-?\d{1,3}(?:\.\d+)?)\s*$")


def latlon(place: str) -> tuple[float, float] | None:
    """The pair of numbers the user can type as their location, if that is what this is."""
    match = _LATLON.match(place)
    if not match:
        return None
    lat, lon = float(match.group(1)), float(match.group(2))
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return lat, lon


def in_us(lat: float, lon: float) -> bool:
    """Fahrenheit for latitudes and longitudes inside the lower 48."""
    return 24.0 <= lat <= 49.5 and -125.0 <= lon <= -66.5


def geocode_url(place: str) -> str:
    return f"{GEOCODE_URL}?name={urllib.parse.quote(place)}&count=1&language=English&format=json"


def parse_geocode(json_text: str) -> tuple[float, float, bool] | None:
    """(latitude, longitude, fahrenheit) for the first hit; None when the name is unknown."""
    results = json.loads(json_text).get("results") or []
    if not results:
        return None
    hit = results[0]
    fahrenheit = "united states" in str(hit.get("country") or "").lower()
    return float(hit["latitude"]), float(hit["longitude"]), fahrenheit


def forecast_url(lat: float, lon: float, fahrenheit: bool) -> str:
    return (
        f"{FORECAST_URL}?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,weather_code"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min"
        f"&forecast_days=2&timezone=auto&temperature_unit={'fahrenheit' if fahrenheit else 'celsius'}"
    )


def describe(code: int) -> str:
    return CODES.get(int(code), "unusual weather")


def parse_forecast(json_text: str, place: str) -> str:
    """One or two spoken sentences: now, today and tomorrow; "" on anything unexpected."""
    try:
        data = json.loads(json_text)
        current = data["current"]
        daily = data["daily"]
        highs = [int(round(float(value))) for value in daily["temperature_2m_max"]]
        lows = [int(round(float(value))) for value in daily["temperature_2m_min"][: len(highs)]]
        codes = [describe(value) for value in daily["weather_code"][: len(highs)]]
        now = f"In {place} it is {int(round(float(current['temperature_2m'])))} degrees and {describe(current['weather_code'])}."
        today = f"Today a high of {highs[0]} and a low of {lows[0]} with {codes[0]}."
        if len(highs) > 1:
            today = f"{today} Tomorrow a high of {highs[1]} and a low of {lows[1]} with {codes[1]}."
        return f"{now} {today}"
    except (ValueError, KeyError, IndexError, TypeError):
        return ""
