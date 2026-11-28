"""#101: local weather from Open-Meteo, local news headlines, and the once-a-day location.

Recorded fixture responses only: no test reaches the network.
"""

from datetime import date

import pytest

from voxa.agent.intents import ToolCall, route
from voxa.agent.tools import local
from voxa.config import ConfigStore

# geocoding-api.open-meteo.com/v1/search?name=Portland%2C%20Oregon&count=1
GEO_PORTLAND = """{"results":[{"id":5742275,"name":"Portland","latitude":45.5234,
"longitude":-122.6763,"country":"United States of America","country_code":"US","admin1":"Oregon"}]}"""

# api.open-meteo.com/v1/forecast?...&temperature_unit=fahrenheit
FORECAST_PORTLAND = """{"latitude":45.52,"longitude":-122.68,"timezone":"America/Los_Angeles",
"current":{"time":"2026-11-28T10:00","temperature_2m":64.2,"weather_code":2},
"daily":{"time":["2026-11-28","2026-11-29"],"weather_code":[3,61],
"temperature_2m_max":[71.4,68.1],"temperature_2m_min":[54.3,51.8]}}"""

GEO_PARIS = """{"results":[{"id":2988507,"name":"Paris","latitude":48.8534,
"longitude":2.3488,"country":"France","country_code":"FR","admin1":"Île-de-France"}]}"""

FORECAST_PARIS = """{"latitude":48.85,"longitude":2.35,"timezone":"Europe/Paris",
"current":{"time":"2026-11-28T11:00","temperature_2m":9.5,"weather_code":3},
"daily":{"time":["2026-11-28","2026-11-29"],"weather_code":[80,0],
"temperature_2m_max":[11.2,8.6],"temperature_2m_min":[5.4,3.2]}}"""

# news.google.com/rss/search?q=Portland%20news (the fourth item must never be read)
RSS_PORTLAND = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>"Portland news" - Google News</title>
<link>https://news.google.com/search?q=Portland+news</link>
<item><title>City council votes on the riverfront plan - KGW</title>
<link>https://news.google.com/rss/articles/one</link><pubDate>Sat, 28 Nov 2026 09:00:00 GMT</pubDate></item>
<item><title>Winter storm warning issued for the Columbia Gorge - OregonLive</title>
<link>https://news.google.com/rss/articles/two</link><pubDate>Sat, 28 Nov 2026 08:30:00 GMT</pubDate></item>
<item><title>MAX Yellow Line returns after signal work - The Oregonian</title>
<link>https://news.google.com/rss/articles/three</link><pubDate>Sat, 28 Nov 2026 07:15:00 GMT</pubDate></item>
<item><title>This fourth story must not be read aloud - KOIN</title>
<link>https://news.google.com/rss/articles/four</link><pubDate>Sat, 28 Nov 2026 06:00:00 GMT</pubDate></item>
</channel></rss>"""


@pytest.fixture
def stub(monkeypatch):
    """Stubbed voice location and fetch; only browser opens land in `opened`."""
    opened, fetched = [], []

    def stub_locate(value):
        monkeypatch.setattr(local, "locate", lambda: value)

    def stub_fetch(handler):
        def get(url):
            fetched.append(url)
            return handler(url)

        monkeypatch.setattr(local, "fetch", get)

    monkeypatch.setattr(local.host, "spawn", lambda command: opened.append(command[-1]) or True)
    return stub_locate, stub_fetch, opened, fetched


def _recorded(url: str) -> str:
    if "geocoding-api.open-meteo.com" in url:
        return GEO_PORTLAND
    if "api.open-meteo.com" in url:
        return FORECAST_PORTLAND
    if "news.google.com/rss" in url:
        return RSS_PORTLAND
    raise AssertionError(f"unexpected url: {url}")


def test_local_weather_speaks_now_today_and_tomorrow(stub):
    locate, fetch, opened, fetched = stub
    locate("Portland, Oregon")
    fetch(_recorded)
    result = local.local_weather({})
    assert result.ok
    assert result.speech == (
        "In Portland it is 64 degrees and partly cloudy. "
        "Today a high of 71 and a low of 54 with cloudy. "
        "Tomorrow a high of 68 and a low of 52 with light rain."
    )
    assert result.speech.count(".") == 3  # one or two sentences, not a paragraph
    forecast = next(url for url in fetched if "v1/forecast" in url)
    assert "latitude=45.5234" in forecast and "temperature_unit=fahrenheit" in forecast
    assert opened == []  # the forecast answers on the exchange card: nothing to open


def test_local_weather_uses_celsius_where_the_us_does_not(stub):
    locate, fetch, opened, fetched = stub
    locate("Paris")
    fetch(lambda url: GEO_PARIS if "geocoding" in url else FORECAST_PARIS)
    result = local.local_weather({})
    assert result.ok and "Paris" in result.speech
    forecast = next(url for url in fetched if "v1/forecast" in url)
    assert "temperature_unit=celsius" in forecast
    assert "In Paris it is 10 degrees and cloudy." in result.speech


def test_typed_coordinates_skip_the_geocoder(stub):
    _locate, fetch, opened, fetched = stub
    fetch(lambda _url: FORECAST_PORTLAND)
    result = local.local_weather({"place": "45.52, -122.68"})
    assert result.ok
    assert "latitude=45.52" in fetched[-1] and "longitude=-122.68" in fetched[-1]
    assert "temperature_unit=fahrenheit" in fetched[-1]  # inside the lower 48
    assert not any("geocoding" in url for url in fetched)
    assert opened == []


def test_weather_falls_back_to_the_page_when_open_meteo_is_down(stub):
    locate, fetch, opened, _fetched = stub
    locate("Portland, Oregon")

    def down(_url):
        raise OSError("no network")

    fetch(down)
    result = local.local_weather({})
    assert result.ok and result.speech == "Here is the weather for Portland."
    assert "weather%20Portland%2C%20Oregon" in opened[-1]


def test_local_news_reads_the_first_three_headlines(stub):
    locate, fetch, opened, fetched = stub
    locate("Portland, Oregon")
    fetch(_recorded)
    result = local.local_news({})
    assert result.ok
    assert "City council votes on the riverfront plan - KGW" in result.speech
    assert "Winter storm warning issued for the Columbia Gorge - OregonLive" in result.speech
    assert "MAX Yellow Line returns after signal work - The Oregonian" in result.speech
    assert "fourth story" not in result.speech
    assert opened == []  # read, not opened
    assert fetched[-1] == "https://news.google.com/rss/search?q=Portland%20news"


def test_open_the_news_opens_the_page_without_reading_it(stub):
    locate, fetch, opened, _fetched = stub
    locate("Portland, Oregon")

    def no_fetch(_url):
        raise AssertionError("open the news must not fetch anything")

    fetch(no_fetch)
    result = local.local_news({"page": "1"})
    assert result.ok
    assert result.detail == "https://news.google.com/search?q=Portland%20news"
    assert opened[-1] == result.detail


def test_news_falls_back_to_the_page_when_the_feed_is_down(stub):
    locate, fetch, opened, _fetched = stub
    locate("Portland, Oregon")
    fetch(lambda _url: "not xml at all")
    result = local.local_news({})
    assert result.ok and result.speech == "Here is the local news for Portland."
    assert "news.google.com" in opened[-1]


def test_location_is_worked_out_once_a_day_and_kept(monkeypatch, tmp_path):
    store = tmp_path / "config.json"
    monkeypatch.setattr(local, "ConfigStore", lambda: ConfigStore(store))
    calls = []

    def detect():
        calls.append(1)
        return "Boise, Idaho"

    monkeypatch.setattr(local, "detect_location", detect)
    assert local.where() == "Boise, Idaho"
    assert local.where() == "Boise, Idaho"
    assert len(calls) == 1  # the answer is cached in the settings for the day
    saved = ConfigStore(store).load()
    assert saved.location_detected == "Boise, Idaho"
    assert saved.location_detected_on == date.today().isoformat()
    # A stale cache is worked out again.
    stale = ConfigStore(store).load()
    stale.location_detected_on = "2025-01-01"
    ConfigStore(store).save(stale)
    assert local.where() == "Boise, Idaho"
    assert len(calls) == 2


def test_where_am_i_answers_the_place(stub):
    locate, fetch, opened, fetched = stub
    locate("Portland, Oregon")
    fetch(_recorded)
    assert local.where_am_i({}).speech == "You are in Portland, Oregon."


def test_news_page_phrases_route():
    assert route("open the news") == ToolCall("local_news", {"page": "1"})
    assert route("open the news page") == ToolCall("local_news", {"page": "1"})
    assert route("show the news") == ToolCall("local_news", {"page": "1"})
    # Still read aloud, not opened:
    assert route("play the local news") == ToolCall("local_news", {})
