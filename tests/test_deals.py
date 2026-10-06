from __future__ import annotations

from datetime import date

import voxa.agent.deals as deals
from voxa.agent.deals import build_url, describe, detect_location, parse_request
from voxa.agent.intents import ToolCall, route

TODAY = date(2026, 10, 6)
LOCATION = "Los Angeles"


def _reset_cache() -> None:
    deals._LOCATION_CACHE = ("", 0.0)


def test_parse_table() -> None:
    cases = [
        ("get me the cheapest ticket to Hawaii", "flight", "Hawaii", "", date(2026, 10, 20), date(2026, 10, 27), ""),
        ("cheapest flight to Paris on Kayak", "flight", "Paris", "", date(2026, 10, 20), date(2026, 10, 27), "kayak"),
        ("flight from Boston to Denver on October 20", "flight", "Denver", "Boston", date(2026, 10, 20), date(2026, 10, 27), ""),
        ("cheapest ticket to Tokyo returning November 5", "flight", "Tokyo", "", date(2026, 10, 20), date(2026, 11, 5), ""),
        ("one way flight to Vegas", "flight", "Vegas", "", date(2026, 10, 20), None, ""),
        ("fly to London next Friday", "flight", "London", "", date(2026, 10, 9), date(2026, 10, 16), ""),
        ("airfare to Rome in December", "flight", "Rome", "", date(2026, 12, 10), date(2026, 12, 17), ""),
        ("book a flight to Cancun tomorrow", "flight", "Cancun", "", date(2026, 10, 7), date(2026, 10, 14), ""),
        ("find me a cheap hotel in Tokyo for 4 nights", "hotel", "Tokyo", "", date(2026, 10, 20), date(2026, 10, 24), ""),
        ("hotel in Denver next week", "hotel", "Denver", "", date(2026, 10, 13), date(2026, 10, 16), ""),
        ("a place to stay in Paris on October 20 for 2 nights", "hotel", "Paris", "", date(2026, 10, 20), date(2026, 10, 22), ""),
        ("motel in Reno", "hotel", "Reno", "", date(2026, 10, 20), date(2026, 10, 23), ""),
        ("rent a car in Denver next week", "rental_car", "Denver", "", date(2026, 10, 13), date(2026, 10, 16), ""),
        ("rental car in Austin", "rental_car", "Austin", "", date(2026, 10, 20), date(2026, 10, 23), ""),
        ("car rental in Boise for 5 nights", "rental_car", "Boise", "", date(2026, 10, 20), date(2026, 10, 25), ""),
        ("cheapest used Toyota Tacoma under 20000", "car_for_sale", "used Toyota Tacoma", "", None, None, ""),
        ("used Ford F-150 for sale", "car_for_sale", "used Ford F-150", "", None, None, ""),
        ("buy a new Honda Civic under 15k", "car_for_sale", "new Honda Civic", "", None, None, ""),
        ("best price on a 4070 Ti Super", "product", "4070 Ti Super", "", None, None, ""),
        ("cheapest laptop on Amazon", "product", "laptop", "", None, None, "amazon"),
        ("a deal on a coffee maker on eBay", "product", "coffee maker", "", None, None, "ebay"),
        ("the lowest price for a standing desk", "product", "standing desk", "", None, None, ""),
    ]
    for text, kind, what, origin, depart, ret, site in cases:
        req = parse_request(text, TODAY)
        assert req is not None, text
        assert req.kind == kind, text
        assert req.what == what, text
        assert req.origin == origin, text
        assert req.depart == depart, text
        assert req.ret == ret, text
        assert req.site == site, text


def test_parse_none() -> None:
    for text in (
        "tell me about Hawaii",
        "play Hawaii Five-O",
        "get milk",
        "open Hawaii",
        "what is the capital of France",
        "search the web for cheap flights",
        "I'm in Portland",
        "my location is Portland",
        "use Portland as my location",
        "forget my location",
    ):
        assert parse_request(text, TODAY) is None, text


def test_max_price() -> None:
    req = parse_request("cheapest used Toyota Tacoma under 20000", TODAY)
    assert req is not None
    assert req.max_price == 20000
    small = parse_request("buy a new Honda Civic under 15k", TODAY)
    assert small is not None
    assert small.max_price == 15000


def test_exact_urls() -> None:
    expected = {
        "get me the cheapest ticket to Hawaii": (
            "https://www.expedia.com/Flights-Search?trip=roundtrip"
            "&leg1=from:Los%20Angeles,to:Hawaii,departure:10/20/2026TANYT"
            "&leg2=from:Hawaii,to:Los%20Angeles,departure:10/27/2026TANYT"
            "&passengers=adults:1&mode=search&sort=PRICE_INCREASING"
        ),
        "find me a cheap hotel in Tokyo for 4 nights": (
            "https://www.expedia.com/Hotel-Search?destination=Tokyo"
            "&startDate=2026-10-20&endDate=2026-10-24&adults=2&sort=PRICE_LOW_TO_HIGH"
        ),
        "rent a car in Denver next week": (
            "https://www.expedia.com/carsearch?locn=Denver"
            "&date1=10/13/2026&date2=10/16/2026&sort=price"
        ),
        "cheapest used Toyota Tacoma under 20000": (
            "https://www.cars.com/shopping/results/?keyword=used%20Toyota%20Tacoma"
            "&sort=list_price&zip=&maximum_distance=all&list_price_max=20000"
        ),
        "best price on a 4070 Ti Super": (
            "https://www.google.com/search?tbm=shop&q=4070%20Ti%20Super&tbs=p_ord:p"
        ),
        "cheapest flight to Paris on Kayak": (
            "https://www.google.com/travel/flights?q=Flights%20to%20Paris%20from%20Los%20Angeles"
            "%20on%202026-10-20%20through%202026-10-27"
        ),
    }
    for text, want in expected.items():
        req = parse_request(text, TODAY)
        assert req is not None, text
        url, _site = build_url(req, LOCATION)
        assert url == want, text


def test_oneway_url() -> None:
    req = parse_request("one way flight to Vegas", TODAY)
    assert req is not None
    url, _site = build_url(req, LOCATION)
    assert "trip=oneway" in url
    assert "leg2" not in url


def test_describe() -> None:
    flight = parse_request("get me the cheapest ticket to Hawaii", TODAY)
    assert flight is not None
    assert describe(flight, LOCATION) == (
        "Here are the cheapest flights from Los Angeles to Hawaii, "
        "leaving October 20, returning October 27, on Expedia."
    )
    hotel = parse_request("find me a cheap hotel in Tokyo for 4 nights", TODAY)
    assert hotel is not None
    assert describe(hotel, LOCATION) == (
        "Here are the cheapest hotels in Tokyo, checking in October 20 for 4 nights, on Expedia."
    )
    car = parse_request("rent a car in Denver next week", TODAY)
    assert car is not None
    assert describe(car, LOCATION) == (
        "Here are the cheapest rental cars in Denver, picking up October 13, on Expedia."
    )
    sale = parse_request("cheapest used Toyota Tacoma under 20000", TODAY)
    assert sale is not None
    assert describe(sale, LOCATION) == (
        "Here are the cheapest used Toyota Tacoma matches under 20000 dollars, on cars.com."
    )
    product = parse_request("best price on a 4070 Ti Super", TODAY)
    assert product is not None
    assert describe(product, LOCATION) == (
        "Here are the cheapest 4070 Ti Super results, on Google Shopping."
    )


def test_location_phrases_route() -> None:
    assert route("I'm in Portland") == ToolCall("set_location", {"city": "Portland"})
    assert route("my location is Portland") == ToolCall("set_location", {"city": "Portland"})
    assert route("use Portland as my location") == ToolCall("set_location", {"city": "Portland"})
    assert route("forget my location") == ToolCall("set_location", {"city": ""})


def test_deal_routes() -> None:
    assert route("get me the cheapest ticket to Hawaii") == ToolCall(
        "find_deal", {"request": "get me the cheapest ticket to Hawaii"}
    )
    assert route("best price on a 4070 Ti Super") == ToolCall(
        "find_deal", {"request": "best price on a 4070 Ti Super"}
    )


def test_detect_location_first_succeeds() -> None:
    _reset_cache()

    def fetch(url: str) -> str:
        return '{"city": "Los Angeles", "region": "California"}'

    assert detect_location(fetch) == "Los Angeles, California"


def test_detect_location_second_succeeds() -> None:
    _reset_cache()

    def fetch(url: str) -> str:
        if "ipapi.co" in url:
            raise OSError("down")
        return '{"city": "Denver", "region": "Colorado"}'

    assert detect_location(fetch) == "Denver, Colorado"


def test_detect_location_all_fail() -> None:
    _reset_cache()

    def fetch(url: str) -> str:
        raise OSError("down")

    assert detect_location(fetch) == ""


def test_detect_location_cache() -> None:
    _reset_cache()

    def good(url: str) -> str:
        return '{"city": "Boise", "region": "Idaho"}'

    assert detect_location(good) == "Boise, Idaho"

    def bad(url: str) -> str:
        raise OSError("down")

    assert detect_location(bad) == "Boise, Idaho"
