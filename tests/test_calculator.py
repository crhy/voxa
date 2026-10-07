"""Tests for the instant calculator (CL1, issue #69)."""

from __future__ import annotations

from voxa.agent import calc, intents


def test_parse_number():
    table = {
        "532": 532.0,
        "1,200": 1200.0,
        "3.5": 3.5,
        "minus 4": -4.0,
        "-4": -4.0,
        "zero": 0.0,
        "nineteen": 19.0,
        "twenty": 20.0,
        "twenty one": 21.0,
        "a hundred": 100.0,
        "three hundred and five": 305.0,
        "two thousand": 2000.0,
        "one million": 1_000_000.0,
        "a half": 0.5,
        "a quarter": 0.25,
        "three point five": 3.5,
    }
    for text, expected in table.items():
        assert calc.parse_number(text) == expected
    for not_number in ("", "hello", "the number", "twelve apples"):
        assert calc.parse_number(not_number) is None


def test_evaluate():
    cases = {
        "532 plus 789": 1321.0,
        "what's 12 times 12": 144.0,
        "2 plus 3 times 4": 14.0,
        "15 percent of 80": 12.0,
        "20 percent off 50": 40.0,
        "square root of 144": 12.0,
        "5 squared": 25.0,
        "two to the power of ten": 1024.0,
        "half of 90": 45.0,
        "seven minus ten": -3.0,
        "3.5 + 1.5": 5.0,
    }
    for text, expected in cases.items():
        assert calc.evaluate(text) == expected
    assert abs(calc.evaluate("100 divided by 7") - 100 / 7) < 1e-6
    assert calc.evaluate("10 divided by 0") is None


def test_evaluate_non_arithmetic():
    for text in (
        "what is 2 plus 2 in binary",
        "what's 9 11",
        "play 1999 by Prince",
        "set a timer for 5 minutes",
        "what is 1 plus the number of moons",
        "42",
        "what is 7",
    ):
        assert calc.evaluate(text) is None


def test_convert():
    value, unit = calc.convert("how many miles is 10 kilometers")
    assert round(value, 2) == 6.21 and unit == "miles"
    value, unit = calc.convert("72 fahrenheit to celsius")
    assert round(value, 2) == 22.22 and unit == "degrees Celsius"
    value, unit = calc.convert("0 celsius to fahrenheit")
    assert round(value, 2) == 32.0 and unit == "degrees Fahrenheit"
    value, unit = calc.convert("5 feet to meters")
    assert round(value, 2) == 1.52 and unit == "meters"
    value, unit = calc.convert("20 pounds to kilograms")
    assert round(value, 2) == 9.07 and unit == "kilograms"
    assert calc.convert("10 km in kilograms") is None


def test_spoken_number():
    table = {
        1321.0: "1,321",
        2.5: "2.5",
        14.29: "14.29",
        100.0: "100",
        0.0: "0",
        1e16: "1e+16",
        0.005: "0.005",
    }
    for value, expected in table.items():
        assert calc.spoken_number(value) == expected


def test_answer():
    assert calc.answer("532 plus 789") == "532 plus 789 is 1,321."
    assert calc.answer("what's 12 times 12") == "12 times 12 is 144."
    assert calc.answer("how many miles is 10 kilometers") == "6.21 miles."
    assert calc.answer("10 divided by 0") is None
    assert calc.answer("play 1999 by Prince") is None


def test_router():
    for text in ("532 plus 789", "what's 12 times 12", "15 percent of 80"):
        call = intents.route(text)
        assert call is not None and call.tool == "calculate"
    for text in ("how many miles is 10 kilometers", "72 fahrenheit to celsius"):
        call = intents.route(text)
        assert call is not None and call.tool == "calculate"
    for text in (
        "what is 2 plus 2 in binary",
        "what's 9 11",
        "play 1999 by Prince",
        "set a timer for 5 minutes",
        "what is 1 plus the number of moons",
    ):
        call = intents.route(text)
        assert call is None or call.tool != "calculate"
